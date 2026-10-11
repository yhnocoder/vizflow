export type GemmType = 'normal' | 'm_grouped_contiguous' | 'm_grouped_masked' | 'm_grouped_psum';

export interface GemmShape {
  m: number;
  n: number;
  k: number;
  blockM: number;
  blockN: number;
  blockK: number;
  numSms: number;
  numMulticast: 1 | 2;
  multicastOnA: boolean;
}

export type ScheduleOptions = GemmShape &
  (
    | { type: 'normal' }
    | { type: 'm_grouped_contiguous'; groupIdxPerRow: readonly number[] }
    | { type: 'm_grouped_masked'; validMPerGroup: readonly number[] }
    | { type: 'm_grouped_psum'; psumMPerGroup: readonly number[] }
  );

export interface Tile {
  sm: number;
  iter: number;
  mBlock: number;
  nBlock: number;
  group: number;
  multicast: boolean;
}

export const WGMMA_M = 64;
export const WGMMA_K = 32;

const ceilDiv = (a: number, b: number) => Math.floor((a + b - 1) / b);
const align = (a: number, b: number) => ceilDiv(a, b) * b;

export function numBlocksPerGroup(blockM: number, blockN: number, numSms: number, multicastOnA: boolean): number {
  let best = 0;
  let minUsage = Infinity;
  for (const candidate of [8, 16]) {
    const usage = multicastOnA
      ? candidate * blockN + ceilDiv(numSms, candidate) * blockM
      : candidate * blockM + ceilDiv(numSms, candidate) * blockN;
    if (usage < minUsage) {
      minUsage = usage;
      best = candidate;
    }
  }
  return best;
}

function numGroupsOf(opts: ScheduleOptions): number {
  if (opts.type === 'm_grouped_masked') return opts.validMPerGroup.length;
  if (opts.type === 'm_grouped_psum') return opts.psumMPerGroup.length;
  return 1;
}

function scheduleSm(opts: ScheduleOptions, sm: number): Tile[] {
  const { blockM, blockN, numSms, numMulticast, multicastOnA } = opts;
  const blocksPerGroup = numBlocksPerGroup(blockM, blockN, numSms, multicastOnA);
  const numGroups = numGroupsOf(opts);
  let numMBlocks = ceilDiv(opts.m, blockM);
  const numNBlocks = ceilDiv(opts.n, blockN);
  const numBlocks = numMBlocks * numNBlocks;
  let currentIter = -1;
  let currentGroupIdx = 0;
  let currentMCumsum = 0;
  let lastPsumM = 0;
  let currentPsumM = 0;
  let currentMBlockCumsum = 0;
  let numBlocksInGroup = 0;
  if (opts.type === 'm_grouped_psum') {
    currentPsumM = opts.psumMPerGroup[0];
    numMBlocks = ceilDiv(currentPsumM, blockM);
  }

  const swizzle = (blockIdx: number): [number, number] => {
    const primary = multicastOnA ? numNBlocks : numMBlocks;
    const secondary = multicastOnA ? numMBlocks : numNBlocks;
    const perGroup = secondary * blocksPerGroup;
    const groupIdx = Math.floor(blockIdx / perGroup);
    let firstBlockIdx = groupIdx * blocksPerGroup;
    let inGroupIdx = blockIdx % perGroup;
    numBlocksInGroup = Math.min(blocksPerGroup, primary - firstBlockIdx);
    if (numMulticast > 1 && numBlocksInGroup % 2 !== 0) {
      const even = numBlocksInGroup ^ 1;
      if (inGroupIdx < even * secondary) {
        numBlocksInGroup = even;
      } else {
        inGroupIdx -= even * secondary;
        firstBlockIdx += even;
        numBlocksInGroup = 1;
      }
    }
    const major = Math.floor(inGroupIdx / numBlocksInGroup);
    const minor = firstBlockIdx + (inGroupIdx % numBlocksInGroup);
    return multicastOnA ? [major, minor] : [minor, major];
  };

  const nextBlock = (): [number, number] | null => {
    currentIter += 1;
    const nextBlockIdx = currentIter * numSms + sm;
    if (opts.type === 'm_grouped_masked') {
      for (;;) {
        if (currentGroupIdx === numGroups) return null;
        numMBlocks = ceilDiv(opts.validMPerGroup[currentGroupIdx], blockM);
        const cumsum = currentMCumsum + numMBlocks;
        if (nextBlockIdx < cumsum * numNBlocks) break;
        currentGroupIdx += 1;
        currentMCumsum = cumsum;
      }
      return swizzle(nextBlockIdx - currentMCumsum * numNBlocks);
    }
    if (opts.type === 'm_grouped_psum') {
      while (nextBlockIdx >= (currentMBlockCumsum + numMBlocks) * numNBlocks) {
        currentGroupIdx += 1;
        if (currentGroupIdx === numGroups) return null;
        lastPsumM = align(currentPsumM, blockM);
        currentPsumM = opts.psumMPerGroup[currentGroupIdx];
        currentMBlockCumsum += numMBlocks;
        numMBlocks = ceilDiv(currentPsumM - lastPsumM, blockM);
      }
      const [mBlock, nBlock] = swizzle(nextBlockIdx - currentMBlockCumsum * numNBlocks);
      return [mBlock + lastPsumM / blockM, nBlock];
    }
    if (nextBlockIdx >= numBlocks) return null;
    return swizzle(nextBlockIdx);
  };

  const groupOf = (mBlock: number): number =>
    opts.type === 'm_grouped_contiguous'
      ? (opts.groupIdxPerRow[mBlock * blockM] ?? -1)
      : opts.type === 'normal'
        ? 0
        : currentGroupIdx;

  const multicastValid = (mBlock: number): boolean => {
    if (numMulticast === 1 || numBlocksInGroup === 1) return false;
    if (opts.type !== 'm_grouped_contiguous' || multicastOnA) return true;
    return groupOf(mBlock) === (opts.groupIdxPerRow[(mBlock ^ 1) * blockM] ?? -1);
  };

  const tiles: Tile[] = [];
  for (let next = nextBlock(); next !== null; next = nextBlock()) {
    const [mBlock, nBlock] = next;
    tiles.push({ sm, iter: currentIter, mBlock, nBlock, group: groupOf(mBlock), multicast: multicastValid(mBlock) });
  }
  return tiles;
}

export function schedule(opts: ScheduleOptions): Tile[][] {
  return Array.from({ length: opts.numSms }, (_, sm) => scheduleSm(opts, sm));
}

export function isComputationValid(opts: ScheduleOptions, tile: Tile, mOffset: number): boolean {
  const row = mOffset + tile.mBlock * opts.blockM;
  switch (opts.type) {
    case 'normal':
      return true;
    case 'm_grouped_contiguous':
      return (opts.groupIdxPerRow[row] ?? -1) >= 0;
    case 'm_grouped_masked':
      return row < opts.validMPerGroup[tile.group];
    case 'm_grouped_psum':
      return row < opts.psumMPerGroup[tile.group];
  }
}

export interface PipelineDurations {
  tmaLoad: number;
  wgmma: number;
  promote: number;
  epilogue: number;
}

export const DEFAULT_PIPELINE_DURATIONS: PipelineDurations = {
  tmaLoad: 4,
  wgmma: 3,
  promote: 1,
  epilogue: 6,
};

export type PipelineRow = 'tma' | 'wg0' | 'wg1' | 'promote' | 'epilogue';

export interface PipelineEvent {
  row: PipelineRow;
  start: number;
  end: number;
  tile: number;
  kBlock: number;
  stage: number;
  wg: number | null;
}

export interface StageRelease {
  tile: number;
  kBlock: number;
  stage: number;
  time: number;
}

export interface PipelineOptions {
  schedule: ScheduleOptions;
  sm: number;
  numStages: number;
  durations?: Partial<PipelineDurations>;
}

export interface PipelineResult {
  numMathWarpgroups: number;
  numWaves: number;
  tiles: Tile[];
  events: PipelineEvent[];
  releases: StageRelease[];
}

export function pipeline(opts: PipelineOptions): PipelineResult {
  const shape = opts.schedule;
  if (shape.blockK !== 128) throw new Error('sm90_fp8_gemm_1d2d requires blockK == 128');
  const d = { ...DEFAULT_PIPELINE_DURATIONS, ...opts.durations };
  const numMathWarpgroups = shape.blockM <= 64 ? 1 : 2;
  const waveBlockM = shape.blockM <= WGMMA_M ? shape.blockM : WGMMA_M * 2;
  const numWaves = shape.blockM / waveBlockM;
  const numKBlocks = ceilDiv(shape.k, shape.blockK);
  const tiles = schedule(shape)[opts.sm];
  const events: PipelineEvent[] = [];
  const releases: StageRelease[] = [];
  const releaseTimes: number[] = [];
  const wgReady = new Array<number>(numMathWarpgroups).fill(0);
  let producer = 0;
  let seq = 0;

  tiles.forEach((tile, tileIdx) => {
    const tileStart = Math.max(...wgReady);
    wgReady.fill(tileStart);
    const valid = wgReady.map((_, wg) => isComputationValid(shape, tile, wg * WGMMA_M));
    for (let kBlock = 0; kBlock < numKBlocks; kBlock += 1, seq += 1) {
      const stage = seq % opts.numStages;
      const issue = Math.max(producer, seq >= opts.numStages ? releaseTimes[seq - opts.numStages] : 0);
      producer = issue;
      const loaded = issue + d.tmaLoad;
      events.push({ row: 'tma', start: issue, end: loaded, tile: tileIdx, kBlock, stage, wg: null });
      let release = 0;
      for (let wg = 0; wg < numMathWarpgroups; wg += 1) {
        let t = Math.max(wgReady[wg], loaded);
        let arrive = t;
        if (valid[wg]) {
          for (let wave = 0; wave < numWaves; wave += 1) {
            const row = wg === 0 ? 'wg0' : 'wg1';
            events.push({ row, start: t, end: t + d.wgmma, tile: tileIdx, kBlock, stage, wg });
            t += d.wgmma;
            if (wave === numWaves - 1) arrive = t;
            events.push({ row: 'promote', start: t, end: t + d.promote, tile: tileIdx, kBlock, stage, wg });
            t += d.promote;
          }
        }
        wgReady[wg] = t;
        release = Math.max(release, arrive);
      }
      releaseTimes.push(release);
      releases.push({ tile: tileIdx, kBlock, stage, time: release });
    }
    const epilogueStart = Math.max(...wgReady);
    events.push({
      row: 'epilogue',
      start: epilogueStart,
      end: epilogueStart + d.epilogue,
      tile: tileIdx,
      kBlock: numKBlocks - 1,
      stage: -1,
      wg: null,
    });
    wgReady.fill(epilogueStart + d.epilogue);
  });
  return { numMathWarpgroups, numWaves, tiles, events, releases };
}
