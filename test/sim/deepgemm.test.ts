import { describe, expect, it } from 'vitest';
import {
  type GemmShape,
  type ScheduleOptions,
  type Tile,
  isComputationValid,
  numBlocksPerGroup,
  pipeline,
  schedule,
} from '../../src/sim';

const base: GemmShape = {
  m: 512,
  n: 512,
  k: 1024,
  blockM: 128,
  blockN: 128,
  blockK: 128,
  numSms: 4,
  numMulticast: 1,
  multicastOnA: false,
};

const mn = (tiles: Tile[]) => tiles.map((t) => [t.mBlock, t.nBlock]);
const mng = (tiles: Tile[]) => tiles.map((t) => [t.mBlock, t.nBlock, t.group]);
const align = (a: number, b: number) => Math.ceil(a / b) * b;

function psumFromMs(ms: number[], alignment: number): number[] {
  let end = 0;
  return ms.map((m) => (end = align(end, alignment) + m));
}

describe('numBlocksPerGroup', () => {
  it.each([
    [128, 128, 4, false, 8],
    [128, 128, 132, false, 8],
    [256, 128, 132, false, 8],
    [64, 256, 132, false, 16],
    [256, 64, 132, true, 16],
  ])('blockM=%i blockN=%i sms=%i onA=%s gives %i', (bm, bn, sms, onA, expected) => {
    expect(numBlocksPerGroup(bm, bn, sms, onA)).toBe(expected);
  });
});

describe('schedule order', () => {
  it('walks M inside a group for a 4x4 grid on 4 SMs', () => {
    const tiles = schedule({ ...base, type: 'normal' });
    expect(mn(tiles[0])).toEqual([[0, 0], [0, 1], [0, 2], [0, 3]]);
    expect(mn(tiles[1])).toEqual([[1, 0], [1, 1], [1, 2], [1, 3]]);
    expect(mn(tiles[3])).toEqual([[3, 0], [3, 1], [3, 2], [3, 3]]);
    expect(tiles[0].map((t) => t.iter)).toEqual([0, 1, 2, 3]);
  });

  it('moves to the next M group after 8 M blocks', () => {
    const tiles = schedule({ ...base, type: 'normal', m: 2048, n: 256 });
    expect(mn(tiles[0])).toEqual([[0, 0], [4, 0], [0, 1], [4, 1], [8, 0], [12, 0], [8, 1], [12, 1]]);
  });

  it('groups on N when multicasting on A', () => {
    const tiles = schedule({ ...base, type: 'normal', m: 256, n: 512, numSms: 2, multicastOnA: true, numMulticast: 2 });
    expect(mn(tiles[0])).toEqual([[0, 0], [0, 2], [1, 0], [1, 2]]);
    expect(mn(tiles[1])).toEqual([[0, 1], [0, 3], [1, 1], [1, 3]]);
    expect(tiles.flat().every((t) => t.multicast)).toBe(true);
  });

  it('splits an odd group so that multicast pairs stay aligned', () => {
    const tiles = schedule({ ...base, type: 'normal', m: 640, n: 256, numMulticast: 2 });
    expect(mn(tiles[0])).toEqual([[0, 0], [0, 1], [4, 0]]);
    expect(mn(tiles[1])).toEqual([[1, 0], [1, 1], [4, 1]]);
    expect(mn(tiles[2])).toEqual([[2, 0], [2, 1]]);
    expect(mn(tiles[3])).toEqual([[3, 0], [3, 1]]);
    expect(tiles[0].map((t) => t.multicast)).toEqual([true, true, false]);
  });

  it('pairs CTAs that share the multicast operand', () => {
    const tiles = schedule({ ...base, type: 'normal', m: 1664, n: 768, numSms: 8, numMulticast: 2 });
    let pairs = 0;
    for (let sm = 0; sm < 8; sm += 2) {
      tiles[sm].forEach((tile, i) => {
        const peer = tiles[sm + 1][i];
        if (!tile.multicast) return;
        pairs += 1;
        expect(peer.nBlock).toBe(tile.nBlock);
        expect(peer.mBlock).toBe(tile.mBlock ^ 1);
      });
    }
    expect(pairs).toBeGreaterThan(0);
  });
});

describe('grouped schedules', () => {
  it('masked layout skips empty groups and keeps M local to the group', () => {
    const opts: ScheduleOptions = {
      ...base,
      type: 'm_grouped_masked',
      m: 256,
      n: 256,
      numSms: 2,
      validMPerGroup: [130, 0, 64],
    };
    const tiles = schedule(opts);
    expect(mng(tiles[0])).toEqual([[0, 0, 0], [0, 1, 0], [0, 0, 2]]);
    expect(mng(tiles[1])).toEqual([[1, 0, 0], [1, 1, 0], [0, 1, 2]]);
    expect(isComputationValid(opts, tiles[1][0], 0)).toBe(true);
    expect(isComputationValid(opts, tiles[1][0], 64)).toBe(false);
    expect(isComputationValid(opts, tiles[0][2], 64)).toBe(false);
  });

  it('psum layout offsets M by the aligned end of the previous group', () => {
    const psumMPerGroup = psumFromMs([100, 172, 0, 120], 128);
    expect(psumMPerGroup).toEqual([100, 300, 384, 504]);
    const opts: ScheduleOptions = { ...base, type: 'm_grouped_psum', m: 512, n: 128, numSms: 3, psumMPerGroup };
    const tiles = schedule(opts);
    expect(mng(tiles[0])).toEqual([[0, 0, 0], [3, 0, 3]]);
    expect(mng(tiles[1])).toEqual([[1, 0, 1]]);
    expect(mng(tiles[2])).toEqual([[2, 0, 1]]);
    expect(isComputationValid(opts, tiles[2][0], 0)).toBe(true);
    expect(isComputationValid(opts, tiles[2][0], 64)).toBe(false);
  });

  it('contiguous layout reads the group of the first row of each M block', () => {
    const groupIdxPerRow = Array.from({ length: 512 }, (_, row) =>
      row < 156 ? 0 : row < 256 ? -1 : row < 456 ? 1 : -1,
    );
    const opts: ScheduleOptions = { ...base, type: 'm_grouped_contiguous', n: 128, groupIdxPerRow };
    const tiles = schedule(opts).flat();
    expect(tiles.map((t) => t.group)).toEqual([0, 0, 1, 1]);
    expect(isComputationValid(opts, tiles[1], 64)).toBe(false);
    expect(isComputationValid(opts, tiles[3], 64)).toBe(true);
  });

  it('contiguous layout disables multicast across a group boundary', () => {
    const groupIdxPerRow = Array.from({ length: 512 }, (_, row) => (row < 256 ? 0 : row < 384 ? 1 : 2));
    const tiles = schedule({ ...base, type: 'm_grouped_contiguous', n: 256, numMulticast: 2, groupIdxPerRow })
      .flat()
      .sort((a, b) => a.mBlock - b.mBlock || a.nBlock - b.nBlock);
    expect(tiles.map((t) => t.multicast)).toEqual([true, true, true, true, false, false, false, false]);
  });
});

describe('schedule coverage', () => {
  const shapes: GemmShape[] = [
    { ...base, m: 1000, n: 1500, numSms: 7 },
    { ...base, m: 4096, n: 7168, blockN: 112, numSms: 132 },
    { ...base, m: 2304, n: 640, numSms: 10, numMulticast: 2 },
    { ...base, m: 640, n: 2304, numSms: 12, numMulticast: 2, multicastOnA: true },
    { ...base, m: 768, n: 4096, blockM: 64, blockN: 256, numSms: 78 },
  ];

  const counts = (tiles: Tile[][], key: (t: Tile) => string) => {
    const seen = new Map<string, number>();
    for (const tile of tiles.flat()) seen.set(key(tile), (seen.get(key(tile)) ?? 0) + 1);
    return seen;
  };

  it.each(shapes)('normal and contiguous cover every tile once (%#)', (shape) => {
    const numM = Math.ceil(shape.m / shape.blockM);
    const numN = Math.ceil(shape.n / shape.blockN);
    const groupIdxPerRow = Array.from({ length: numM * shape.blockM }, (_, row) => Math.floor(row / 384));
    const variants: ScheduleOptions[] = [
      { ...shape, type: 'normal' },
      { ...shape, type: 'm_grouped_contiguous', groupIdxPerRow },
    ];
    for (const opts of variants) {
      const tiles = schedule(opts);
      const seen = counts(tiles, (t) => `${t.mBlock},${t.nBlock}`);
      expect(seen.size).toBe(numM * numN);
      expect([...seen.values()].every((c) => c === 1)).toBe(true);
      tiles.forEach((list, sm) => list.forEach((t, i) => expect([t.sm, t.iter]).toEqual([sm, i])));
    }
  });

  it.each(shapes)('masked and psum cover every valid tile of every group once (%#)', (shape) => {
    const ms = [shape.m, 0, Math.floor(shape.m / 3) + 1, 17, shape.blockM];
    const numN = Math.ceil(shape.n / shape.blockN);
    const masked = schedule({ ...shape, type: 'm_grouped_masked', validMPerGroup: ms });
    const maskedSeen = counts(masked, (t) => `${t.group},${t.mBlock},${t.nBlock}`);
    const expected = ms.reduce((sum, m) => sum + Math.ceil(m / shape.blockM) * numN, 0);
    expect(maskedSeen.size).toBe(expected);
    expect([...maskedSeen.values()].every((c) => c === 1)).toBe(true);
    for (const tile of masked.flat()) expect(tile.mBlock).toBeLessThan(Math.ceil(ms[tile.group] / shape.blockM));

    const psumMPerGroup = psumFromMs(ms, shape.blockM);
    const psum = schedule({ ...shape, type: 'm_grouped_psum', psumMPerGroup });
    const psumSeen = counts(psum, (t) => `${t.mBlock},${t.nBlock}`);
    expect(psumSeen.size).toBe(expected);
    expect([...psumSeen.values()].every((c) => c === 1)).toBe(true);
    for (const tile of psum.flat()) {
      const start = tile.group === 0 ? 0 : align(psumMPerGroup[tile.group - 1], shape.blockM);
      expect(tile.mBlock * shape.blockM).toBeGreaterThanOrEqual(start);
      expect(tile.mBlock * shape.blockM).toBeLessThan(psumMPerGroup[tile.group]);
    }
  });
});

describe('pipeline', () => {
  it('reproduces a hand-computed timeline with one stage', () => {
    const result = pipeline({
      schedule: { ...base, type: 'normal', m: 128, n: 128, k: 256, numSms: 1 },
      sm: 0,
      numStages: 1,
      durations: { tmaLoad: 4, wgmma: 3, promote: 1, epilogue: 6 },
    });
    const rows = result.events.map((e) => [e.row, e.start, e.end, e.kBlock, e.wg]);
    expect(rows).toEqual([
      ['tma', 0, 4, 0, null],
      ['wg0', 4, 7, 0, 0],
      ['promote', 7, 8, 0, 0],
      ['wg1', 4, 7, 0, 1],
      ['promote', 7, 8, 0, 1],
      ['tma', 7, 11, 1, null],
      ['wg0', 11, 14, 1, 0],
      ['promote', 14, 15, 1, 0],
      ['wg1', 11, 14, 1, 1],
      ['promote', 14, 15, 1, 1],
      ['epilogue', 15, 21, 1, null],
    ]);
    expect(result.releases.map((r) => r.time)).toEqual([7, 14]);
  });

  it('lets the producer run ahead into the next tile while math does the epilogue', () => {
    const result = pipeline({
      schedule: { ...base, type: 'normal', m: 256, n: 128, k: 256, numSms: 1 },
      sm: 0,
      numStages: 4,
      durations: { tmaLoad: 4, wgmma: 3, promote: 1, epilogue: 6 },
    });
    const loads = result.events.filter((e) => e.row === 'tma');
    expect(loads.map((e) => [e.tile, e.kBlock, e.start])).toEqual([
      [0, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 0],
    ]);
    const secondTileMath = result.events.find((e) => e.tile === 1 && e.row === 'wg0')!;
    expect(secondTileMath.start).toBe(18);
  });

  const cases: { name: string; opts: ScheduleOptions; numStages: number }[] = [
    { name: 'normal 128x128', opts: { ...base, type: 'normal', m: 1024, numSms: 3 }, numStages: 4 },
    { name: 'normal 256x128', opts: { ...base, type: 'normal', blockM: 256, m: 1024, numSms: 2 }, numStages: 3 },
    { name: 'normal 64x256', opts: { ...base, type: 'normal', blockM: 64, blockN: 256, numSms: 2 }, numStages: 6 },
    {
      name: 'masked',
      opts: { ...base, type: 'm_grouped_masked', m: 256, n: 256, numSms: 2, validMPerGroup: [130, 0, 64] },
      numStages: 2,
    },
  ];

  it.each(cases)('keeps barrier invariants ($name)', ({ opts, numStages }) => {
    for (const durations of [{}, { tmaLoad: 20 }, { wgmma: 10, promote: 4 }]) {
      const result = pipeline({ schedule: opts, sm: 1, numStages, durations });
      const loads = result.events.filter((e) => e.row === 'tma');
      expect(loads.length).toBe(result.releases.length);
      loads.forEach((load, q) => {
        expect(load.stage).toBe(q % numStages);
        const release = result.releases[q];
        expect([release.tile, release.kBlock]).toEqual([load.tile, load.kBlock]);
        expect(release.time).toBeGreaterThanOrEqual(load.end);
        if (q >= numStages) expect(load.start).toBeGreaterThanOrEqual(result.releases[q - numStages].time);
        const outstanding = loads.filter((l, p) => l.start <= load.start && load.start < result.releases[p].time);
        expect(outstanding.length).toBeLessThanOrEqual(numStages);
      });
      for (const e of result.events.filter((e) => e.row === 'wg0' || e.row === 'wg1')) {
        const load = loads.find((l) => l.tile === e.tile && l.kBlock === e.kBlock)!;
        expect(e.start).toBeGreaterThanOrEqual(load.end);
        expect(e.stage).toBe(load.stage);
      }
      result.tiles.forEach((tile, t) => {
        const math = result.events.filter((e) => e.tile === t && e.row !== 'tma' && e.row !== 'epilogue');
        const epilogue = result.events.find((e) => e.tile === t && e.row === 'epilogue')!;
        expect(Math.max(0, ...math.map((e) => e.end))).toBeLessThanOrEqual(epilogue.start);
        for (let wg = 0; wg < result.numMathWarpgroups; wg += 1) {
          const n = math.filter((e) => e.wg === wg && e.row !== 'promote').length;
          const valid = isComputationValid(opts, tile, wg * 64);
          expect(n).toBe(valid ? (opts.k / 128) * result.numWaves : 0);
        }
      });
    }
  });

  it('derives warpgroups and waves from blockM', () => {
    const at = (blockM: number) => pipeline({ schedule: { ...base, type: 'normal', blockM }, sm: 0, numStages: 2 });
    expect([at(64).numMathWarpgroups, at(64).numWaves]).toEqual([1, 1]);
    expect([at(128).numMathWarpgroups, at(128).numWaves]).toEqual([2, 1]);
    expect([at(256).numMathWarpgroups, at(256).numWaves]).toEqual([2, 2]);
    expect(at(64).events.some((e) => e.row === 'wg1')).toBe(false);
  });
});
