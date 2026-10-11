export interface EpTopology {
  numScaleoutRanks: number;
  numScaleupRanks: number;
  numExperts: number;
}

export type TopkIdx = readonly (readonly (readonly number[])[])[];

export type EpMode = 'direct' | 'hybrid';
export type Via = 'nvlink' | 'rdma' | 'local';
export type Hop = 'send' | 'forward';

export interface Transfer {
  tokenRank: number;
  token: number;
  srcRank: number;
  dstRank: number;
  via: Via;
  hop: Hop;
  bytes: number;
  experts: number[];
}

export const TMA_ALIGNMENT_BYTES = 32;
export const RDMA_ALIGNMENT_BYTES = 64;
export const SF_BLOCK = 128;
export const SF_PACK_BYTES = 4;

const ceilDiv = (a: number, b: number) => Math.floor((a + b - 1) / b);
const align = (a: number, b: number) => ceilDiv(a, b) * b;

export function numRanks(topo: EpTopology): number {
  return topo.numScaleoutRanks * topo.numScaleupRanks;
}

export function numExpertsPerRank(topo: EpTopology): number {
  const ranks = numRanks(topo);
  if (topo.numExperts % ranks !== 0) throw new Error('numExperts must be a multiple of the number of ranks');
  return topo.numExperts / ranks;
}

export function rankOfExpert(topo: EpTopology, expert: number): number {
  return Math.floor(expert / numExpertsPerRank(topo));
}

export function scaleoutOf(topo: EpTopology, rank: number): number {
  return Math.floor(rank / topo.numScaleupRanks);
}

export function scaleupOf(topo: EpTopology, rank: number): number {
  return rank % topo.numScaleupRanks;
}

export function rankAt(topo: EpTopology, scaleout: number, scaleup: number): number {
  return scaleout * topo.numScaleupRanks + scaleup;
}

export function tokenLayoutBytes(
  numHiddenBytes: number,
  numSfBytes: number,
  numTopk: number,
  withMetadata: boolean,
): number {
  const metadata = numTopk * 8 + (withMetadata ? (1 + numTopk) * 4 : 0);
  return align(
    align(numHiddenBytes, TMA_ALIGNMENT_BYTES) + align(numSfBytes, TMA_ALIGNMENT_BYTES) + align(metadata, TMA_ALIGNMENT_BYTES),
    RDMA_ALIGNMENT_BYTES,
  );
}

export type DispatchDtype = 'fp8' | 'bf16';

export function dispatchTokenBytes(hidden: number, dtype: DispatchDtype, numTopk: number): number {
  const hiddenBytes = dtype === 'fp8' ? hidden : hidden * 2;
  const sfBytes = dtype === 'fp8' ? ceilDiv(hidden, SF_BLOCK) * SF_PACK_BYTES : 0;
  return tokenLayoutBytes(hiddenBytes, sfBytes, numTopk, true);
}

export function combineTokenBytes(hidden: number, numTopk: number): number {
  return tokenLayoutBytes(hidden * 2, 0, numTopk, false);
}

function groupBy(experts: readonly number[], key: (expert: number) => number): Map<number, number[]> {
  const groups = new Map<number, number[]>();
  for (const expert of experts) {
    if (expert < 0) continue;
    const k = key(expert);
    const list = groups.get(k);
    if (list) list.push(expert);
    else groups.set(k, [expert]);
  }
  return new Map([...groups].sort((a, b) => a[0] - b[0]));
}

function topkWidth(topkIdx: TopkIdx): number {
  for (const tokens of topkIdx) if (tokens.length > 0) return tokens[0].length;
  return 0;
}

export interface RankNotify {
  numRecvTokensPerScaleupRank: number[];
  psumNumRecvTokensPerScaleupRank: number[];
  numUnalignedRecvTokensPerExpert: number[];
  psumNumRecvTokensPerExpertExclusive: number[];
  psumNumRecvTokensPerExpert: number[];
  numRecvTokens: number;
  numExpandedRows: number;
}

export interface NotifyResult {
  numTokensPerRank: number[][];
  numTokensPerExpert: number[][];
  ranks: RankNotify[];
}

export function notify(topo: EpTopology, topkIdx: TopkIdx, mode: EpMode, expertAlignment: number): NotifyResult {
  const ranks = numRanks(topo);
  const perRank = numExpertsPerRank(topo);
  const numTokensPerRank = topkIdx.map((tokens) => {
    const counts = new Array<number>(ranks).fill(0);
    for (const row of tokens) for (const dst of groupBy(row, (e) => rankOfExpert(topo, e)).keys()) counts[dst] += 1;
    return counts;
  });
  const numTokensPerExpert = topkIdx.map((tokens) => {
    const counts = new Array<number>(topo.numExperts).fill(0);
    for (const row of tokens) for (const e of row) if (e >= 0) counts[e] += 1;
    return counts;
  });
  const result: RankNotify[] = [];
  for (let r = 0; r < ranks; r += 1) {
    const numRecvTokensPerScaleupRank =
      mode === 'direct'
        ? numTokensPerRank.map((counts) => counts[r])
        : Array.from({ length: topo.numScaleupRanks }, (_, s) => {
            let sum = 0;
            for (let node = 0; node < topo.numScaleoutRanks; node += 1) sum += numTokensPerRank[rankAt(topo, node, s)][r];
            return sum;
          });
    const psumNumRecvTokensPerScaleupRank = inclusiveSum(numRecvTokensPerScaleupRank);
    const numUnalignedRecvTokensPerExpert = Array.from({ length: perRank }, (_, e) =>
      numTokensPerExpert.reduce((sum, counts) => sum + counts[r * perRank + e], 0),
    );
    const aligned = numUnalignedRecvTokensPerExpert.map((count) => align(count, expertAlignment));
    const psumNumRecvTokensPerExpertExclusive = [0, ...inclusiveSum(aligned)];
    result.push({
      numRecvTokensPerScaleupRank,
      psumNumRecvTokensPerScaleupRank,
      numUnalignedRecvTokensPerExpert,
      psumNumRecvTokensPerExpertExclusive,
      psumNumRecvTokensPerExpert: numUnalignedRecvTokensPerExpert.map(
        (count, e) => psumNumRecvTokensPerExpertExclusive[e] + count,
      ),
      numRecvTokens: psumNumRecvTokensPerScaleupRank[psumNumRecvTokensPerScaleupRank.length - 1] ?? 0,
      numExpandedRows: psumNumRecvTokensPerExpertExclusive[perRank],
    });
  }
  return { numTokensPerRank, numTokensPerExpert, ranks: result };
}

function inclusiveSum(values: readonly number[]): number[] {
  let sum = 0;
  return values.map((v) => (sum += v));
}

export interface DispatchOptions {
  mode: EpMode;
  hidden: number;
  dtype: DispatchDtype;
}

function viaBetween(topo: EpTopology, src: number, dst: number): Via {
  if (src === dst) return 'local';
  return scaleoutOf(topo, src) === scaleoutOf(topo, dst) ? 'nvlink' : 'rdma';
}

export function dispatchPlan(topo: EpTopology, topkIdx: TopkIdx, opts: DispatchOptions): Transfer[] {
  const bytes = dispatchTokenBytes(opts.hidden, opts.dtype, topkWidth(topkIdx));
  const transfers: Transfer[] = [];
  topkIdx.forEach((tokens, tokenRank) => {
    tokens.forEach((row, token) => {
      const base = { tokenRank, token, bytes };
      if (opts.mode === 'direct') {
        for (const [dstRank, experts] of groupBy(row, (e) => rankOfExpert(topo, e))) {
          transfers.push({ ...base, srcRank: tokenRank, dstRank, via: viaBetween(topo, tokenRank, dstRank), hop: 'send', experts });
        }
        return;
      }
      for (const [node, nodeExperts] of groupBy(row, (e) => scaleoutOf(topo, rankOfExpert(topo, e)))) {
        const forwarder = rankAt(topo, node, scaleupOf(topo, tokenRank));
        transfers.push({
          ...base,
          srcRank: tokenRank,
          dstRank: forwarder,
          via: viaBetween(topo, tokenRank, forwarder),
          hop: 'send',
          experts: nodeExperts,
        });
        for (const [dstRank, experts] of groupBy(nodeExperts, (e) => rankOfExpert(topo, e))) {
          transfers.push({ ...base, srcRank: forwarder, dstRank, via: viaBetween(topo, forwarder, dstRank), hop: 'forward', experts });
        }
      }
    });
  });
  return transfers;
}

export type ExpandedRow =
  | { kind: 'token'; srcRank: number; token: number; expert: number }
  | { kind: 'padding'; expert: number };

export function expandedLayout(topo: EpTopology, topkIdx: TopkIdx, expertAlignment: number): ExpandedRow[][] {
  const perRank = numExpertsPerRank(topo);
  return Array.from({ length: numRanks(topo) }, (_, rank) => {
    const rows: ExpandedRow[] = [];
    for (let local = 0; local < perRank; local += 1) {
      const expert = rank * perRank + local;
      const start = rows.length;
      topkIdx.forEach((tokens, srcRank) =>
        tokens.forEach((row, token) => {
          if (row.includes(expert)) rows.push({ kind: 'token', srcRank, token, expert });
        }),
      );
      const end = start + align(rows.length - start, expertAlignment);
      while (rows.length < end) rows.push({ kind: 'padding', expert });
    }
    return rows;
  });
}

export interface CombineOptions {
  mode: EpMode;
  allowMultipleReduction: boolean;
  hidden: number;
}

export function combinePlan(topo: EpTopology, topkIdx: TopkIdx, opts: CombineOptions): Transfer[] {
  const numTopk = topkWidth(topkIdx);
  const layoutBytes = combineTokenBytes(opts.hidden, numTopk);
  const hiddenBytes = opts.hidden * 2;
  const transfers: Transfer[] = [];
  const split = (experts: number[]) => (opts.allowMultipleReduction ? [experts] : experts.map((e) => [e]));
  topkIdx.forEach((tokens, tokenRank) => {
    tokens.forEach((row, token) => {
      if (opts.mode === 'direct') {
        for (const [srcRank, rankExperts] of groupBy(row, (e) => rankOfExpert(topo, e))) {
          const via = viaBetween(topo, srcRank, tokenRank);
          const bytes = opts.allowMultipleReduction && via === 'rdma' ? layoutBytes : hiddenBytes;
          for (const experts of split(rankExperts)) {
            transfers.push({ tokenRank, token, srcRank, dstRank: tokenRank, via, hop: 'send', bytes, experts });
          }
        }
        return;
      }
      for (const [node, nodeExperts] of groupBy(row, (e) => scaleoutOf(topo, rankOfExpert(topo, e)))) {
        const forwarder = rankAt(topo, node, scaleupOf(topo, tokenRank));
        for (const [srcRank, rankExperts] of groupBy(nodeExperts, (e) => rankOfExpert(topo, e))) {
          for (const experts of split(rankExperts)) {
            transfers.push({
              tokenRank,
              token,
              srcRank,
              dstRank: forwarder,
              via: viaBetween(topo, srcRank, forwarder),
              hop: 'send',
              bytes: layoutBytes,
              experts,
            });
          }
        }
        for (const experts of split(nodeExperts)) {
          transfers.push({
            tokenRank,
            token,
            srcRank: forwarder,
            dstRank: tokenRank,
            via: viaBetween(topo, forwarder, tokenRank),
            hop: 'forward',
            bytes: layoutBytes,
            experts,
          });
        }
      }
    });
  });
  return transfers;
}

export function bytesByVia(transfers: readonly Transfer[]): Record<Via, number> {
  const totals: Record<Via, number> = { nvlink: 0, rdma: 0, local: 0 };
  for (const t of transfers) totals[t.via] += t.bytes;
  return totals;
}

export function bytesByNodePair(topo: EpTopology, transfers: readonly Transfer[]): number[][] {
  const matrix = Array.from({ length: topo.numScaleoutRanks }, () => new Array<number>(topo.numScaleoutRanks).fill(0));
  for (const t of transfers) matrix[scaleoutOf(topo, t.srcRank)][scaleoutOf(topo, t.dstRank)] += t.bytes;
  return matrix;
}

export function bytesPerRank(
  topo: EpTopology,
  transfers: readonly Transfer[],
  vias: readonly Via[] = ['nvlink', 'rdma'],
): { sent: number[]; received: number[] } {
  const sent = new Array<number>(numRanks(topo)).fill(0);
  const received = new Array<number>(numRanks(topo)).fill(0);
  for (const t of transfers) {
    if (!vias.includes(t.via)) continue;
    sent[t.srcRank] += t.bytes;
    received[t.dstRank] += t.bytes;
  }
  return { sent, received };
}

export function maxBytesPerRank(
  topo: EpTopology,
  transfers: readonly Transfer[],
  vias: readonly Via[] = ['nvlink', 'rdma'],
): { sent: number; received: number } {
  const { sent, received } = bytesPerRank(topo, transfers, vias);
  return { sent: Math.max(0, ...sent), received: Math.max(0, ...received) };
}
