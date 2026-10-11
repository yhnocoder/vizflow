import { describe, expect, it } from 'vitest';
import {
  type EpTopology,
  type Transfer,
  bytesByNodePair,
  bytesByVia,
  combinePlan,
  combineTokenBytes,
  dispatchPlan,
  dispatchTokenBytes,
  expandedLayout,
  maxBytesPerRank,
  notify,
  rankOfExpert,
  scaleoutOf,
} from '../../src/sim';

const small: EpTopology = { numScaleoutRanks: 2, numScaleupRanks: 2, numExperts: 8 };
const smallTopk = [
  [
    [4, 5, 6, 7],
    [0, 1, 2, -1],
  ],
  [],
  [],
  [],
];

const edges = (transfers: Transfer[]) => transfers.map((t) => [t.token, t.srcRank, t.dstRank, t.via, t.hop]);

describe('token bytes', () => {
  it('follows TokenLayout alignment', () => {
    expect(dispatchTokenBytes(7168, 'fp8', 4)).toBe(7488);
    expect(dispatchTokenBytes(7168, 'fp8', 8)).toBe(7552);
    expect(dispatchTokenBytes(7168, 'bf16', 8)).toBe(14336 + 128);
    expect(combineTokenBytes(7168, 8)).toBe(14336 + 64);
  });
});

describe('dispatch on a hand example', () => {
  it('direct mode sends once per destination rank', () => {
    expect(edges(dispatchPlan(small, smallTopk, { mode: 'direct', hidden: 128, dtype: 'bf16' }))).toEqual([
      [0, 0, 2, 'rdma', 'send'],
      [0, 0, 3, 'rdma', 'send'],
      [1, 0, 0, 'local', 'send'],
      [1, 0, 1, 'nvlink', 'send'],
    ]);
  });

  it('hybrid mode sends once per node and forwards inside the node', () => {
    const plan = dispatchPlan(small, smallTopk, { mode: 'hybrid', hidden: 128, dtype: 'bf16' });
    expect(edges(plan)).toEqual([
      [0, 0, 2, 'rdma', 'send'],
      [0, 2, 2, 'local', 'forward'],
      [0, 2, 3, 'nvlink', 'forward'],
      [1, 0, 0, 'local', 'send'],
      [1, 0, 0, 'local', 'forward'],
      [1, 0, 1, 'nvlink', 'forward'],
    ]);
    expect(plan[2].experts).toEqual([6, 7]);
  });

  it('halves RDMA bytes when two destination ranks share a node', () => {
    const opts = { hidden: 7168, dtype: 'fp8' } as const;
    const direct = bytesByVia(dispatchPlan(small, smallTopk, { ...opts, mode: 'direct' }));
    const hybrid = bytesByVia(dispatchPlan(small, smallTopk, { ...opts, mode: 'hybrid' }));
    expect(direct.rdma).toBe(2 * 7488);
    expect(hybrid.rdma).toBe(7488);
    expect(bytesByNodePair(small, dispatchPlan(small, smallTopk, { ...opts, mode: 'direct' }))).toEqual([
      [2 * 7488, 2 * 7488],
      [0, 0],
    ]);
  });

  it('notify deduplicates ranks and pads experts', () => {
    const direct = notify(small, smallTopk, 'direct', 4);
    expect(direct.numTokensPerRank[0]).toEqual([1, 1, 1, 1]);
    expect(direct.numTokensPerExpert[0]).toEqual([1, 1, 1, 0, 1, 1, 1, 1]);
    expect(direct.ranks[0]).toEqual({
      numRecvTokensPerScaleupRank: [1, 0, 0, 0],
      psumNumRecvTokensPerScaleupRank: [1, 1, 1, 1],
      numUnalignedRecvTokensPerExpert: [1, 1],
      psumNumRecvTokensPerExpertExclusive: [0, 4, 8],
      psumNumRecvTokensPerExpert: [1, 5],
      numRecvTokens: 1,
      numExpandedRows: 8,
    });
    expect(direct.ranks[1].numUnalignedRecvTokensPerExpert).toEqual([1, 0]);
    expect(direct.ranks[1].psumNumRecvTokensPerExpertExclusive).toEqual([0, 4, 4]);
    const hybrid = notify(small, smallTopk, 'hybrid', 4);
    expect(hybrid.ranks[2].numRecvTokensPerScaleupRank).toEqual([1, 0]);
    expect(hybrid.ranks[3].numRecvTokensPerScaleupRank).toEqual([1, 0]);
  });

  it('expanded layout groups rows by expert with zero padding', () => {
    const layout = expandedLayout(small, smallTopk, 4);
    expect(layout[1]).toEqual([
      { kind: 'token', srcRank: 0, token: 1, expert: 2 },
      { kind: 'padding', expert: 2 },
      { kind: 'padding', expert: 2 },
      { kind: 'padding', expert: 2 },
    ]);
  });
});

function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function nodeLimitedRouter(topo: EpTopology, tokensPerRank: number, topk: number, maxNodes: number, seed: number) {
  const random = mulberry32(seed);
  const expertsPerNode = topo.numExperts / topo.numScaleoutRanks;
  const ranks = topo.numScaleoutRanks * topo.numScaleupRanks;
  return Array.from({ length: ranks }, () =>
    Array.from({ length: tokensPerRank }, () => {
      const scores = Array.from({ length: topo.numExperts }, () => random());
      const nodeScore = Array.from({ length: topo.numScaleoutRanks }, (_, node) => {
        const s = scores.slice(node * expertsPerNode, (node + 1) * expertsPerNode).sort((a, b) => b - a);
        return s[0] + s[1];
      });
      const nodes = nodeScore
        .map((s, node) => [s, node])
        .sort((a, b) => b[0] - a[0])
        .slice(0, maxNodes)
        .map(([, node]) => node);
      return scores
        .map((s, e) => [s, e])
        .filter(([, e]) => nodes.includes(Math.floor(e / expertsPerNode)))
        .sort((a, b) => b[0] - a[0])
        .slice(0, topk)
        .map(([, e]) => e);
    }),
  );
}

describe('DeepSeek-V3-like routing', () => {
  const topo: EpTopology = { numScaleoutRanks: 8, numScaleupRanks: 8, numExperts: 256 };
  const topkIdx = nodeLimitedRouter(topo, 24, 8, 4, 2024);
  const key = (t: Transfer) => `${t.tokenRank}/${t.token}`;
  const groupCount = (transfers: Transfer[]) => {
    const counts = new Map<string, number>();
    for (const t of transfers) counts.set(key(t), (counts.get(key(t)) ?? 0) + 1);
    return counts;
  };
  const distinct = (values: number[]) => new Set(values).size;

  it('hybrid dispatch crosses RDMA at most four times per token', () => {
    const plan = dispatchPlan(topo, topkIdx, { mode: 'hybrid', hidden: 7168, dtype: 'fp8' });
    for (const count of groupCount(plan.filter((t) => t.via === 'rdma')).values()) expect(count).toBeLessThanOrEqual(4);
    const forwards = groupCount(plan.filter((t) => t.hop === 'forward'));
    topkIdx.forEach((tokens, rank) =>
      tokens.forEach((row, token) => {
        const ranks = distinct(row.map((e) => rankOfExpert(topo, e)));
        expect(forwards.get(`${rank}/${token}`)).toBe(ranks);
        const nodes = distinct(row.map((e) => scaleoutOf(topo, rankOfExpert(topo, e))));
        expect(nodes).toBeLessThanOrEqual(4);
      }),
    );
    const direct = dispatchPlan(topo, topkIdx, { mode: 'direct', hidden: 7168, dtype: 'fp8' });
    expect(direct.length).toBe(plan.filter((t) => t.hop === 'forward').length);
    expect(bytesByVia(direct).rdma).toBeGreaterThan(bytesByVia(plan).rdma);
    expect(maxBytesPerRank(topo, plan).sent).toBeGreaterThan(0);
  });

  it('notify counts match the plan and prefix sums match the expanded layout', () => {
    for (const mode of ['direct', 'hybrid'] as const) {
      const result = notify(topo, topkIdx, mode, 128);
      const plan = dispatchPlan(topo, topkIdx, { mode, hidden: 7168, dtype: 'fp8' });
      const arrivals = plan.filter((t) => (mode === 'direct' ? t.hop === 'send' : t.hop === 'forward'));
      const layout = expandedLayout(topo, topkIdx, 128);
      result.ranks.forEach((rank, r) => {
        expect(rank.numRecvTokens).toBe(arrivals.filter((t) => t.dstRank === r).length);
        expect(layout[r].length).toBe(rank.numExpandedRows);
        rank.numUnalignedRecvTokensPerExpert.forEach((count, e) => {
          const start = rank.psumNumRecvTokensPerExpertExclusive[e];
          const end = rank.psumNumRecvTokensPerExpertExclusive[e + 1];
          expect(start % 128).toBe(0);
          expect(end - start).toBe(Math.ceil(count / 128) * 128);
          expect(rank.psumNumRecvTokensPerExpert[e]).toBe(start + count);
          const rows = layout[r].slice(start, end);
          expect(rows.filter((row) => row.kind === 'token').length).toBe(count);
          expect(rows.slice(0, count).every((row) => row.kind === 'token' && row.expert === r * 4 + e)).toBe(true);
        });
      });
    }
  });

  it.each([
    ['direct', true],
    ['direct', false],
    ['hybrid', true],
    ['hybrid', false],
  ] as const)('combine (%s, multiple reduction %s) returns every expert row to its token', (mode, allowMultipleReduction) => {
    const plan = combinePlan(topo, topkIdx, { mode, allowMultipleReduction, hidden: 7168 });
    const finalHop = mode === 'direct' ? 'send' : 'forward';
    const received = new Map<string, number[]>();
    for (const t of plan.filter((t) => t.hop === finalHop)) {
      expect(t.dstRank).toBe(t.tokenRank);
      if (!allowMultipleReduction) expect(t.experts.length).toBe(1);
      received.set(key(t), [...(received.get(key(t)) ?? []), ...t.experts]);
    }
    topkIdx.forEach((tokens, rank) =>
      tokens.forEach((row, token) => {
        expect(received.get(`${rank}/${token}`)!.sort((a, b) => a - b)).toEqual([...row].sort((a, b) => a - b));
      }),
    );
    const layout = expandedLayout(topo, topkIdx, 1);
    const firstHop = new Set(plan.filter((t) => t.hop === 'send').flatMap((t) => t.experts.map((e) => `${t.srcRank}/${t.tokenRank}/${t.token}/${e}`)));
    layout.forEach((rows, r) =>
      rows.forEach((row) => {
        if (row.kind === 'token') expect(firstHop.has(`${r}/${row.srcRank}/${row.token}/${row.expert}`)).toBe(true);
      }),
    );
    if (mode === 'hybrid' && allowMultipleReduction) {
      for (const count of groupCount(plan.filter((t) => t.via === 'rdma')).values()) expect(count).toBeLessThanOrEqual(4);
      const dispatch = dispatchPlan(topo, topkIdx, { mode, hidden: 7168, dtype: 'fp8' });
      const forward = dispatch.filter((t) => t.hop === 'forward').map((t) => `${key(t)}:${t.dstRank}->${t.srcRank}`);
      const back = plan.filter((t) => t.hop === 'send').map((t) => `${key(t)}:${t.srcRank}->${t.dstRank}`);
      expect(back.sort()).toEqual(forward.sort());
    }
    if (mode === 'direct' && allowMultipleReduction) {
      const dispatch = dispatchPlan(topo, topkIdx, { mode, hidden: 7168, dtype: 'fp8' });
      const forward = dispatch.map((t) => `${key(t)}:${t.dstRank}->${t.srcRank}`);
      expect(plan.map((t) => `${key(t)}:${t.srcRank}->${t.dstRank}`).sort()).toEqual(forward.sort());
    }
  });

  it('hybrid combine with reduction moves fewer RDMA bytes than without', () => {
    const reduce = bytesByVia(combinePlan(topo, topkIdx, { mode: 'hybrid', allowMultipleReduction: true, hidden: 7168 }));
    const expand = bytesByVia(combinePlan(topo, topkIdx, { mode: 'hybrid', allowMultipleReduction: false, hidden: 7168 }));
    expect(reduce.rdma).toBeLessThan(expand.rdma);
  });
});
