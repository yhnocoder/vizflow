import type { EpTopology } from './deepep';

export const NODE_SCORE_TOP = 2;

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function nodeLimitedRouter(topo: EpTopology, tokensPerRank: number, topk: number, maxNodes: number, seed: number): number[][][] {
  const random = mulberry32(seed);
  const expertsPerNode = topo.numExperts / topo.numScaleoutRanks;
  const ranks = topo.numScaleoutRanks * topo.numScaleupRanks;
  return Array.from({ length: ranks }, () =>
    Array.from({ length: tokensPerRank }, () => {
      const scores = Array.from({ length: topo.numExperts }, () => random());
      const nodeScore = Array.from({ length: topo.numScaleoutRanks }, (_, node) =>
        scores
          .slice(node * expertsPerNode, (node + 1) * expertsPerNode)
          .sort((a, b) => b - a)
          .slice(0, NODE_SCORE_TOP)
          .reduce((sum, s) => sum + s, 0),
      );
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
