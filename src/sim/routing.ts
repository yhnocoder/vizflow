import type { EpTopology } from './deepep';
import { mulberry32 } from '../tensor';

export const NODE_SCORE_TOP = 2;


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
