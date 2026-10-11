import { describe, expect, it } from 'vitest';
import { type EpTopology, nodeLimitedRouter } from '../../src/sim';
import { mulberry32 } from '../../src/tensor';

const topo: EpTopology = { numScaleoutRanks: 4, numScaleupRanks: 2, numExperts: 32 };
const expertsPerNode = 8;

function topNodes(scores: number[], perNodeTop: number, maxNodes: number) {
  const sums = [0, 1, 2, 3].map((node) =>
    scores
      .slice(node * expertsPerNode, (node + 1) * expertsPerNode)
      .sort((a, b) => b - a)
      .slice(0, perNodeTop)
      .reduce((a, b) => a + b, 0),
  );
  return [0, 1, 2, 3].sort((a, b) => sums[b] - sums[a]).slice(0, maxNodes).sort();
}

describe('nodeLimitedRouter', () => {
  it('scores each node by its top topk/maxNodes experts when that is not 2', () => {
    const topk = 6;
    const maxNodes = 2;
    const seed = 7;
    const routed = nodeLimitedRouter(topo, 16, topk, maxNodes, seed);
    const random = mulberry32(seed);
    let differsFromTopTwo = 0;
    for (const tokens of routed) {
      for (const row of tokens) {
        const scores = Array.from({ length: topo.numExperts }, () => random());
        const expected = topNodes(scores, 3, maxNodes);
        const used = [...new Set(row.map((e) => Math.floor(e / expertsPerNode)))].sort();
        expect(row).toHaveLength(topk);
        expect(used.every((node) => expected.includes(node))).toBe(true);
        if (topNodes(scores, 2, maxNodes).join() !== expected.join()) differsFromTopTwo++;
      }
    }
    expect(differsFromTopTwo).toBeGreaterThan(0);
  });

  it('rejects topk not divisible by maxNodes', () => {
    expect(() => nodeLimitedRouter(topo, 1, 5, 2, 1)).toThrow('not divisible');
  });
});
