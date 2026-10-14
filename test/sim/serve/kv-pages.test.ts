import { describe, expect, it } from 'vitest';
import { simulateSglangPages, simulateVllmPages, type PagedScenario } from '../../../src/sim/serve';

const scenario: PagedScenario = {
  requests: [
    { id: 'A', prompt: ['the', 'cat'], outputs: ['sat', 'on', 'mat'], arrive: 0 },
    { id: 'B', prompt: ['the', 'cat', 'on', 'mat'], outputs: [], arrive: 1 },
  ],
  blockSize: 2,
  numBlocks: 8,
  steps: 4,
};

describe('paged KV cache of two requests that share a prompt prefix', () => {
  it('vLLM shares the full prompt block by its hash and gives each request its own blocks after it', () => {
    const steps = simulateVllmPages(scenario);
    expect(steps[0].requests.map((r) => [r.id, r.table])).toEqual([['A', [1]]]);
    expect(steps[0].blocks![1]).toMatchObject({ refCnt: 1, hashed: true, tokens: ['the', 'cat'] });
    expect(steps[1].scheduled).toEqual([{ id: 'A', tokens: 1, hit: 0 }, { id: 'B', tokens: 2, hit: 2 }]);
    expect(steps[1].requests.map((r) => [r.id, r.table])).toEqual([['A', [1, 2]], ['B', [1, 3]]]);
    expect(steps[1].blocks![1].refCnt).toBe(2);
    expect(steps[1].blocks![1].owners).toEqual(['A', 'B']);
    expect(steps[2].requests.find((r) => r.id === 'B')!.table).toEqual([1, 3, 4]);
    expect(steps[3].requests.find((r) => r.id === 'A')!.table).toEqual([1, 2, 5]);
  });

  it('SGLang matches the prompt in the radix tree token by token and runs the new prefill before the next decode', () => {
    const steps = simulateSglangPages(scenario);
    expect(steps[0].requests.map((r) => [r.id, r.table])).toEqual([['A', [1, 2]]]);
    expect(steps[1].scheduled).toEqual([{ id: 'B', tokens: 2, hit: 2 }]);
    expect(steps[1].requests.map((r) => [r.id, r.table])).toEqual([['A', [1, 2]], ['B', [1, 2, 3, 4]]]);
    expect(steps[2].scheduled.map((s) => s.id)).toEqual(['A', 'B']);
    const keys = steps[1].tree!.nodes.filter((n) => n.parent !== null).map((n) => n.key.join(' '));
    expect(keys).toEqual(['the cat', 'on mat']);
  });
});
