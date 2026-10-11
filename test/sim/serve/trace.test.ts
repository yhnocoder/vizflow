import { describe, expect, it } from 'vitest';
import { simulateSglang, simulateVllm, type EngineRun, type ServingConfig } from '../../../src/sim/serve';

const three: ServingConfig = {
  budget: 512, kvTokens: 65536, overlap: true, policy: 'lpm',
  requests: [
    { id: 'A', arrive: 1, prompt: 1000, out: 8 },
    { id: 'B', arrive: 1, prompt: 20, shared: 16, group: 'p', out: 8 },
    { id: 'C', arrive: 1, prompt: 20, shared: 16, group: 'p', out: 8 },
  ],
};

const rows = (run: EngineRun, step: number) => run.steps[step - 1].rows.map((r) => `${r.id}${r.from}+${r.n}`);

describe('report trace', () => {
  it('vllm shares the prefix block registered in the same step', () => {
    const run = simulateVllm(three);
    expect(rows(run, 1)).toEqual(['A0+512']);
    expect(rows(run, 2)).toEqual(['C16+4', 'B0+20', 'A512+488']);
    expect(rows(run, 3)).toEqual(['A1000+1', 'B20+1', 'C20+1']);
    expect(run.steps[2]).toMatchObject({ graph: 'full', padded: 1 });
    expect(run.requests.map((r) => r.firstToken)).toEqual([2, 2, 2]);
  });

  it('sglang runs prefill before decode and recomputes the shared prefix', () => {
    const run = simulateSglang(three);
    expect(rows(run, 1)).toEqual(['A0+512']);
    expect(rows(run, 2)).toEqual(['A512+488', 'B0+20', 'C0+4']);
    expect(rows(run, 3)).toEqual(['C4+16']);
    expect(rows(run, 4)).toEqual(['A1000+1', 'B20+1', 'C20+1']);
    expect(run.requests.map((r) => r.firstToken)).toEqual([2, 2, 3]);
    expect(run.requests.reduce((s, r) => s + r.duplicateTokens, 0)).toBe(16);
  });
});
