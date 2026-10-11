import { describe, expect, it } from 'vitest';
import { describeEvent, simulateSglang, type EngineRun } from '../../../src/sim/serve';

const rows = (run: EngineRun, step: number) => run.steps[step - 1].rows.map((r) => `${r.id}${r.from}+${r.n}`);

describe('input validation (sglang d62b9c24 scheduler.py init_req_max_new_tokens, managers/utils.py validate_input_length)', () => {
  it('a prompt of at least kvTokens - 6 tokens is aborted and runs as a 1-token prefill', () => {
    const run = simulateSglang({ budget: 64, kvTokens: 64, overlap: false, requests: [{ id: 'A', arrive: 1, prompt: 58, out: 8 }] });
    expect(run.steps[0].events.filter((e) => e.kind === 'reject')).toEqual([{ kind: 'reject', ids: ['A'] }]);
    expect(rows(run, 1)).toEqual(['A0+1']);
    expect(run.steps[0].requests[0].prompt).toBe(1);
    expect(run.requests[0]).toMatchObject({ prefillTokens: 1, decodeTokens: 0 });
    expect(run.steps[0].events.filter((e) => e.kind === 'finish')).toEqual([{ kind: 'finish', ids: ['A'] }]);
    expect(describeEvent(run.steps[0].events.find((e) => e.kind === 'reject')!, 'sglang')).toContain('A');
  });

  it('max_new_tokens is clipped to kvTokens - prompt - 2', () => {
    const run = simulateSglang({ budget: 64, kvTokens: 64, overlap: false, requests: [{ id: 'A', arrive: 1, prompt: 40, out: 100 }] });
    expect(run.requests[0]).toMatchObject({ decodeTokens: 21 });
    expect(run.steps.at(-1)!.events.filter((e) => e.kind === 'finish')).toEqual([{ kind: 'finish', ids: ['A'] }]);
  });

  it('requests larger than the KV pool never make the simulator throw', () => {
    const config = {
      budget: 16, kvTokens: 64, overlap: true, policy: 'lpm' as const,
      requests: [
        { id: 'A', arrive: 1, prompt: 2000, out: 2000 },
        { id: 'B', arrive: 1, prompt: 50, shared: 40, group: 'g', out: 2000 },
        { id: 'C', arrive: 2, prompt: 50, shared: 40, group: 'g', out: 2000 },
        { id: 'D', arrive: 3, prompt: 30, out: 30 },
      ],
    };
    const run = simulateSglang(config);
    expect(run.requests.every((r) => r.finished !== null)).toBe(true);
    expect(Math.max(...run.steps.map((s) => s.kvUsed))).toBeLessThanOrEqual(64);
  });
});

describe('prefill admission budget (sglang d62b9c24 schedule_policy.py add_one_req, prefill_budget.py PrefillBudget)', () => {
  it('a request that will be chunked is still checked against its whole extend length plus max_new_tokens', () => {
    const run = simulateSglang({
      budget: 64, kvTokens: 512, overlap: false,
      requests: [{ id: 'A', arrive: 1, prompt: 200, out: 100 }, { id: 'B', arrive: 1, prompt: 200, out: 120 }],
    });
    expect(rows(run, 1)).toEqual(['A0+64']);
    expect(run.steps.slice(0, 4).flatMap((s) => s.rows.map((r) => r.id))).not.toContain('B');
  });
});
