import { describe, expect, it } from 'vitest';
import { simulateVllm, type EngineRun } from '../../../src/sim/serve';

const rows = (run: EngineRun, step: number) => run.steps[step - 1].rows.map((r) => `${r.id}${r.from}+${r.n}`).sort();
const eventsAt = (run: EngineRun, step: number, kind: string) => run.steps[step - 1].events.filter((e) => e.kind === kind);

describe('preemption under a tiny pool (vllm f26d9f4 scheduler.py)', () => {
  const run = simulateVllm({
    budget: 64, kvTokens: 64, overlap: false,
    requests: [{ id: 'A', arrive: 1, prompt: 24, out: 12 }, { id: 'B', arrive: 1, prompt: 24, out: 12 }],
  });

  it('allocate_slots failure preempts running[-1], frees its blocks and resets its computed tokens (scheduler.py:802, :1604)', () => {
    expect(rows(run, 10)).toEqual(['A32+1']);
    expect(eventsAt(run, 10, 'preempt')).toEqual([{ kind: 'preempt', ids: ['B'] }]);
    expect(run.steps[9].requests.find((r) => r.id === 'B')).toMatchObject({ status: 'preempted', computed: 0 });
    expect(run.steps[9].kvUsed).toBe(48);
  });

  it('admission needs the whole sequence to fit, so B waits while A runs (scheduler_reserve_full_isl, scheduler.py:1270)', () => {
    expect(eventsAt(run, 11, 'blocked')).toEqual([{ kind: 'blocked', ids: ['B'] }]);
    expect(eventsAt(run, 12, 'finish')).toEqual([{ kind: 'finish', ids: ['A'] }]);
  });

  it('the resumed request recomputes prompt and outputs with a prefix hit on its surviving block (scheduler.py:1129)', () => {
    expect(eventsAt(run, 13, 'resume')).toEqual([{ kind: 'resume', ids: ['B'], hit: 16 }]);
    expect(rows(run, 13)).toEqual(['B16+17']);
    expect(run.requests.find((r) => r.id === 'B')).toMatchObject({ finished: 15, evictions: 1 });
  });
});

describe('chunked prefill (vllm f26d9f4 scheduler.py)', () => {
  it('a 40-token prompt is prefilled in chunks of the token budget and samples only after the last chunk', () => {
    const run = simulateVllm({ budget: 16, kvTokens: 1024, overlap: false, requests: [{ id: 'A', arrive: 1, prompt: 40, out: 2 }] });
    expect(run.steps.map((s) => [s.rows.map((r) => `${r.from}+${r.n}`), s.rows.some((r) => r.sampled)])).toEqual([
      [['0+16'], false],
      [['16+16'], false],
      [['32+8'], true],
      [['40+1'], true],
    ]);
    expect(run.steps.map((s) => s.kvUsed)).toEqual([16, 32, 48, 48]);
  });
});
