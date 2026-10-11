import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { SERVING_PRESETS, simulateVllm, type EngineRun } from '../../src/serving/sim';

type GoldenStep = [number, [string, number, number][], string[], Record<string, number>, number];
const golden: Record<string, GoldenStep[]> = JSON.parse(readFileSync(new URL('../fixtures/serving/vllm-golden.json', import.meta.url), 'utf8'));

function actual(run: EngineRun) {
  return run.steps.map((s) => {
    const sortKey = (a: [string, number, number], b: [string, number, number]) => a[0].localeCompare(b[0]);
    const scheduled = s.rows.map((r) => [r.id, r.from, r.n] as [string, number, number]).sort(sortKey);
    const preempted = s.events.filter((e) => e.kind === 'preempt').flatMap((e) => e.ids).sort();
    const outputs = Object.fromEntries(s.events.filter((e) => e.kind === 'process').flatMap((e) => e.ids).map((id) => [id, 1]));
    return { step: s.step, scheduled, preempted, outputs, free: (run.kvCapacity - s.kvUsed) / 16 };
  });
}

function expected(steps: GoldenStep[]) {
  return steps.map(([step, scheduled, preempted, outputs, free]) => ({
    step, scheduled: [...scheduled].sort((a, b) => a[0].localeCompare(b[0])), preempted: [...preempted].sort(), outputs, free,
  }));
}

describe('vllm simulator matches the real vLLM scheduler', () => {
  for (const preset of SERVING_PRESETS) {
    for (const mode of ['sync', 'async'] as const) {
      it(`${preset.name} ${mode}`, () => {
        const run = simulateVllm({ ...preset.config, overlap: mode === 'async' });
        expect(actual(run)).toEqual(expected(golden[`${preset.name}.${mode}`]));
      });
    }
  }
});
