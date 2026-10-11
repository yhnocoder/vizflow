import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { SERVING_PRESETS, simulateSglang, type EngineRun } from '../../../src/sim/serve';

type GoldenPass = [number, string, [string, number, number][], string[], string[], number];
const golden: Record<string, GoldenPass[]> = JSON.parse(readFileSync(new URL('../../fixtures/serve/sglang-golden.json', import.meta.url), 'utf8'));

function actual(run: EngineRun) {
  return run.steps.map((s) => ({
    mode: s.mode === 'decode' || s.mode === 'idle' ? s.mode : 'extend',
    scheduled: s.rows.map((r) => [r.id, r.from, r.n]),
    retracted: s.events.filter((e) => e.kind === 'retract').flatMap((e) => e.ids),
    outputs: s.events.filter((e) => e.kind === 'process').flatMap((e) => e.ids).sort(),
  }));
}

function expected(passes: GoldenPass[]) {
  return passes.map(([, mode, scheduled, retracted, outputs]) => ({
    mode, scheduled, retracted, outputs: [...outputs].sort(),
  }));
}

describe('sglang simulator matches the real SGLang scheduler', () => {
  for (const preset of SERVING_PRESETS) {
    for (const mode of ['normal', 'overlap'] as const) {
      it(`${preset.name} ${mode}`, () => {
        const run = simulateSglang({ ...preset.config, overlap: mode === 'overlap' });
        expect(actual(run)).toEqual(expected(golden[`${preset.name}.${mode}`]));
      });
    }
  }
});
