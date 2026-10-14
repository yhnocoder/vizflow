import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { drawToken, uniforms } from '../../src/steps';

interface Fixture {
  logp: number[][];
  cases: { setting: { temperature: number; top_k?: number; top_p?: number }; q: number[][] }[];
}

const fixture = JSON.parse(readFileSync(new URL('../fixtures/torch/sampling.json', import.meta.url), 'utf8')) as Fixture;

describe('sampling', () => {
  it('matches the Hugging Face temperature, top-k and top-p warpers', () => {
    for (const { setting, q } of fixture.cases) {
      const spec = { temperature: setting.temperature, topK: setting.top_k, topP: setting.top_p, seed: 0 };
      fixture.logp.forEach((row, r) => {
        const draw = drawToken(row, spec, 0.5);
        draw.final.forEach((p, v) => expect(p).toBeCloseTo(q[r][v], 12));
      });
    }
  });

  it('picks the token where the cumulative probability in descending order first exceeds u', () => {
    const row = [Math.log(0.5), Math.log(0.3), Math.log(0.2)];
    const spec = { temperature: 1, seed: 0 };
    expect([0.1, 0.49, 0.51, 0.79, 0.81, 0.99].map((u) => drawToken(row, spec, u).token)).toEqual([0, 0, 1, 1, 2, 2]);
    expect(drawToken(row, undefined, 0.99).token).toBe(0);
  });

  it('draws the same numbers for the same seed', () => {
    const a = uniforms(6);
    const b = uniforms(6);
    const xs = [a(), a(), a()];
    expect([b(), b(), b()]).toEqual(xs);
    for (const x of xs) expect(x >= 0 && x < 1).toBe(true);
  });
});
