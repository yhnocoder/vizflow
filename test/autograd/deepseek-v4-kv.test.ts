import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
// @ts-expect-error untyped example module
import { buildStep, stateBefore, stepOf, windowIdxs } from '../../examples/kv/deepseek-v4.js';

const fixture = JSON.parse(readFileSync(resolve(import.meta.dirname, '../../examples/kv/deepseek-v4-fixture.json'), 'utf8'));
const decode = (v: unknown): unknown => (v === 'inf' ? Infinity : v === '-inf' ? -Infinity : Array.isArray(v) ? v.map(decode) : v);
const flat = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v as number]);
const TOL = 2e-5;

function close(actual: ArrayLike<number>, expected: unknown, what: string) {
  const e = flat(decode(expected));
  expect(actual.length, what).toBe(e.length);
  Array.from(actual).forEach((a, i) => {
    if (!Number.isFinite(e[i])) expect(a, `${what}[${i}]`).toBe(e[i]);
    else expect(Math.abs(a - e[i]), `${what}[${i}]: ${a} vs ${e[i]}`).toBeLessThan(TOL * Math.max(1, Math.abs(e[i])));
  });
}

describe(`DeepSeek V4 attention decode matches the official inference code (${fixture.source})`, () => {
  for (const kind of ['swa', 'csa', 'hca']) {
    for (let t = fixture.prefill; t < fixture.prefill + fixture.layers[kind].steps.length; t++) {
      it(`${kind} t=${t}`, () => {
        const { N, info } = buildStep(fixture, kind, t);
        const step = stepOf(fixture, kind, t);
        expect(windowIdxs(fixture.config.window_size, t)).toEqual(step.topk_idxs.slice(0, fixture.config.window_size));
        expect(Array.from(N.idx.value.data)).toEqual(step.topk_idxs);
        close(N.o0.value.data, step.o, 'attention output before inverse rope');
        close(N.y.value.data, step.y, 'layer output');
        close(N.cacheF.value.data, step.after.kv_cache, 'kv_cache after');
        if (kind === 'csa') close(N.idxCache.value.data, step.after.index_cache, 'indexer cache after');
        if (info.ratio > 0) {
          const after = decode(step.after.kv_state) as number[][];
          const written = info.mainComp.sKv1.value;
          const r = info.ratio;
          const shift = info.emits ? r : 0;
          if (info.emits && r === 4) close(written.data.slice(r * written.shape[1]), after.slice(0, r), 'overlap state shifted forward');
          if (!info.emits) close(written.data, after, 'compressor kv_state after');
          if (info.emits && r !== 4) expect(shift).toBe(r);
        }
        expect(stateBefore(fixture, kind, t).kv_cache.length).toBe(info.nCache);
      });
    }
  }
});
