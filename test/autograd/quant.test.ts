import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { quantMax, roundTo, scaleOf } from '../../src/autograd/quant';

describe('quantization formats', () => {
  it('saturates out-of-range values and rounds integers half to even', () => {
    expect([500, -1e6, Infinity].map((v) => roundTo('fp8_e4m3', v))).toEqual([448, -448, 448]);
    expect(roundTo('fp8_e5m2', 70000)).toBe(57344);
    expect(roundTo('fp4_e2m1', -6.5)).toBe(-6);
    expect([2.5, 3.5, -2.5, 200, -9].map((v) => roundTo('int8', v))).toEqual([2, 4, -2, 127, -9]);
    expect(roundTo('int4', -8.4)).toBe(-7);
    expect(Object.is(roundTo('fp8_e4m3', -1e-5), 0)).toBe(true);
    expect(['fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1', 'int8', 'int4'].map((f) => quantMax(f as 'int8'))).toEqual([448, 57344, 6, 127, 7]);
  });

  it('computes each scale rule', () => {
    expect(scaleOf('fp8_e4m3', 'amax', 6)).toBe(Math.fround(6 / 448));
    expect(scaleOf('fp8_e4m3', 'amax', 0)).toBe(Math.fround(Math.fround(1e-4) / 448));
    expect(scaleOf('fp8_e4m3', 'ue8m0', 6)).toBe(2 ** -6);
    expect(scaleOf('fp8_e4m3', 'mx', 6)).toBe(2 ** -6);
    expect(scaleOf('fp4_e2m1', 'mx', 6)).toBe(1);
    expect(scaleOf('fp4_e2m1', 'mx', 0)).toBe(1);
    expect(scaleOf('fp4_e2m1', 'e4m3', 6)).toBe(1);
    expect(scaleOf('fp4_e2m1', 'e4m3', 7)).toBe(1.125);
    expect(scaleOf('fp4_e2m1', 'e4m3', 1e-4)).toBe(2 ** -9);
    expect(() => scaleOf('int8', 'mx', 1)).toThrow("block_scale: rule 'mx' needs a float format, got 'int8'");
  });
});

describe('block quantization ops', () => {
  it('blocks the last two dimensions, rounding the block count up', () => {
    const g = graph({ dims: { b: 2 } });
    const x = g.input('x', [2, 3, 5], Array.from({ length: 30 }, (_, i) => i - 10));
    const s = g.block_scale(x, 2, 'b', { name: 's' });
    expect([...s.shape]).toEqual([2, 2, 3]);
    const v = g.input('v', [5], [1, -9, 2, 3, 0]);
    expect([...g.block_scale(v, 1, 2, 'int8', 'amax').shape]).toEqual([3]);
    expect(() => g.block_scale(v, 2, 2)).toThrow(/one-dimensional input \(5,\) needs br = 1, got 2/);
  });

  it('gives each op its dtype', () => {
    const g = graph({ dtype: 'bf16' });
    const x = g.input('x', [2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
    const rules = ['amax', 'ue8m0', 'mx', 'e4m3'] as const;
    expect(rules.map((rule) => g.block_scale(x, 1, 2, 'fp8_e4m3', rule).dtype)).toEqual(['fp32', 'e8m0', 'e8m0', 'fp8_e4m3']);
    const s = g.block_scale(x, 1, 2, 'fp4_e2m1', 'e4m3');
    const q = g.quantize(x, s, 1, 2, 'fp4_e2m1');
    expect(q.dtype).toBe('fp4_e2m1');
    expect(g.dequantize(q, s, 1, 2, { dtype: 'bf16' }).dtype).toBe('bf16');
    expect(g.dequantize(q, s, 1, 2).dtype).toBe('fp8_e4m3');
    const e8 = g.block_scale(x, 1, 2, 'fp4_e2m1', 'mx');
    const q4 = g.quantize(x, e8, 1, 2, 'fp4_e2m1');
    expect(() => g.dequantize(q4, e8, 1, 2, { name: 'mix' })).toThrow("dequantize 'mix': cannot promote fp4_e2m1 with e8m0, give the dtype explicitly");
  });

  it('passes the gradient straight through except where the value was clipped', () => {
    const g = graph({ check: true });
    const x = g.input('x', [1, 2], [3, -0.5]);
    const s = g.input('s', [1, 1], [0.005]);
    const q = g.quantize(x, s, 1, 2, { name: 'q' });
    expect([...q.value.data]).toEqual([448, -96]);
    g.loss(g.sum(q));
    expect([...x.grad!.data]).toEqual([0, 200]);
    expect(s.grad!.data[0]).toBe(0);
  });

  it('rejects unknown formats, rules and mismatched scales', () => {
    const g = graph();
    const x = g.input('x', [2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
    expect(() => g.block_scale(x, 1, 2, 'fp8' as 'int8')).toThrow("block_scale: unknown fmt 'fp8', expected one of fp8_e4m3, fp8_e5m2, fp4_e2m1, int8, int4");
    expect(() => g.block_scale(x, 1, 2, { rule: 'max' as 'mx' })).toThrow("block_scale: unknown rule 'max', expected one of amax, ue8m0, mx, e4m3");
    expect(() => g.block_scale(x, 1, 2, 'int8', 'mx')).toThrow("block_scale: rule 'mx' needs a float format, got 'int8'");
    const s = g.block_scale(x, 1, 2);
    expect(() => g.quantize(x, s, 2, 2)).toThrow('quantize: scale (2, 2) does not match the blocks (1, 2) of input (2, 4)');
    expect(() => g.dequantize(x, s, 1, 4)).toThrow('dequantize: scale (2, 2) does not match the blocks (2, 1) of input (2, 4)');
  });
});
