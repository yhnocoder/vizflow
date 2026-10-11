import { describe, expect, it } from 'vitest';
import { type Tensor, graph } from '../../src/core';
import { FLOAT_FORMATS, roundToDtype, roundToFormat, roundUpToPowerOfTwo } from '../../src/autograd/rounding';

function bf16ByBits(x: number): number {
  const f = new Float32Array([x]);
  const u = new Uint32Array(f.buffer);
  u[0] = (u[0] + 0x7fff + ((u[0] >>> 16) & 1)) & 0xffff0000;
  return f[0];
}

describe('rounding', () => {
  it('derives the largest finite value of each format', () => {
    expect(FLOAT_FORMATS.fp8_e4m3.max).toBe(448);
    expect(FLOAT_FORMATS.fp8_e5m2.max).toBe(57344);
    expect(FLOAT_FORMATS.fp16.max).toBe(65504);
    expect(FLOAT_FORMATS.bf16.max).toBe(3.3895313892515355e38);
    expect(FLOAT_FORMATS.fp4_e2m1.max).toBe(6);
  });

  it('rounds bf16 like the round-to-nearest-even bit trick on float32', () => {
    const values = [1, 1.00390625, 1.005859375, 3.14159, -2.71828, 1e-30, 65504.5, 0.1, 123456.789];
    for (const v of values) expect(roundToDtype(v, 'bf16')).toBe(bf16ByBits(v));
  });

  it('rounds fp16 to nearest even with subnormals and overflows to infinity', () => {
    expect(roundToDtype(1 + 2 ** -11, 'fp16')).toBe(1);
    expect(roundToDtype(1 + 3 * 2 ** -11, 'fp16')).toBe(1 + 2 ** -9);
    expect(roundToDtype(2 ** -24, 'fp16')).toBe(2 ** -24);
    expect(roundToDtype(2 ** -26, 'fp16')).toBe(0);
    expect(roundToDtype(65519, 'fp16')).toBe(65504);
    expect(roundToDtype(65520, 'fp16')).toBe(Infinity);
    expect(roundToDtype(-65520, 'fp16')).toBe(-Infinity);
  });

  it('rounds fp8_e4m3 like float8_e4m3fn: max 448, saturating on overflow, subnormal step 2^-9', () => {
    expect(roundToDtype(464, 'fp8_e4m3')).toBe(448);
    expect(roundToDtype(465, 'fp8_e4m3')).toBe(448);
    expect(roundToDtype(-1000, 'fp8_e4m3')).toBe(-448);
    expect(roundToDtype(NaN, 'fp8_e4m3')).toBeNaN();
    expect(roundToDtype(0.3, 'fp8_e4m3')).toBe(0.3125);
    expect(roundToDtype(2 ** -9, 'fp8_e4m3')).toBe(2 ** -9);
    expect(roundToDtype(1.5 * 2 ** -10, 'fp8_e4m3')).toBe(2 ** -9);
    expect(roundToDtype(2 ** -10, 'fp8_e4m3')).toBe(0);
  });

  it('rounds fp8_e5m2 with infinity on overflow', () => {
    expect(roundToDtype(57344, 'fp8_e5m2')).toBe(57344);
    expect(roundToDtype(61440, 'fp8_e5m2')).toBe(Infinity);
    expect(roundToDtype(1.2, 'fp8_e5m2')).toBe(1.25);
  });

  it('saturates fp4_e2m1 and keeps its value grid', () => {
    const grid = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6];
    for (const v of grid) expect(roundToFormat(v, FLOAT_FORMATS.fp4_e2m1)).toBe(v);
    expect(roundToFormat(2.5, FLOAT_FORMATS.fp4_e2m1)).toBe(2);
    expect(roundToFormat(5, FLOAT_FORMATS.fp4_e2m1)).toBe(4);
    expect(roundToFormat(100, FLOAT_FORMATS.fp4_e2m1)).toBe(6);
  });

  it('rounds through float32 first like a cast from float64 in PyTorch', () => {
    expect(roundToDtype(0.1, 'fp32')).toBe(Math.fround(0.1));
    expect(roundToDtype(0.1, 'fp64')).toBe(0.1);
    expect(roundToDtype(-2.7, 'int8')).toBe(-2);
  });

  it('rounds positive numbers up to a power of two', () => {
    expect(roundUpToPowerOfTwo(448)).toBe(512);
    expect(roundUpToPowerOfTwo(0.25)).toBe(0.25);
    expect(roundUpToPowerOfTwo(3e-5)).toBe(2 ** -15);
    expect(() => roundUpToPowerOfTwo(0)).toThrow(/positive/);
  });
});

describe('graph rounding', () => {
  it('leaves values unrounded by default', () => {
    const g = graph({ dims: { V: 2 }, dtype: 'bf16' });
    const x = g.input('x', ['V'], [0.1, 1 / 3]);
    expect([...x.value.data]).toEqual([0.1, 1 / 3]);
    expect(g.rounding).toBe(false);
  });

  it('rounds leaf and op values to each node dtype and recomputes with rounding', () => {
    const g = graph({ dims: { V: 2 }, dtype: 'bf16', rounding: true });
    const x = g.input('x', ['V'], [0.1, 1 / 3]);
    const w = g.param('w', ['V'], [3, 3], { dtype: 'fp32' });
    const y = g.mul(x, w, { name: 'y' });
    const z = g.mul(x, w, { name: 'z', dtype: 'fp8_e4m3' });
    const loss = g.sum(y, null, { name: 'loss' });
    expect([...x.value.data]).toEqual([roundToDtype(0.1, 'bf16'), roundToDtype(1 / 3, 'bf16')]);
    expect(y.dtype).toBe('fp32');
    expect([...y.value.data]).toEqual([...x.value.data].map((v) => Math.fround(v * 3)));
    expect([...z.value.data]).toEqual([...x.value.data].map((v) => roundToDtype(v * 3, 'fp8_e4m3')));
    x.set([1000, 0.2]);
    expect(x.value.data[0]).toBe(1000);
    expect(z.value.data[0]).toBe(448);
    expect([...(w.grad as Tensor).data]).toEqual([...x.value.data]);
    expect(loss.value.data[0]).toBe(Math.fround(Math.fround(3000) + Math.fround(roundToDtype(0.2, 'bf16') * 3)));
  });
});
