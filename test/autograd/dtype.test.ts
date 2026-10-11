import { describe, expect, it } from 'vitest';
import { DTYPES, DTYPE_BITS, Tensor, checkOp, defineOp, graph } from '../../src/core';
import { roundToDtype } from '../../src/autograd/rounding';

describe('dtype', () => {
  it('uses the graph dtype for leaves, int64 for labels and promotes floating inputs for ops', () => {
    const g = graph({ dims: { T: 2, V: 3 }, dtype: 'bf16' });
    const ids = g.input('ids', ['T'], [0, 2], { dtype: 'int64' });
    const E = g.param('E', ['V', 'V'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    const x = g.embedding(ids, E, { name: 'x' });
    const z = g.log_softmax(x, -1, { name: 'z', dtype: 'fp32' });
    const y = g.label('y', [1, 0], { classes: 3 });
    const loss = g.nll_loss(z, y, { name: 'loss' });
    const back = g.mul(z, x, { name: 'back' });
    expect([ids, E, x, z, y, loss, back].map((n) => n.dtype)).toEqual(['int64', 'bf16', 'bf16', 'fp32', 'int64', 'fp32', 'fp32']);
    expect(E.T.dtype).toBe('bf16');
  });

  it('defaults to fp32 and promotes bf16 with fp16 to fp32', () => {
    const g = graph({ dims: { V: 2 } });
    const a = g.input('a', ['V'], [1, 2], { dtype: 'bf16' });
    const b = g.input('b', ['V'], [1, 2], { dtype: 'fp16' });
    expect(g.dtype).toBe('fp32');
    expect(g.add(a, b).dtype).toBe('fp32');
    expect(g.add(a, a).dtype).toBe('bf16');
  });

  it('promotes integers by width independent of order and rejects mixing the two fp8 formats', () => {
    const g = graph({ dims: { V: 2 } });
    const make = (name: string, dtype: string) => g.input(name, ['V'], [1, 2], { dtype });
    const i32 = make('i32', 'int32');
    const i64 = make('i64', 'int64');
    const b = make('b', 'bool');
    const u8 = make('u8', 'uint8');
    const i8 = make('i8', 'int8');
    expect([g.add(i32, i64), g.add(i64, i32), g.add(b, i32), g.add(u8, i8)].map((n) => n.dtype)).toEqual(['int64', 'int64', 'int32', 'int16']);
    const e4 = make('e4', 'fp8_e4m3');
    const e5 = make('e5', 'fp8_e5m2');
    expect(() => g.add(e4, e5, { name: 'mix' })).toThrow(/add 'mix': cannot promote fp8_e4m3 with fp8_e5m2/);
    expect(g.add(e4, e5, { name: 'cast', dtype: 'bf16' }).dtype).toBe('bf16');
  });

  it('rejects unknown dtypes', () => {
    expect(() => graph({ dtype: 'float' })).toThrow(/unknown dtype 'float'/);
    const g = graph({ dims: { V: 2 } });
    expect(() => g.input('a', ['V'], [1, 2], { dtype: 'f32' })).toThrow(/node 'a': unknown dtype 'f32'/);
  });

  it('ranks the narrow formats below fp8 and int4 between bool and the 8-bit integers', () => {
    const g = graph({ dims: { V: 2 } });
    const make = (name: string, dtype: string) => g.input(name, ['V'], [1, 2], { dtype });
    const [f4, e8, e4, i4, b, i8] = [['f4', 'fp4_e2m1'], ['e8', 'e8m0'], ['e4', 'fp8_e4m3'], ['i4', 'int4'], ['b', 'bool'], ['i8', 'int8']].map(([n, d]) => make(n, d));
    expect([g.add(f4, e4), g.add(e8, e4), g.add(i4, b), g.add(i4, i8), g.add(f4, i8)].map((n) => n.dtype)).toEqual(['fp8_e4m3', 'fp8_e4m3', 'int4', 'int8', 'fp4_e2m1']);
    expect(() => g.add(f4, e8, { name: 'mix' })).toThrow("add 'mix': cannot promote fp4_e2m1 with e8m0, give the dtype explicitly");
  });

  it('gives the bit width of every dtype', () => {
    expect(Object.keys(DTYPE_BITS).sort()).toEqual([...DTYPES].sort());
    expect([DTYPE_BITS.bool, DTYPE_BITS.int4, DTYPE_BITS.fp4_e2m1, DTYPE_BITS.e8m0, DTYPE_BITS.bf16, DTYPE_BITS.fp64]).toEqual([8, 4, 4, 8, 16, 64]);
  });

  it('rounds to fp4_e2m1 and to int4 by truncating toward zero and clamping to [-8, 7]', () => {
    expect([2.5, 5.2, 7, -0.3].map((v) => roundToDtype(v, 'fp4_e2m1'))).toEqual([2, 6, 6, -0.5]);
    expect([-3.7, 3.7, 7.9, 9, -8.5, -20, -0.4].map((v) => roundToDtype(v, 'int4'))).toEqual([-3, 3, 7, 7, -8, -8, 0]);
  });

  it('rounds to e8m0 the way torch converts to float8_e8m0fnu', () => {
    const cases: [number, number][] = [
      [0.0078125, 0.0078125],
      [1.4142135, 1],
      [1.5, 2],
      [1.49999999999, 2],
      [2.9, 2],
      [3, 4],
      [0.75, 1],
      [0.74, 0.5],
      [6, 8],
      [-3, 4],
      [-1, 1],
      [0, 2 ** -127],
      [1e-45, 2 ** -127],
      [2 ** -128, 2 ** -127],
      [2 ** 127 * 1.4999, 2 ** 127],
    ];
    expect(cases.map(([v]) => roundToDtype(v, 'e8m0'))).toEqual(cases.map(([, r]) => r));
    expect([2 ** 127 * 1.5, Infinity, NaN].map((v) => roundToDtype(v, 'e8m0'))).toEqual([NaN, NaN, NaN]);
  });

  it('uses the dtype function of an op definition and only checks shapes for straight-through ops', () => {
    defineOp<{ fmt: 'int8' | 'fp4_e2m1' }>({
      name: 'test_round_ste',
      args: ['input', 'fmt'],
      tensors: ['input'],
      defaults: { fmt: 'int8' },
      ste: true,
      dtype: (attrs) => attrs.fmt,
      shape: ([a]) => [...a],
      forward: ([a]) => new Tensor(a.shape, a.data.map((v) => Math.round(v))),
      backward: (gradOut) => [gradOut.clone()],
    });
    const g = graph({ dims: { V: 2 }, dtype: 'bf16', check: true });
    const call = (g as unknown as Record<string, (...a: unknown[]) => { dtype: string }>).test_round_ste;
    const x = g.input('x', ['V'], [0.3, 1.7]);
    expect(call(x).dtype).toBe('int8');
    expect(call(x, { fmt: 'fp4_e2m1' }).dtype).toBe('fp4_e2m1');
    expect(call(x, { dtype: 'bf16' }).dtype).toBe('bf16');
    const op = { name: 'step', args: ['input'], tensors: ['input'], shape: ([a]: number[][]) => [...a], forward: ([a]: Tensor[]) => new Tensor(a.shape, a.data.map((v) => Math.round(v))), backward: (gradOut: Tensor) => [gradOut.clone()] };
    const input = new Tensor([2], Float64Array.from([0.3, 1.7]));
    expect(() => checkOp(op, [input], {})).toThrow(/gradient check failed/);
    expect(() => checkOp({ ...op, ste: true }, [input], {})).not.toThrow();
    expect(() => checkOp({ ...op, ste: true, backward: () => [new Tensor([3])] }, [input], {})).toThrow(/backward returned \(3,\) for shape \(2,\)/);
  });
});
