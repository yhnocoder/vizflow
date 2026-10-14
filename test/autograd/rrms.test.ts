import { describe, expect, it } from 'vitest';
import { Tensor, checkOp, graph, opDef } from '../../src/core';

describe('rrms', () => {
  it.each([[[4]], [[2, 3]], [[2, 3, 4]]])('matches the composed forward and backward for %j', (shape) => {
    const values = Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => Math.sin(i + 1));
    const g = graph({ check: false });
    const x = g.input('x', shape, values);
    const r = g.rrms(x, 1e-6);
    expect(g.nodes).toHaveLength(2);
    expect(r.shape).toEqual([...shape.slice(0, -1), 1]);
    const reference = g.rsqrt(g.add(g.mean(g.pow(x, 2), -1, { keepdim: true }), 1e-6));
    expect(r.value.toNested()).toEqual(reference.value.toNested());
    const upstream = g.input('upstream', [...r.shape], Array.from(r.value.data, (_, i) => i + 1));
    g.loss(g.sum(g.mul(r, upstream)));
    const actual = [...x.grad!.data];
    g.loss(g.sum(g.mul(reference, upstream)));
    x.grad!.data.forEach((value, i) => expect(actual[i]).toBeCloseTo(value, 12));
    checkOp(opDef('rrms')!, [new Tensor(shape, Float64Array.from(values))], { eps: 1e-6 });
  });

  it('preserves floating dtype and promotes integer inputs like rsqrt', () => {
    const g = graph({ dtype: 'fp32', check: false });
    for (const dtype of ['bf16', 'fp16', 'fp32', 'fp64', 'int32'] as const) {
      const x = g.input(dtype, [2], [3, 4], { dtype });
      expect(g.rrms(x, 1e-6).dtype).toBe(g.rsqrt(x).dtype);
    }
  });

  it('handles zero rows with positive epsilon and validates the contract', () => {
    const g = graph({ check: false });
    const x = g.input('x', [2, 3], [0, 0, 0, 0, 0, 0]);
    const r = g.rrms(x, 0.25);
    expect(r.value.toNested()).toEqual([[2], [2]]);
    g.loss(g.sum(r));
    expect([...x.grad!.data]).toEqual([0, 0, 0, 0, 0, 0]);
    for (const eps of [-1, NaN, Infinity]) expect(() => g.rrms(x, eps)).toThrow('eps must be a finite non-negative number');
    expect(() => g.rrms(x, undefined as never)).toThrow("missing argument 'eps'");
    expect(() => g.rrms(g.input('scalar', [], 1), 1e-6)).toThrow('non-empty last dimension');
    expect(() => g.rrms(g.input('empty', [2, 0], []), 1e-6)).toThrow('non-empty last dimension');
  });
});
