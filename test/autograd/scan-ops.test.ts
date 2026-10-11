import { describe, expect, it } from 'vitest';
import { Tensor, checkOp, graph, opDef } from '../../src/core';

function filled(shape: number[], seed: number): Tensor {
  return new Tensor(shape, Float64Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => Math.sin(seed * 7.3 + i * 1.7) * 1.5));
}

describe('cumsum, unsqueeze, pad, triu, masked_fill, batched matmul and solve_triangular', () => {
  it('pass the gradient check', () => {
    checkOp(opDef('cumsum')!, [filled([2, 3, 2], 1)], { dim: 1 });
    checkOp(opDef('cumsum')!, [filled([3, 4], 2)], { dim: -1 });
    checkOp(opDef('unsqueeze')!, [filled([3, 4], 3)], { dim: 1 });
    checkOp(opDef('pad')!, [filled([2, 3, 2], 4)], { pad: [0, 0, 0, 2] });
    checkOp(opDef('pad')!, [filled([2, 3], 5)], { pad: [1, 1] });
    checkOp(opDef('triu')!, [filled([2, 3, 3], 6)], { diagonal: 1 });
    checkOp(opDef('masked_fill')!, [filled([2, 2, 2], 7), new Tensor([2, 2], [0, 1, 0, 0])], { value: -5 });
    checkOp(opDef('matmul')!, [filled([2, 3, 4], 8), filled([2, 4, 2], 9)], {});
    checkOp(opDef('matmul')!, [filled([2, 3, 4], 10), filled([4, 2], 11)], {});
    checkOp(opDef('solve_triangular')!, [filled([2, 3, 3], 12), filled([2, 3, 2], 13)], { upper: false, unitriangular: true });
    const diagonal = new Tensor([3, 3], [2, 0.3, -0.4, 0.1, 1.5, 0.2, -0.3, 0.5, 2.5]);
    checkOp(opDef('solve_triangular')!, [diagonal, filled([3, 2], 14)], { upper: true, unitriangular: false });
    checkOp(opDef('solve_triangular')!, [diagonal, filled([3, 2], 15)], { upper: false, unitriangular: false });
  });

  it('compute the values torch gives', () => {
    const g = graph();
    const x = g.input('x', [2, 3], [[1, 2, 3], [4, 5, 6]]);
    expect(g.cumsum(x, 1).value.toNested()).toEqual([[1, 3, 6], [4, 9, 15]]);
    expect(g.cumsum(x, 0).value.toNested()).toEqual([[1, 2, 3], [5, 7, 9]]);
    expect(g.unsqueeze(x, -1).shape).toEqual([2, 3, 1]);
    expect(g.unsqueeze(x, 0).shape).toEqual([1, 2, 3]);
    expect(g.pad(x, 0, 0, 0, 1).value.toNested()).toEqual([[1, 2, 3], [4, 5, 6], [0, 0, 0]]);
    expect(g.pad(x, 1, 0).value.toNested()).toEqual([[0, 1, 2, 3], [0, 4, 5, 6]]);
    const ones = g.buffer('ones', [3, 3], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]);
    const mask = g.triu(ones, 1);
    expect(mask.value.toNested()).toEqual([[0, 1, 1], [0, 0, 1], [0, 0, 0]]);
    const s = g.input('s', [3, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    expect(g.masked_fill(s, mask, -Infinity).value.toNested()).toEqual([[1, -Infinity, -Infinity], [4, 5, -Infinity], [7, 8, 9]]);
    const a = g.input('a', [2, 1, 2], [[[1, 2]], [[3, 4]]]);
    const b = g.input('b', [2, 2, 1], [[[1], [1]], [[2], [0]]]);
    expect(g.matmul(a, b).value.toNested()).toEqual([[[3]], [[6]]]);
  });

  it('solve a unit lower triangular system and ignore the diagonal and upper part', () => {
    const g = graph();
    const A = g.input('A', [3, 3], [[9, 9, 9], [2, 9, 9], [1, 3, 9]]);
    const B = g.input('B', [3, 1], [[1], [4], [10]]);
    const X = g.solve_triangular(A, B, { upper: false, unitriangular: true });
    expect(X.value.toNested()).toEqual([[1], [2], [3]]);
    const upper = g.solve_triangular(g.input('U', [2, 2], [[2, 1], [7, 4]]), g.input('C', [2, 1], [[5], [8]]), { upper: true });
    expect(upper.value.toNested()).toEqual([[1.5], [2]]);
    expect(() => g.solve_triangular(A, B, {})).toThrow(/upper must be given/);
    expect(() => g.solve_triangular(A, g.input('D', [2, 1], [[1], [2]]), { upper: false })).toThrow(/must be \(\.\.\., n, n\)/);
  });

  it('reject bad shapes', () => {
    const g = graph();
    const x = g.input('x', [2, 3], [[1, 2, 3], [4, 5, 6]]);
    expect(() => g.pad(x, 1)).toThrow(/pad: \(1\) must be pairs/);
    expect(() => g.triu(g.input('v', [3], [1, 2, 3]))).toThrow(/at least two dimensions/);
    expect(() => g.masked_fill(x, g.label('m', [[[1, 0, 1]], [[0, 1, 0]]]), 0)).toThrow(/must broadcast to input/);
    expect(() => g.matmul(g.input('a', [2, 3, 4], filled([2, 3, 4], 1).toNested()), g.input('b', [3, 4, 2], filled([3, 4, 2], 2).toNested()))).toThrow(/cannot multiply/);
  });
});
