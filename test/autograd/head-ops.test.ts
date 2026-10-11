import { describe, expect, it } from 'vitest';
import { Tensor, checkOp, graph, opDef } from '../../src/core';

function filled(shape: number[], seed: number): Tensor {
  return new Tensor(shape, Float64Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => Math.sin(seed * 7.3 + i * 1.7) * 1.5));
}

describe('ops for multi-head tensors laid out as (T, H * d_h)', () => {
  it('pass the gradient check', () => {
    checkOp(opDef('group_mean')!, [filled([3, 8], 1)], { size: 4 });
    checkOp(opDef('group_mul')!, [filled([3, 8], 2), filled([3, 2], 3)], {});
    checkOp(opDef('tile_mul')!, [filled([3, 8], 4), filled([4], 5)], {});
    checkOp(opDef('tile_mul')!, [filled([3, 8], 6), filled([3, 4], 7)], {});
    checkOp(opDef('rotate_half')!, [filled([3, 8], 8)], { size: 4 });
    checkOp(opDef('narrow_matmul')!, [filled([3, 8], 9), filled([4, 3], 10)], { start: 4 });
  });

  it('compute per-head values', () => {
    const g = graph();
    const x = g.input('x', [1, 8], [[1, 2, 3, 4, 5, 6, 7, 8]]);
    expect(Array.from(g.group_mean(x, 4).value.data)).toEqual([2.5, 6.5]);
    expect(Array.from(g.group_mul(x, g.input('r', [1, 2], [[1, 10]])).value.data)).toEqual([1, 2, 3, 4, 50, 60, 70, 80]);
    expect(Array.from(g.tile_mul(x, g.input('c', [2], [1, -1])).value.data)).toEqual([1, -2, 3, -4, 5, -6, 7, -8]);
    expect(Array.from(g.rotate_half(x, 4).value.data)).toEqual([-3, -4, 1, 2, -7, -8, 5, 6]);
    const k = g.input('k', [4, 1], [[1], [1], [1], [1]]);
    expect(Array.from(g.narrow_matmul(x, k, 4).value.data)).toEqual([26]);
  });
});
