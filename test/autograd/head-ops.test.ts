import { describe, expect, it } from 'vitest';
import { Tensor, checkOp, graph, opDef } from '../../src/core';

function filled(shape: number[], seed: number): Tensor {
  return new Tensor(shape, Float64Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => Math.sin(seed * 7.3 + i * 1.7) * 1.5));
}

function lossOf(build: (x: Tensor) => number, x: Tensor, j: number, delta: number): number {
  const changed = x.clone();
  changed.data[j] += delta;
  return build(changed);
}

describe('ops for multi-head tensors laid out as (T, H, d_h)', () => {
  it('pass the gradient check', () => {
    checkOp(opDef('view')!, [filled([3, 8], 1)], { shape: [3, 2, 4] });
    checkOp(opDef('view')!, [filled([3, 2, 4], 2)], { shape: [3, -1] });
    checkOp(opDef('select')!, [filled([3, 2, 4], 3)], { dim: 1, index: 1 });
    checkOp(opDef('select')!, [filled([3, 2, 4], 4)], { dim: -1, index: -1 });
    checkOp(opDef('select_matmul')!, [filled([3, 2, 4], 5), filled([4, 3], 6)], { index: 1 });
    checkOp(opDef('rotate_half')!, [filled([3, 2, 4], 7)], {});
  });

  it('reshape, select and multiply one head', () => {
    const g = graph({ dims: { T: 1, H: 2, dh: 4 } });
    const x = g.input('x', [1, 8], [[1, 2, 3, 4, 5, 6, 7, 8]]);
    const heads = g.view(x, 'T', 'H', 'dh');
    expect(heads.shape).toEqual([1, 2, 4]);
    expect(g.view(x, [1, -1, 2]).shape).toEqual([1, 4, 2]);
    expect(Array.from(heads.value.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(g.select(heads, 1, 1).value.data)).toEqual([5, 6, 7, 8]);
    expect(Array.from(g.rotate_half(heads).value.data)).toEqual([-3, -4, 1, 2, -7, -8, 5, 6]);
    const k = g.input('k', [4, 1], [[1], [1], [1], [1]]);
    expect(Array.from(g.select_matmul(heads, k, 1).value.data)).toEqual([26]);
    expect(() => g.view(x, 3, 3)).toThrow(/view: shape \(3, 3\) is invalid for input \(1, 8\)/);
    expect(() => g.select(heads, 1, 2)).toThrow(/select: index 2 is out of range/);
    expect(() => g.select_matmul(heads, k, 2)).toThrow(/select_matmul: cannot multiply/);
  });

  it('take finite-difference gradients through a per-head RMSNorm on rank-3 tensors', () => {
    const T = 3;
    const H = 2;
    const dh = 4;
    const values = filled([T, H * dh], 11);
    const weight = filled([dh], 12);
    const run = (x0: Tensor) => {
      const g = graph();
      const x = g.input('x', [T, H * dh], x0.toNested());
      const w = g.input('w', [dh], weight.toNested());
      const q = g.view(x, T, H, dh);
      const r = g.rsqrt(g.add(g.mean(g.pow(q, 2), -1, { keepdim: true }), 1e-6));
      const out = g.mul(g.mul(q, r), w);
      const cos = g.input('cos', [T, 1, dh], filled([T, 1, dh], 13).toNested());
      const rot = g.add(g.mul(out, cos), g.rotate_half(out));
      const head = g.select_matmul(rot, g.transpose(g.select(rot, 1, 0), 0, 1), 1);
      const loss = g.loss(g.sum(g.mul(head, head)));
      return { g, x, w, loss };
    };
    const { x, w, loss } = run(values);
    expect(w.shape).toEqual([dh]);
    const value = (x0: Tensor) => run(x0).loss.value.data[0];
    expect(loss.value.data[0]).toBeCloseTo(value(values), 12);
    for (let j = 0; j < values.size; j++) {
      const numeric = (lossOf(value, values, j, 1e-6) - lossOf(value, values, j, -1e-6)) / 2e-6;
      expect(Math.abs(x.grad!.data[j] - numeric), `x[${j}]`).toBeLessThan(1e-6 + 1e-5 * Math.abs(numeric));
    }
  });
});
