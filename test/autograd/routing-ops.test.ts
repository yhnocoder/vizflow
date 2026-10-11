import { describe, expect, it } from 'vitest';
import { type Node, type Tensor, checkOp, graph, opDef } from '../../src/core';
import { Tensor as T } from '../../src/tensor';

const values = (t: Tensor | null) => Array.from((t as Tensor).data);

describe('ops that route tokens to experts, with hand-computed values', () => {
  it('topk_indices lists the largest entries first, lower index first on ties, with no gradient', () => {
    const g = graph({ check: true });
    const p = g.input('p', [2, 4], [[0.1, 0.4, 0.4, 0.1], [0.7, 0.05, 0.15, 0.1]]);
    const idx = g.topk_indices(p, 2, -1);
    expect(idx.shape).toEqual([2, 2]);
    expect(values(idx.value)).toEqual([1, 2, 0, 2]);
    expect(idx.dtype).toBe('int64');
    expect(values(g.topk_indices(p, 1, 0).value)).toEqual([1, 0, 0, 0]);
    g.loss(g.sum(g.gather(p, -1, idx)));
    expect(values(p.grad)).toEqual([0, 1, 1, 0, 1, 0, 1, 0]);
    expect(() => g.topk_indices(p, 5)).toThrow('topk_indices: k 5 is out of range for input (2, 4) along dim 1');
  });

  it('one_hot, permute, sum over two dims and greater give the expert mask and the experts that were hit', () => {
    const g = graph({ check: true });
    const idx = g.label('idx', [[1, 0], [2, 1], [1, 2]]);
    const mask = g.one_hot(idx, 4);
    expect(mask.shape).toEqual([3, 2, 4]);
    expect(mask.dtype).toBe('int64');
    expect(values(mask.value)).toEqual([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    const maskp = g.permute(mask, 2, 1, 0);
    expect(maskp.shape).toEqual([4, 2, 3]);
    expect(values(maskp.value)).toEqual([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
    const count = g.sum(maskp, [-1, -2]);
    expect(values(count.value)).toEqual([1, 3, 2, 0]);
    expect(count.dtype).toBe('int64');
    const hit = g.greater(count, 0);
    expect(values(hit.value)).toEqual([1, 1, 1, 0]);
    expect(hit.dtype).toBe('bool');
    expect(() => g.one_hot(idx, 2)).toThrow('one_hot: class 2 at [1, 0] is not in 0..1');
    expect(() => g.permute(mask, 0, 0, 1)).toThrow('permute: dims (0, 0, 1) repeat a dimension');
  });

  it('nonzero_static lists indices in row-major order, pads with fill_value and truncates', () => {
    const g = graph();
    const m = g.label('m', [[[0, 1], [1, 0]], [[0, 0], [0, 1]]]);
    const nz = g.nonzero_static(m, { size: 4 });
    expect(nz.shape).toEqual([4, 3]);
    expect(nz.dtype).toBe('int64');
    expect(values(nz.value)).toEqual([0, 0, 1, 0, 1, 0, 1, 1, 1, -1, -1, -1]);
    expect(values(g.nonzero_static(m, 2).value)).toEqual([0, 0, 1, 0, 1, 0]);
    expect(values(g.nonzero_static(m, { size: 4, fill_value: 9 }).value).slice(9)).toEqual([9, 9, 9]);
    expect(() => g.nonzero_static(m, 2, { size: 2 })).toThrow("nonzero_static: argument 'size' is given both by position and by keyword");
  });

  it('index picks rows and sums the gradient of a row used twice', () => {
    const g = graph({ check: true });
    const x = g.input('x', [3, 2], [[1, 2], [3, 4], [5, 6]]);
    const tok = g.label('tok', [2, 0, 2]);
    const xs = g.index(x, [tok]);
    expect(xs.shape).toEqual([3, 2]);
    expect(values(xs.value)).toEqual([5, 6, 1, 2, 5, 6]);
    const pos = g.label('pos', [1, 0, -1]);
    const w = g.index(x, [tok, pos]);
    expect(values(w.value)).toEqual([6, 1, 6]);
    g.loss(g.add(g.sum(xs), g.sum(w)));
    expect(values(x.grad)).toEqual([2, 1, 0, 0, 2, 4]);
    expect(() => g.index(x, [g.label('bad', [3])])).toThrow('index: index 3 at [0] is out of range for input (3, 2) along dim 0');
  });

  it('index_add adds rows of the source into the rows that index names', () => {
    const g = graph({ check: true });
    const z = g.input('z', [2, 2], [[0, 0], [10, 10]]);
    const tok = g.label('tok', [1, 0, 1]);
    const y = g.input('y', [3, 2], [[1, 2], [3, 4], [5, 6]]);
    const out = g.index_add(z, 0, tok, y);
    expect(values(out.value)).toEqual([3, 4, 16, 18]);
    const up = g.input('up', [2, 2], [[1, 2], [3, 4]]);
    g.loss(g.sum(g.mul(out, up)));
    expect(values(z.grad)).toEqual([1, 2, 3, 4]);
    expect(values(y.grad)).toEqual([3, 4, 1, 2, 3, 4]);
  });

  it('grouped_linear multiplies each row by the transpose of its own group of the weight', () => {
    const g = graph({ check: true });
    const x = g.input('x', [3, 2], [[1, 2], [3, 4], [5, 6]]);
    const W = g.input('W', [2, 1, 2], [[[1, 0]], [[0, 1]]]);
    const ge = g.label('ge', [1, 0, 1]);
    const out = g.grouped_linear(x, W, ge);
    expect(out.shape).toEqual([3, 1]);
    expect(values(out.value)).toEqual([2, 3, 6]);
    g.loss(g.sum(out));
    expect(values(x.grad)).toEqual([0, 1, 1, 0, 0, 1]);
    expect(values(W.grad)).toEqual([3, 4, 6, 8]);
    expect(() => g.grouped_linear(x, W, g.label('wide', [0, 2, 1]))).toThrow('grouped_linear: group 2 of row 1 is out of range for weight (2, 1, 2)');
  });

  it('passes the finite-difference check on random values', () => {
    const r = (shape: number[], seed: number) => new T(shape, Float64Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => Math.sin(seed + i * 1.7)));
    checkOp(opDef('grouped_linear')!, [r([5, 3], 1), r([4, 2, 3], 2), T.from([0, 3, 3, 1, 0])], {});
    checkOp(opDef('index_add')!, [r([3, 2], 3), T.from([2, 0, 2, 2]), r([4, 2], 4)], { dim: 0 });
    checkOp(opDef('index_add')!, [r([2, 3], 5), T.from([2, 2]), r([2, 2], 6)], { dim: 1 });
    checkOp(opDef('index')!, [r([3, 2, 2], 7), T.from([2, 0, 2]), T.from([1, 1, 0])], {});
    checkOp(opDef('permute')!, [r([2, 3, 4], 8)], { dims: [2, 0, 1] });
  });
});

describe('keyword arguments and output dtypes of ops', () => {
  it('accepts any non-tensor argument by keyword, as torch does', () => {
    const g = graph();
    const x = g.input('x', [2, 3], [[1, 2, 3], [4, 5, 6]]);
    const softmax = g.softmax as unknown as (...args: unknown[]) => Node;
    expect(values(softmax(x, { dim: 0 }).value)).toEqual(values(g.softmax(x, 0).value));
    expect(() => softmax(x)).toThrow("softmax: missing argument 'dim'");
  });

  it('keeps an explicit dtype over the dtype an op declares', () => {
    const g = graph({ dtype: 'bf16' });
    const p = g.input('p', [3], [1, 3, 2]);
    expect(g.topk_indices(p, 2).dtype).toBe('int64');
    expect(g.topk_indices(p, 2, -1, { dtype: 'int32' }).dtype).toBe('int32');
    expect(g.index(p, [g.topk_indices(p, 2)]).dtype).toBe('bf16');
  });
});
