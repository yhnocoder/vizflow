import { describe, expect, it } from 'vitest';
import { Tensor, checkOp, graph, opDef } from '../../src/core';

const SEL = [
  [0, 1, 1, 0],
  [1, 0, 1, 0],
  [0, 0, 1, 1],
];

describe('route_rows and route_experts', () => {
  it('list (token, expert) pairs ordered by expert, then by token', () => {
    const g = graph();
    const sel = g.label('sel', SEL);
    const rows = g.route_rows(sel, 2);
    const experts = g.route_experts(sel, 2);
    expect(rows.shape).toEqual([6]);
    expect(Array.from(rows.value.data)).toEqual([1, 0, 0, 1, 2, 2]);
    expect(Array.from(experts.value.data)).toEqual([0, 1, 2, 2, 2, 3]);
  });

  it('take the row count from the dims table', () => {
    const g = graph({ dims: { T: 3, E: 4, K: 2 } });
    const sel = g.label('sel', SEL);
    expect(g.route_rows(sel, 'K').shape).toEqual([6]);
  });

  it('carry no gradient', () => {
    const g = graph({ check: true });
    const x = g.input('x', [3, 4], [[0.1, 0.9, 0.8, 0.2], [0.7, 0.1, 0.6, 0.3], [0.2, 0.3, 0.9, 0.8]]);
    const sel = g.topk_mask(x, 2, -1);
    expect(Array.from(sel.value.data)).toEqual(SEL.flat());
    const rows = g.route_rows(sel, 2);
    g.loss(g.sum(g.mul(x, 2)));
    expect(rows.grad).toBeNull();
  });

  it('reject a row that does not have exactly k ones', () => {
    const g = graph();
    const sel = g.label('sel', [[1, 1, 0], [1, 0, 0]]);
    expect(() => g.route_rows(sel, 2)).toThrow('route_rows: row 1 of sel has 1 ones, expected k = 2');
    expect(() => g.route_experts(sel, 4)).toThrow('route_experts: k 4 is out of range for sel (2, 3)');
    expect(() => g.route_rows(g.label('half', [[0.5, 1, 0.5]]), 2)).toThrow('route_rows: sel[0, 0] is 0.5, expected 0 or 1');
    expect(() => g.route_rows(g.label('flat', [1, 0]), 1)).toThrow('route_rows: sel (2,) must be (T, E)');
  });
});

describe('grouped_mm', () => {
  it('multiplies each row by the weight block of its expert', () => {
    const g = graph();
    const x = g.input('x', [3, 2], [[1, 2], [3, 4], [5, 6]]);
    const w = g.param('w', [2, 1, 2], [[[1, 0]], [[0, 10]]]);
    const group = g.label('group', [1, 0, 1]);
    const out = g.grouped_mm(x, w, group);
    expect(out.shape).toEqual([3, 1]);
    expect(Array.from(out.value.data)).toEqual([20, 3, 60]);
    g.loss(g.sum(out));
    expect(Array.from(w.grad!.data)).toEqual([3, 4, 6, 8]);
    expect(Array.from(x.grad!.data)).toEqual([0, 10, 1, 0, 0, 10]);
  });

  it('matches finite differences and leaves unused expert blocks at zero gradient', () => {
    const def = opDef('grouped_mm')!;
    const x = new Tensor([4, 3], [0.3, -1.2, 0.5, 1.1, 0.4, -0.7, -0.2, 0.9, 1.3, 0.6, -0.5, 0.8]);
    const w = new Tensor([3, 2, 3], Float64Array.from({ length: 18 }, (_, i) => Math.sin(i + 1)));
    const group = new Tensor([4], [2, 0, 2, 2]);
    expect(() => checkOp(def, [x, w, group], {})).not.toThrow();
    const [, gw] = def.backward(new Tensor([4, 2], Float64Array.from({ length: 8 }, () => 1)), [x, w, group], def.forward([x, w, group], {}), {});
    expect(Array.from(gw!.data.subarray(6, 12))).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it('rejects mismatched shapes and out-of-range groups', () => {
    const g = graph();
    const x = g.input('x', [2, 3], [[1, 2, 3], [4, 5, 6]]);
    const w = g.param('w', [2, 4, 3], Array.from({ length: 24 }, () => 0));
    expect(() => g.grouped_mm(x, g.param('w2', [4, 3], Array.from({ length: 12 }, () => 0)), g.label('a', [0, 0]))).toThrow(
      'grouped_mm: x (2, 3) must be (R, d), w (4, 3) must be (E, n, d) and group (2,) must be (R)',
    );
    expect(() => g.grouped_mm(x, w, g.label('b', [0, 2]))).toThrow('grouped_mm: group 2 at [1] is out of range for 2 experts');
  });
});

describe('index_add', () => {
  it('adds rows of src into the rows named by index', () => {
    const g = graph();
    const src = g.input('src', [4, 2], [[1, 2], [3, 4], [5, 6], [7, 8]]);
    const index = g.label('index', [2, 0, 2, 2]);
    const out = g.index_add(src, index, 3);
    expect(out.shape).toEqual([3, 2]);
    expect(Array.from(out.value.data)).toEqual([3, 4, 0, 0, 13, 16]);
    const u = g.input('u', [3, 2], [[1, -1], [2, -2], [3, -3]]);
    g.loss(g.sum(g.mul(out, u)));
    expect(Array.from(src.grad!.data)).toEqual([3, -3, 1, -1, 3, -3, 3, -3]);
  });

  it('matches finite differences', () => {
    const def = opDef('index_add')!;
    const src = new Tensor([5, 2], Float64Array.from({ length: 10 }, (_, i) => Math.cos(i)));
    const index = new Tensor([5], [1, 0, 1, 3, 1]);
    expect(() => checkOp(def, [src, index], { size: 4 })).not.toThrow();
  });

  it('rejects an index that does not fit', () => {
    const g = graph();
    const src = g.input('src', [2, 2], [[1, 2], [3, 4]]);
    expect(() => g.index_add(src, g.label('a', [0, 1, 1]), 2)).toThrow('index_add: index (3,) must be one-dimensional with one entry per row of src (2, 2)');
    expect(() => g.index_add(src, g.label('b', [0, 2]), 2)).toThrow('index_add: index 2 at [1] is out of range for size 2');
  });
});

describe('sorted MoE pieces compose', () => {
  it('dispatch by embedding and combine by index_add are transposes of each other', () => {
    const g = graph();
    const x = g.input('x', [3, 2], [[1, 2], [3, 4], [5, 6]]);
    const sel = g.label('sel', SEL);
    const rows = g.route_rows(sel, 2);
    const dispatched = g.embedding(rows, x);
    const back = g.index_add(dispatched, rows, 3);
    expect(Array.from(back.value.data)).toEqual([2, 4, 6, 8, 10, 12]);
    g.loss(g.sum(back));
    expect(Array.from(x.grad!.data)).toEqual([2, 2, 2, 2, 2, 2]);
  });
});
