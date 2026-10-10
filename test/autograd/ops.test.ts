import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';

const close = (actual: unknown, expected: number[] | number[][], digits = 10) => {
  const flat = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v as number]);
  const a = flat(actual);
  const e = flat(expected);
  expect(a.length).toBe(e.length);
  a.forEach((v, i) => expect(v).toBeCloseTo(e[i], digits));
};

describe('reductions', () => {
  it('sum and mean over all elements, one dim or several dims', () => {
    const g = graph();
    const x = g.input('x', [2, 3], [[1, 2, 3], [4, 5, 6]]);
    expect(g.sum(x).shape).toEqual([]);
    expect(g.sum(x).value.toNested()).toBe(21);
    expect(g.sum(x, -1).value.toNested()).toEqual([6, 15]);
    expect(g.sum(x, -1, { keepdim: true }).value.toNested()).toEqual([[6], [15]]);
    expect(g.sum(x, [0, 1], { keepdim: true }).value.toNested()).toEqual([[21]]);
    expect(g.sum(x, { keepdim: true }).value.toNested()).toEqual([[21]]);
    expect(g.sum(x, { dim: 0 }).value.toNested()).toEqual([5, 7, 9]);
    expect(g.mean(x).value.toNested()).toBe(3.5);
    expect(g.mean(x, 0).value.toNested()).toEqual([2.5, 3.5, 4.5]);
    expect(g.mean(x, [1], { keepdim: true }).value.toNested()).toEqual([[2], [5]]);
    expect(g.sum(x, -1).attrs).toEqual({ dim: -1, keepdim: false });
  });

  it('max returns values and sends the gradient to the first maximum', () => {
    const g = graph({ check: false });
    const x = g.input('x', [2, 3], [[1, 3, 3], [4, 2, 4]]);
    const m = g.max(x, 1);
    expect(m.value.toNested()).toEqual([3, 4]);
    expect(g.max(x, -2, { keepdim: true }).value.toNested()).toEqual([[4, 3, 4]]);
    g.loss(g.sum(m));
    expect(x.grad!.toNested()).toEqual([[0, 1, 0], [1, 0, 0]]);
    expect(() => g.max(x, 2)).toThrow('out of range');
    const checked = graph({ check: true });
    const tied = checked.input('x', [2], [3, 3]);
    expect(() => checked.max(tied, 0)).toThrow(/^gradient check failed: op 'max' \(node 'max1'\), input 0 'input' at \[0\]/);
  });

  it('logsumexp matches the direct formula and subtracts the maximum', () => {
    const g = graph();
    const x = g.input('x', [1, 3], [[1000, 1000, 1000]]);
    const L = g.logsumexp(x, -1, { keepdim: true });
    expect(L.shape).toEqual([1, 1]);
    expect(L.value.get([0, 0])).toBeCloseTo(1000 + Math.log(3), 9);
    const y = g.input('y', [3], [1, 2, 3]);
    expect(g.logsumexp(y, 0).value.toNested()).toBeCloseTo(Math.log(Math.E + Math.E ** 2 + Math.E ** 3), 12);
    expect(g.logsumexp(y, [0], { keepdim: true }).shape).toEqual([1]);
  });
});

describe('softmax family', () => {
  it('softmax rows sum to one and log_softmax is its log', () => {
    const g = graph();
    const x = g.input('x', [2, 3], [[1, 2, 3], [1000, 1000, 1000]]);
    const p = g.softmax(x, -1);
    const lp = g.log_softmax(x, 1);
    close(p.value.toNested(), [
      [0.09003057317038046, 0.24472847105479764, 0.6652409557748219],
      [1 / 3, 1 / 3, 1 / 3],
    ]);
    close(lp.value.toNested(), (p.value.toNested() as number[][]).map((row) => row.map(Math.log)));
    const q = g.softmax(x, 0);
    close(q.value.toNested(), [[0, 0, 0], [1, 1, 1]]);
  });
});

describe('matmul and transpose', () => {
  it('multiplies 2d by 2d, 1d by 2d and 2d by 1d', () => {
    const g = graph();
    const A = g.input('A', [2, 3], [[1, 2, 3], [4, 5, 6]]);
    const B = g.input('B', [3, 2], [[1, 0], [0, 1], [1, 1]]);
    const v = g.input('v', [3], [1, 1, 1]);
    const w = g.input('w', [2], [1, 2]);
    expect(g.matmul(A, B).value.toNested()).toEqual([[4, 5], [10, 11]]);
    expect(g.matmul(v, B).value.toNested()).toEqual([2, 2]);
    expect(g.matmul(A, v).value.toNested()).toEqual([6, 15]);
    expect(g.matmul(w, A).value.toNested()).toEqual([9, 12, 15]);
    expect(() => g.matmul(A, A)).toThrow('matmul: cannot multiply (2, 3) by (2, 3)');
    expect(() => g.matmul(v, w)).toThrow('matmul: cannot multiply (3,) by (2,)');
    expect(() => g.matmul(g.input('s', [], 1), A)).toThrow('matmul: cannot multiply () by (2, 3)');
  });

  it('transposes any two dims', () => {
    const g = graph();
    const x = g.input('x', [2, 3, 1], [1, 2, 3, 4, 5, 6]);
    expect(g.transpose(x, 0, 1).value.toNested()).toEqual([[[1], [4]], [[2], [5]], [[3], [6]]]);
    expect(g.transpose(x, -1, 0).shape).toEqual([1, 3, 2]);
  });
});

describe('gather', () => {
  it('picks one element per index entry along dim', () => {
    const g = graph();
    const p = g.input('p', [3, 4], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    const t = g.label('t', [[0], [1], [3]]);
    const y = g.gather(p, -1, t);
    expect(y.shape).toEqual([3, 1]);
    expect(y.value.toNested()).toEqual([[1], [6], [12]]);
    g.loss(g.sum(y));
    expect(p.grad!.toNested()).toEqual([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]);
    expect(y.inputs).toEqual([p, t]);
  });

  it('gathers along dim 0 and rejects bad indices', () => {
    const g = graph();
    const p = g.input('p', [2, 2], [[1, 2], [3, 4]]);
    const rows = g.label('r', [[1, 0]]);
    expect(g.gather(p, 0, rows).value.toNested()).toEqual([[3, 2]]);
    expect(() => g.gather(p, 1, g.label('flat', [0]))).toThrow('gather: index (1,) must have the same number of dimensions as input (2, 2)');
    expect(() => g.gather(p, 1, g.label('big', [[0], [0], [0]]))).toThrow('gather: index (3, 1) is larger than input (2, 2) along dim 0');
    expect(() => g.gather(p, 1, g.label('oob', [[2], [0]]))).toThrow('gather: index 2 at [0, 0] is out of range for input (2, 2) along dim 1');
  });
});

describe('nll_loss', () => {
  const logp = [
    [-0.5, -1.5, -2.5],
    [-0.1, -2.0, -3.0],
    [-1.0, -1.0, -1.0],
  ];

  it('supports mean, sum and none reductions', () => {
    const g = graph();
    const x = g.input('x', [3, 3], logp);
    const t = g.label('t', [0, 1, 2]);
    const mean = g.nll_loss(x, t);
    const sum = g.nll_loss(x, t, { reduction: 'sum' });
    const none = g.nll_loss(x, t, { reduction: 'none' });
    expect(mean.shape).toEqual([]);
    expect(mean.value.toNested()).toBeCloseTo(3.5 / 3, 12);
    expect(sum.value.toNested()).toBeCloseTo(3.5, 12);
    expect(none.shape).toEqual([3]);
    expect(none.value.toNested()).toEqual([0.5, 2.0, 1.0]);
    expect(mean.attrs).toEqual({ reduction: 'mean', ignore_index: -100 });
    g.loss(mean);
    close(x.grad!.toNested(), [[-1 / 3, 0, 0], [0, -1 / 3, 0], [0, 0, -1 / 3]]);
    g.loss(sum);
    close(x.grad!.toNested(), [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]);
  });

  it('pairs a (C,) input with a () target', () => {
    const g = graph();
    const x = g.input('x', [3], logp[0]);
    const t = g.label('t', 1);
    const l = g.nll_loss(x, t);
    expect(l.value.toNested()).toBe(1.5);
    expect(g.nll_loss(x, t, { reduction: 'none' }).shape).toEqual([]);
    expect(x.grad!.toNested()).toEqual([0, -1, 0]);
    expect(() => g.nll_loss(g.input('m', [1, 3], logp[0]), t)).toThrow('nll_loss: input (1, 3) does not match target ()');
    expect(() => g.nll_loss(x, g.label('bad', 3))).toThrow('nll_loss: target 3 at row 0 is out of range for 3 classes');
    expect(() => g.nll_loss(x, t, { reduction: 'avg' as never })).toThrow("nll_loss: reduction must be one of 'mean', 'sum', 'none'");
  });

  it('skips ignore_index rows and divides the mean by the rows kept', () => {
    const g = graph();
    const x = g.input('x', [3, 3], logp);
    const t = g.label('t', [0, -100, 2]);
    const mean = g.nll_loss(x, t);
    expect(mean.value.toNested()).toBeCloseTo(1.5 / 2, 12);
    expect(g.nll_loss(x, t, { reduction: 'none' }).value.toNested()).toEqual([0.5, 0, 1.0]);
    g.loss(mean);
    close(x.grad!.toNested(), [[-0.5, 0, 0], [0, 0, 0], [0, 0, -0.5]]);
    const custom = g.nll_loss(x, g.label('u', [0, 1, 7]), { ignore_index: 7, reduction: 'sum' });
    expect(custom.value.toNested()).toBeCloseTo(2.5, 12);
  });

  it('returns NaN for mean when every row is ignored', () => {
    const g = graph();
    const x = g.input('x', [2, 3], logp.slice(0, 2));
    const t = g.label('t', [-100, -100]);
    const mean = g.nll_loss(x, t);
    expect(Number.isNaN(mean.value.get([]))).toBe(true);
    expect(g.nll_loss(x, t, { reduction: 'sum' }).value.toNested()).toBe(0);
    expect(g.nll_loss(x, t, { reduction: 'none' }).value.toNested()).toEqual([0, 0]);
    g.loss(mean);
    expect(x.grad!.toNested()).toEqual([[0, 0, 0], [0, 0, 0]]);
  });
});
