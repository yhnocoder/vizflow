import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';

const X = [
  [2.0, 1.0, 0.2, -0.6, -1.4],
  [-0.5, 1.8, 0.3, 0.9, -1.0],
  [0.4, -0.2, 1.5, 0.1, 0.6],
];
const T = [0, 1, 2];

function softmaxRow(x: number[]): number[] {
  const u = x.map(Math.exp);
  const s = u.reduce((a, b) => a + b, 0);
  return u.map((v) => v / s);
}

function closeTo(actual: unknown, expected: unknown, digits = 10): void {
  const flat = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v as number]);
  const a = flat(actual);
  const e = flat(expected);
  expect(a.length).toBe(e.length);
  a.forEach((v, i) => expect(v).toBeCloseTo(e[i], digits));
}

describe('softmax with cross entropy, single sample', () => {
  it('computes p, the loss and dl/dx = p - e_t', () => {
    const g = graph({ check: true });
    const x = g.input('x', [5], X[0]);
    const t = g.label('t', [0]);
    const u = g.exp(x, { name: 'u' });
    const S = g.sum(u, -1, { keepdim: true, name: 'S' });
    const p = g.div(u, S, { name: 'p' });
    const y = g.gather(p, -1, t, { name: 'y' });
    const l = g.neg(g.log(y), { name: 'l' });
    const pRef = softmaxRow(X[0]);
    expect(S.shape).toEqual([1]);
    expect(p.shape).toEqual([5]);
    expect(y.shape).toEqual([1]);
    expect(l.shape).toEqual([1]);
    expect(g.loss()).toBe(l);
    closeTo(p.value.toNested(), pRef);
    expect(l.value.get([0])).toBeCloseTo(-Math.log(pRef[0]), 12);
    closeTo(x.grad!.toNested(), pRef.map((v, i) => v - (i === 0 ? 1 : 0)));
    t.set([3]);
    expect(l.value.get([0])).toBeCloseTo(-Math.log(pRef[3]), 12);
    closeTo(x.grad!.toNested(), pRef.map((v, i) => v - (i === 3 ? 1 : 0)));
  });
});

describe('softmax with cross entropy, batch', () => {
  it('computes dl/dX = (P - Y) / B', () => {
    const g = graph({ dims: { B: 3, C: 5 }, check: true });
    const x = g.input('X', ['B', 'C'], X);
    const t = g.label('t', T.map((v) => [v]));
    const U = g.exp(x, { name: 'U' });
    const S = g.sum(U, -1, { keepdim: true, name: 'S' });
    const P = g.div(U, S, { name: 'P' });
    const Y = g.gather(P, -1, t, { name: 'Y' });
    const lb = g.neg(g.log(Y), { name: 'l_b' });
    const l = g.mean(lb, { name: 'l' });
    const pRef = X.map(softmaxRow);
    expect(t.shape).toEqual([3, 1]);
    expect(S.shape).toEqual([3, 1]);
    expect(Y.shape).toEqual([3, 1]);
    expect(lb.shape).toEqual([3, 1]);
    expect(l.shape).toEqual([]);
    closeTo(P.value.toNested(), pRef);
    closeTo(lb.value.toNested(), T.map((tb, b) => [-Math.log(pRef[b][tb])]));
    expect(l.value.get([])).toBeCloseTo(T.reduce((s, tb, b) => s - Math.log(pRef[b][tb]), 0) / 3, 12);
    closeTo(x.grad!.toNested(), pRef.map((row, b) => row.map((v, i) => (v - (i === T[b] ? 1 : 0)) / 3)));
    closeTo(lb.grad!.toNested(), [[1 / 3], [1 / 3], [1 / 3]]);
    expect(t.index(1, 0)).toBe(1);
  });
});

describe('jacobian', () => {
  it('gives diag(p) - p p^T for the softmax probabilities', () => {
    const g = graph();
    const x = g.input('x', [5], X[0]);
    const u = g.exp(x, { name: 'u' });
    const p = g.div(u, g.sum(u, -1, { keepdim: true }), { name: 'p' });
    const pRef = softmaxRow(X[0]);
    const J = g.jacobian(p, x);
    expect(J.shape).toEqual([5, 5]);
    closeTo(J.toNested(), pRef.map((pi, i) => pRef.map((pj, j) => (i === j ? pi : 0) - pi * pj)));
  });

  it('follows the current input values and leaves the loss gradient unchanged', () => {
    const g = graph();
    const x = g.input('x', [5], X[0]);
    const t = g.label('t', [1]);
    const p = g.softmax(x, -1, { name: 'p' });
    g.neg(g.log(g.gather(p, -1, t)), { name: 'l' });
    const before = x.grad!.toNested();
    x.set(X[1]);
    const pRef = softmaxRow(X[1]);
    closeTo(g.jacobian(p, x).toNested(), pRef.map((pi, i) => pRef.map((pj, j) => (i === j ? pi : 0) - pi * pj)));
    closeTo(x.grad!.toNested(), pRef.map((v, i) => v - (i === 1 ? 1 : 0)));
    expect(before).not.toEqual(x.grad!.toNested());
  });
});

describe('jacobian of one batch row', () => {
  it('returns the (V, V) block of the selected sample', () => {
    const g = graph({ dims: { B: 3, C: 5 } });
    const x = g.input('X', ['B', 'C'], X);
    const p = g.softmax(x, -1, { name: 'P' });
    X.forEach((row, b) => {
      const pRef = softmaxRow(row);
      const J = g.jacobian(p, x, { row: b });
      expect(J.shape).toEqual([5, 5]);
      closeTo(J.toNested(), pRef.map((pi, i) => pRef.map((pj, j) => (i === j ? pi : 0) - pi * pj)));
    });
    expect(g.jacobian(p, x).shape).toEqual([3, 5, 3, 5]);
    expect(() => g.jacobian(p, x, { row: 3 })).toThrow('row 3 is out of range');
  });
});
