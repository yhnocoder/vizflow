import { describe, expect, it } from 'vitest';
import { type Graph, type Node, builtinOps, graph } from '../../src/core';
import { explain, explainOf } from '../../src/explain';
import { forEachIndex, numel } from '../../src/tensor';

const ABS = 1e-9;
const REL = 1e-9;

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomValues(random: () => number, shape: number[], positive = false): number[] {
  return Array.from({ length: numel(shape) }, () => (positive ? 0.5 + random() * 2 : random() * 4 - 2));
}

interface Case {
  op: string;
  build(g: Graph, random: () => number): { out: Node; inputs: Node[] };
}

const cases: Case[] = [
  { op: 'add', build: (g, r) => binary(g, r, 'add', [2, 3], [2, 3]) },
  { op: 'sub', build: (g, r) => binary(g, r, 'sub', [1, 5], [1, 1]) },
  { op: 'sub', build: (g, r) => binary(g, r, 'sub', [3], [2, 3]) },
  { op: 'mul', build: (g, r) => binary(g, r, 'mul', [2, 3], [3]) },
  { op: 'div', build: (g, r) => binary(g, r, 'div', [3, 5], [3, 1], true) },
  { op: 'neg', build: (g, r) => unary(g, r, 'neg', [2, 3]) },
  { op: 'exp', build: (g, r) => unary(g, r, 'exp', [2, 3]) },
  { op: 'log', build: (g, r) => unary(g, r, 'log', [2, 3], true) },
  {
    op: 'transpose',
    build: (g, r) => {
      const x = g.input('x', [2, 3], randomValues(r, [2, 3]));
      return { out: g.transpose(x, 0, 1), inputs: [x] };
    },
  },
  {
    op: 'transpose',
    build: (g, r) => {
      const x = g.input('x', [2, 3, 4], randomValues(r, [2, 3, 4]));
      return { out: g.transpose(x, -1, 0), inputs: [x] };
    },
  },
  { op: 'matmul', build: (g, r) => matmul(g, r, [1, 4], [4, 5]) },
  { op: 'matmul', build: (g, r) => matmul(g, r, [3, 4], [4, 2]) },
  { op: 'matmul', build: (g, r) => matmul(g, r, [4], [4, 2]) },
  { op: 'matmul', build: (g, r) => matmul(g, r, [3, 4], [4]) },
  { op: 'sum', build: (g, r) => reduce(g, r, 'sum', [2, 3], undefined, false) },
  { op: 'sum', build: (g, r) => reduce(g, r, 'sum', [2, 3], -1, true) },
  { op: 'sum', build: (g, r) => reduce(g, r, 'sum', [2, 3, 4], [0, 2], false) },
  { op: 'mean', build: (g, r) => reduce(g, r, 'mean', [2, 3], undefined, false) },
  { op: 'mean', build: (g, r) => reduce(g, r, 'mean', [3, 1], 0, true) },
  { op: 'max', build: (g, r) => reduce(g, r, 'max', [2, 3], 1, false) },
  { op: 'max', build: (g, r) => reduce(g, r, 'max', [2, 3], 0, true) },
  { op: 'logsumexp', build: (g, r) => reduce(g, r, 'logsumexp', [1, 5], -1, true) },
  { op: 'logsumexp', build: (g, r) => reduce(g, r, 'logsumexp', [3, 4], [0, 1], false) },
  { op: 'softmax', build: (g, r) => unaryDim(g, r, 'softmax', [3, 5], -1) },
  { op: 'softmax', build: (g, r) => unaryDim(g, r, 'softmax', [3, 5], 0) },
  { op: 'log_softmax', build: (g, r) => unaryDim(g, r, 'log_softmax', [3, 5], 1) },
  { op: 'log_softmax', build: (g, r) => unaryDim(g, r, 'log_softmax', [5], 0) },
  {
    op: 'gather',
    build: (g, r) => {
      const x = g.input('x', [3, 5], randomValues(r, [3, 5]));
      const t = g.label('t', [[1], [4], [1]]);
      return { out: g.gather(x, -1, t), inputs: [x] };
    },
  },
  {
    op: 'gather',
    build: (g, r) => {
      const x = g.input('x', [3, 5], randomValues(r, [3, 5]));
      const t = g.label('t', [[0, 2, 2, 1, 0]]);
      return { out: g.gather(x, 0, t), inputs: [x] };
    },
  },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [5], 3, {}) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [3, 5], [2, 0, 4], {}) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [3, 5], [2, 0, 4], { reduction: 'sum' }) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [3, 5], [2, 0, 4], { reduction: 'none' }) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [4, 5], [2, -100, 4, 1], {}) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [4, 5], [2, 7, 4, 1], { reduction: 'none', ignore_index: 7 }) },
];

function binary(g: Graph, r: () => number, op: string, a: number[], b: number[], positive = false) {
  const x = g.input('x', a, randomValues(r, a, positive));
  const y = g.input('y', b, randomValues(r, b, positive));
  return { out: (g as unknown as Record<string, (...args: unknown[]) => Node>)[op](x, y), inputs: [x, y] };
}

function unary(g: Graph, r: () => number, op: string, shape: number[], positive = false) {
  const x = g.input('x', shape, randomValues(r, shape, positive));
  return { out: (g as unknown as Record<string, (...args: unknown[]) => Node>)[op](x), inputs: [x] };
}

function unaryDim(g: Graph, r: () => number, op: string, shape: number[], dim: number) {
  const x = g.input('x', shape, randomValues(r, shape));
  return { out: (g as unknown as Record<string, (...args: unknown[]) => Node>)[op](x, dim), inputs: [x] };
}

function matmul(g: Graph, r: () => number, a: number[], b: number[]) {
  const A = g.input('A', a, randomValues(r, a));
  const B = g.input('B', b, randomValues(r, b));
  return { out: g.matmul(A, B), inputs: [A, B] };
}

function reduce(g: Graph, r: () => number, op: string, shape: number[], dim: number | number[] | undefined, keepdim: boolean) {
  const x = g.input('x', shape, randomValues(r, shape));
  const fn = (g as unknown as Record<string, (...args: unknown[]) => Node>)[op];
  return { out: dim === undefined ? fn(x, { keepdim }) : fn(x, dim, { keepdim }), inputs: [x] };
}

function nll(g: Graph, r: () => number, shape: number[], target: number | number[], options: object) {
  const x = g.input('x', shape, randomValues(r, shape));
  const t = g.label('t', target);
  return { out: g.nll_loss(x, t, options), inputs: [x] };
}

describe('edge sums reproduce backward', () => {
  it.each(cases.map((c, i) => [c.op, i] as const))('%s (case %i)', (_op, i) => {
    const random = seeded(1234 + i);
    const g = graph();
    const { out, inputs } = cases[i].build(g, random);
    const u = g.input('u', [...out.shape], randomValues(random, [...out.shape]));
    g.loss(g.sum(g.mul(out, u)));
    const ex = explain(g);
    for (const input of inputs) {
      const grad = input.grad!;
      forEachIndex(input.shape, (index) => {
        const edges = ex.outEdges({ node: input, index });
        const summed = edges.reduce((s, e) => s + ex.contribution(e), 0);
        const expected = grad.get(index);
        expect(Math.abs(summed - expected)).toBeLessThanOrEqual(ABS + REL * Math.abs(expected));
      });
    }
  });

  it('covers every builtin op that has an explain definition', () => {
    const covered = new Set(cases.map((c) => c.op));
    for (const def of builtinOps) {
      expect(explainOf(def.name)).toBeDefined();
      expect(covered.has(def.name), `op '${def.name}' has no edge-sum case`).toBe(true);
    }
  });
});
