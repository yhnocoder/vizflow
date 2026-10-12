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
  { op: 'add', build: (g, r) => scalar(g, r, 'add', 0.5) },
  { op: 'sub', build: (g, r) => scalar(g, r, 'sub', -1.5) },
  { op: 'mul', build: (g, r) => scalar(g, r, 'mul', -0.7) },
  { op: 'div', build: (g, r) => scalar(g, r, 'div', 4) },
  {
    op: 'pow',
    build: (g, r) => {
      const x = g.input('x', [2, 3], randomValues(r, [2, 3]));
      return { out: g.pow(x, 3), inputs: [x] };
    },
  },
  {
    op: 'pow',
    build: (g, r) => {
      const x = g.input('x', [2, 3], randomValues(r, [2, 3], true));
      return { out: g.pow(x, -0.5), inputs: [x] };
    },
  },
  { op: 'rsqrt', build: (g, r) => unary(g, r, 'rsqrt', [2, 3], true) },
  {
    op: 'rsqrt',
    build: (g, r) => {
      const x = g.input('x', [2, 3], randomValues(r, [2, 3], true));
      return { out: g.rsqrt(x, { eps: 0.25 }), inputs: [x] };
    },
  },
  { op: 'sqrt', build: (g, r) => unary(g, r, 'sqrt', [2, 3], true) },
  { op: 'reciprocal', build: (g, r) => unary(g, r, 'reciprocal', [2, 3], true) },
  { op: 'erf', build: (g, r) => unary(g, r, 'erf', [2, 3]) },
  { op: 'sigmoid', build: (g, r) => unary(g, r, 'sigmoid', [2, 3]) },
  { op: 'silu', build: (g, r) => unary(g, r, 'silu', [2, 3]) },
  { op: 'relu', build: (g, r) => unary(g, r, 'relu', [2, 3]) },
  { op: 'gelu', build: (g, r) => unary(g, r, 'gelu', [2, 3]) },
  { op: 'identity', build: (g, r) => unary(g, r, 'identity', [2, 3]) },
  { op: 'rotate_half', build: (g, r) => unary(g, r, 'rotate_half', [3, 4]) },
  { op: 'rotate_half', build: (g, r) => unary(g, r, 'rotate_half', [6]) },
  { op: 'rotate_half', build: (g, r) => unary(g, r, 'rotate_half', [3, 2, 4]) },
  {
    op: 'clamp',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      return { out: g.clamp(x, -0.5, 0.7), inputs: [x] };
    },
  },
  {
    op: 'clamp',
    build: (g, r) => {
      const x = g.input('x', [2, 5], randomValues(r, [2, 5]));
      return { out: g.clamp(x, { max: 0.3 }), inputs: [x] };
    },
  },
  { op: 'mul', build: (g, r) => binary(g, r, 'mul', [3, 2, 4], [3, 1, 4]) },
  { op: 'mean', build: (g, r) => reduce(g, r, 'mean', [3, 2, 4], -1, true) },
  {
    op: 'view',
    build: (g, r) => {
      const x = g.input('x', [3, 8], randomValues(r, [3, 8]));
      return { out: g.view(x, 3, 2, 4), inputs: [x] };
    },
  },
  {
    op: 'select',
    build: (g, r) => {
      const x = g.input('x', [3, 2, 4], randomValues(r, [3, 2, 4]));
      return { out: g.select(x, 1, 1), inputs: [x] };
    },
  },
  {
    op: 'select_matmul',
    build: (g, r) => {
      const A = g.input('A', [3, 2, 4], randomValues(r, [3, 2, 4]));
      const B = g.input('B', [4, 3], randomValues(r, [4, 3]));
      return { out: g.select_matmul(A, B, 1), inputs: [A, B] };
    },
  },
  {
    op: 'embedding',
    build: (g, r) => {
      const ids = g.label('ids', [[0, 2], [2, 2]]);
      const W = g.input('W', [3, 4], randomValues(r, [3, 4]));
      return { out: g.embedding(ids, W), inputs: [W] };
    },
  },
  {
    op: 'narrow',
    build: (g, r) => {
      const x = g.input('x', [3, 5], randomValues(r, [3, 5]));
      return { out: g.narrow(x, 1, 1, 3), inputs: [x] };
    },
  },
  {
    op: 'narrow',
    build: (g, r) => {
      const x = g.input('x', [3, 5], randomValues(r, [3, 5]));
      return { out: g.narrow(x, -2, -2, 2), inputs: [x] };
    },
  },
  {
    op: 'cat',
    build: (g, r) => {
      const x = g.input('x', [2, 3], randomValues(r, [2, 3]));
      const y = g.input('y', [2, 2], randomValues(r, [2, 2]));
      return { out: g.cat([x, y], 1), inputs: [x, y] };
    },
  },
  {
    op: 'cat',
    build: (g, r) => {
      const x = g.input('x', [1, 2], randomValues(r, [1, 2]));
      const y = g.input('y', [2, 2], randomValues(r, [2, 2]));
      return { out: g.cat([x, y, x]), inputs: [x, y] };
    },
  },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [5], 3, {}) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [3, 5], [2, 0, 4], {}) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [3, 5], [2, 0, 4], { reduction: 'sum' }) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [3, 5], [2, 0, 4], { reduction: 'none' }) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [4, 5], [2, -100, 4, 1], {}) },
  { op: 'nll_loss', build: (g, r) => nll(g, r, [4, 5], [2, 7, 4, 1], { reduction: 'none', ignore_index: 7 }) },
  { op: 'softplus', build: (g, r) => unary(g, r, 'softplus', [2, 3]) },
  {
    op: 'repeat_interleave',
    build: (g, r) => {
      const x = g.input('x', [3, 2, 2], randomValues(r, [3, 2, 2]));
      return { out: g.repeat_interleave(x, 3, 1), inputs: [x] };
    },
  },
  { op: 'logical_not', build: (g, r) => unary(g, r, 'logical_not', [2, 3]) },
  {
    op: 'scatter',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      const i = g.label('i', [[2, 0], [1, 1], [3, 2]]);
      return { out: g.scatter(x, 1, i, 1), inputs: [x] };
    },
  },
  {
    op: 'expand',
    build: (g, r) => {
      const x = g.input('x', [2, 1, 3], randomValues(r, [2, 1, 3]));
      return { out: g.expand(x, 2, 2, 4, -1), inputs: [x] };
    },
  },
  {
    op: 'reshape',
    build: (g, r) => {
      const x = g.input('x', [2, 3, 2], randomValues(r, [2, 3, 2]));
      return { out: g.reshape(x, -1, 4), inputs: [x] };
    },
  },
  {
    op: 'causal_conv1d',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      const w = g.input('w', [4, 4], randomValues(r, [4, 4]));
      return { out: g.causal_conv1d(x, w), inputs: [x, w] };
    },
  },
  {
    op: 'causal_conv1d',
    build: (g, r) => {
      const x = g.input('x', [5, 2], randomValues(r, [5, 2]));
      const w = g.input('w', [2, 3], randomValues(r, [2, 3]));
      return { out: g.causal_conv1d(x, w), inputs: [x, w] };
    },
  },
  {
    op: 'topk_mask',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      return { out: g.topk_mask(x, 2, -1), inputs: [x] };
    },
  },
  {
    op: 'where',
    build: (g, r) => {
      const keep = g.label('keep', [[1], [0], [1]]);
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      return { out: g.where(keep, x, -1e9), inputs: [x] };
    },
  },
  {
    op: 'greater',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      return { out: g.greater(x, 0), inputs: [x] };
    },
  },
  {
    op: 'topk_indices',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      return { out: g.topk_indices(x, 2, -1), inputs: [x] };
    },
  },
  {
    op: 'one_hot',
    build: (g) => {
      const t = g.label('t', [[0, 2], [1, 0]]);
      return { out: g.one_hot(t, 3), inputs: [] };
    },
  },
  {
    op: 'nonzero_static',
    build: (g) => {
      const m = g.label('m', [[0, 1, 1], [1, 0, 0]]);
      return { out: g.nonzero_static(m, { size: 4 }), inputs: [] };
    },
  },
  {
    op: 'permute',
    build: (g, r) => {
      const x = g.input('x', [2, 3, 4], randomValues(r, [2, 3, 4]));
      return { out: g.permute(x, 2, 0, 1), inputs: [x] };
    },
  },
  {
    op: 'index',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      const rows = g.label('rows', [2, 0, 2, 1, 2]);
      return { out: g.index(x, [rows]), inputs: [x] };
    },
  },
  {
    op: 'index',
    build: (g, r) => {
      const x = g.input('x', [3, 2], randomValues(r, [3, 2]));
      const rows = g.label('rows', [2, 0, 2, 1]);
      const cols = g.label('cols', [1, 1, 1, 0]);
      return { out: g.index(x, [rows, cols]), inputs: [x] };
    },
  },
  {
    op: 'index_add',
    build: (g, r) => {
      const x = g.input('x', [3, 2], randomValues(r, [3, 2]));
      const index = g.label('index', [2, 0, 2, 2]);
      const source = g.input('s', [4, 2], randomValues(r, [4, 2]));
      return { out: g.index_add(x, 0, index, source), inputs: [x, source] };
    },
  },
  {
    op: 'grouped_linear',
    build: (g, r) => {
      const x = g.input('x', [5, 3], randomValues(r, [5, 3]));
      const w = g.input('w', [4, 2, 3], randomValues(r, [4, 2, 3]));
      const group = g.label('group', [0, 0, 1, 3, 3]);
      return { out: g.grouped_linear(x, w, group), inputs: [x, w] };
      },
    },
  { op: 'matmul', build: (g, r) => matmul(g, r, [2, 3, 4], [2, 4, 3]) },
  { op: 'matmul', build: (g, r) => matmul(g, r, [2, 3, 4], [4, 2]) },
  { op: 'matmul', build: (g, r) => matmul(g, r, [3, 4], [2, 1, 4, 2]) },
  {
    op: 'cumsum',
    build: (g, r) => {
      const x = g.input('x', [2, 3, 2], randomValues(r, [2, 3, 2]));
      return { out: g.cumsum(x, 1), inputs: [x] };
    },
  },
  {
    op: 'cumsum',
    build: (g, r) => {
      const x = g.input('x', [2, 4], randomValues(r, [2, 4]));
      return { out: g.cumsum(x, -1), inputs: [x] };
    },
  },
  {
    op: 'unsqueeze',
    build: (g, r) => {
      const x = g.input('x', [2, 3], randomValues(r, [2, 3]));
      return { out: g.unsqueeze(x, -1), inputs: [x] };
    },
  },
  {
    op: 'pad',
    build: (g, r) => {
      const x = g.input('x', [2, 3, 2], randomValues(r, [2, 3, 2]));
      return { out: g.pad(x, 0, 0, 1, 2), inputs: [x] };
    },
  },
  {
    op: 'triu',
    build: (g, r) => {
      const x = g.input('x', [2, 3, 3], randomValues(r, [2, 3, 3]));
      return { out: g.triu(x, 1), inputs: [x] };
    },
  },
  {
    op: 'masked_fill',
    build: (g, r) => {
      const mask = g.label('mask', [[0, 1, 1], [0, 0, 1], [0, 0, 0]]);
      const x = g.input('x', [2, 3, 3], randomValues(r, [2, 3, 3]));
      return { out: g.masked_fill(x, mask, -1e9), inputs: [x] };
    },
  },
  {
    op: 'solve_triangular',
    build: (g, r) => {
      const A = g.input('A', [2, 3, 3], randomValues(r, [2, 3, 3]));
      const B = g.input('B', [2, 3, 2], randomValues(r, [2, 3, 2]));
      return { out: g.solve_triangular(A, B, { upper: false, unitriangular: true }), inputs: [A, B] };
    },
  },
  {
    op: 'solve_triangular',
    build: (g, r) => {
      const A = g.input('A', [3, 3], randomValues(r, [3, 3], true));
      const B = g.input('B', [3, 2], randomValues(r, [3, 2]));
      return { out: g.solve_triangular(A, B, { upper: true }), inputs: [A, B] };
      },
    },
  { op: 'remainder', build: (g, r) => binary(g, r, 'remainder', [2, 3], [3], true) },
  { op: 'remainder', build: (g, r) => scalar(g, r, 'remainder', 0.7) },
  ...(['bitwise_xor', 'bitwise_and', 'bitwise_or', 'floor_divide', 'eq', 'ge', 'lt'] as const).map((op) => ({
    op,
    build: (g: Graph) => {
      const a = g.label('a', [[3, 1, 2], [0, 5, 4]]);
      const b = g.label('b', [2, 1, 6]);
      return { out: (g as unknown as Record<string, (...args: unknown[]) => Node>)[op](a, b), inputs: [] };
    },
  })),
  { op: 'abs', build: (g, r) => unary(g, r, 'abs', [2, 3]) },
  { op: 'sign', build: (g, r) => unary(g, r, 'sign', [2, 3]) },
  { op: 'clamp_min', build: (g, r) => scalar(g, r, 'clamp_min', 0.3) },
  {
    op: 'repeat',
    build: (g, r) => {
      const x = g.input('x', [2, 3], randomValues(r, [2, 3]));
      return { out: g.repeat(x, 2, 1, 2), inputs: [x] };
    },
  },
  {
    op: 'cummax',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      return { out: g.cummax(x, 1), inputs: [x] };
    },
  },
  {
    op: 'causal_conv1d',
    build: (g, r) => {
      const x = g.input('x', [7, 2], randomValues(r, [7, 2]));
      const w = g.input('w', [2, 3], randomValues(r, [2, 3]));
      return { out: g.causal_conv1d(x, w, { dilation: 2 }), inputs: [x, w] };
    },
  },
  {
    op: 'logical_not',
    build: (g) => {
      const m = g.label('m', [[0, 1, 1], [1, 0, 0]]);
      return { out: g.logical_not(m), inputs: [] };
    },
  },
  { op: 'copysign', build: (g, r) => binary(g, r, 'copysign', [2, 3], [3]) },
  { op: 'copysign', build: (g, r) => scalar(g, r, 'copysign', -1) },
  {
    op: 'where',
    build: (g, r) => {
      const keep = g.label('keep', [[1], [0], [1]]);
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      const y = g.input('y', [4], randomValues(r, [4]));
      return { out: g.where(keep, x, y), inputs: [x, y] };
    },
  },
  {
    op: 'sort',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      return { out: g.sort(x, -1), inputs: [x] };
    },
  },
  {
    op: 'scatter',
    build: (g, r) => {
      const x = g.input('x', [3, 4], randomValues(r, [3, 4]));
      const index = g.label('index', [[3, 0], [1, 2], [0, 3]]);
      const src = g.input('s', [3, 2], randomValues(r, [3, 2]));
      return { out: g.scatter(x, -1, index, src), inputs: [x, src] };
    },
  },
  {
    op: 'index_put',
    build: (g, r) => {
      const x = g.input('x', [4, 3], randomValues(r, [4, 3]));
      const mask = g.label('mask', [1, 0, 1, 0]);
      const v = g.input('v', [3], randomValues(r, [3]));
      return { out: g.index_put(x, mask, v), inputs: [x, v] };
    },
  },
  {
    op: 'index_put',
    build: (g, r) => {
      const x = g.input('x', [4, 3], randomValues(r, [4, 3]));
      const mask = g.label('mask', [0, 1, 1, 0]);
      const v = g.input('v', [2, 3], randomValues(r, [2, 3]));
      return { out: g.index_put(x, mask, v), inputs: [x, v] };
    },
  },
  {
    op: 'index',
    build: (g, r) => {
      const x = g.input('x', [4, 3], randomValues(r, [4, 3]));
      const rows = g.label('rows', [[2, 0], [3, 3], [1, 0]]);
      return { out: g.index(x, [rows]), inputs: [x] };
    },
  },
  {
    op: 'pad',
    build: (g, r) => {
      const x = g.input('x', [2, 3], randomValues(r, [2, 3]));
      return { out: g.pad(x, [0, 1], { value: -2 }), inputs: [x] };
    },
  },
];

function binary(g: Graph, r: () => number, op: string, a: number[], b: number[], positive = false) {
  const x = g.input('x', a, randomValues(r, a, positive));
  const y = g.input('y', b, randomValues(r, b, positive));
  return { out: (g as unknown as Record<string, (...args: unknown[]) => Node>)[op](x, y), inputs: [x, y] };
}

function scalar(g: Graph, r: () => number, op: string, other: number) {
  const x = g.input('x', [2, 3], randomValues(r, [2, 3]));
  return { out: (g as unknown as Record<string, (...args: unknown[]) => Node>)[op](x, other), inputs: [x] };
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
