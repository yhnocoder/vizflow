import { Tensor, broadcastIndex, broadcastShapes, forEachIndex, formatShape, offsetOf } from '../../tensor';
import type { OpCost, OpDef } from '../op';
import { numel } from './cost';

const MULTIPLY_ADD = 2;

function contraction(dims: (inputs: number[][], out: number[]) => number[]): OpCost {
  return {
    saves: [[1], [0]],
    flops: (inputs, out) => {
      const count = [MULTIPLY_ADD, ...dims(inputs, out)];
      return { forward: count, backward: inputs.map(() => count) };
    },
  };
}

type NoAttrs = Record<string, never>;

function matmulShape(a: readonly number[], b: readonly number[]): number[] {
  const fail = () => new Error(`matmul: cannot multiply ${formatShape(a)} by ${formatShape(b)}`);
  if (isBatched(a, b)) {
    if (a.length < 2 || b.length < 2 || a[a.length - 1] !== b[b.length - 2]) throw fail();
    let batch: number[];
    try {
      batch = broadcastShapes(a.slice(0, -2), b.slice(0, -2));
    } catch {
      throw fail();
    }
    return [...batch, a[a.length - 2], b[b.length - 1]];
  }
  if (a.length === 2 && b.length === 2) {
    if (a[1] !== b[0]) throw fail();
    return [a[0], b[1]];
  }
  if (a.length === 1 && b.length === 2) {
    if (a[0] !== b[0]) throw fail();
    return [b[1]];
  }
  if (a.length === 2 && b.length === 1) {
    if (a[1] !== b[0]) throw fail();
    return [a[0]];
  }
  throw fail();
}

function as2d(t: Tensor, position: 'left' | 'right'): { rows: number; cols: number; data: Float64Array } {
  if (t.ndim === 2) return { rows: t.shape[0], cols: t.shape[1], data: t.data };
  return position === 'left' ? { rows: 1, cols: t.shape[0], data: t.data } : { rows: t.shape[0], cols: 1, data: t.data };
}

function multiply(a: Tensor, b: Tensor, shape: number[]): Tensor {
  const A = as2d(a, 'left');
  const B = as2d(b, 'right');
  const out = new Tensor(shape);
  for (let r = 0; r < A.rows; r++) {
    for (let i = 0; i < B.cols; i++) {
      let s = 0;
      for (let k = 0; k < A.cols; k++) s += A.data[r * A.cols + k] * B.data[k * B.cols + i];
      out.data[r * B.cols + i] = s;
    }
  }
  return out;
}

function transposed2d(t: Tensor, position: 'left' | 'right'): Tensor {
  const M = as2d(t, position);
  const out = new Tensor([M.cols, M.rows]);
  for (let r = 0; r < M.rows; r++) {
    for (let c = 0; c < M.cols; c++) out.data[c * M.rows + r] = M.data[r * M.cols + c];
  }
  return out;
}

function lifted(t: Tensor, position: 'left' | 'right'): Tensor {
  const M = as2d(t, position);
  return new Tensor([M.rows, M.cols], M.data);
}

export function isBatched(a: readonly number[], b: readonly number[]): boolean {
  return a.length > 2 || b.length > 2;
}

interface BatchPlan {
  batch: number[];
  rows: number;
  inner: number;
  cols: number;
  aOffset(index: number[]): number;
  bOffset(index: number[]): number;
}

function batchPlan(a: readonly number[], b: readonly number[]): BatchPlan {
  const out = matmulShape(a, b);
  const batch = out.slice(0, -2);
  const aBatch = a.slice(0, -2);
  const bBatch = b.slice(0, -2);
  const rows = a[a.length - 2];
  const inner = a[a.length - 1];
  const cols = b[b.length - 1];
  return {
    batch,
    rows,
    inner,
    cols,
    aOffset: (index) => offsetOf(aBatch, broadcastIndex(index, aBatch)) * rows * inner,
    bOffset: (index) => offsetOf(bBatch, broadcastIndex(index, bBatch)) * inner * cols,
  };
}

function batchedForward(a: Tensor, b: Tensor): Tensor {
  const plan = batchPlan(a.shape, b.shape);
  const { rows, inner, cols } = plan;
  const out = new Tensor([...plan.batch, rows, cols]);
  forEachIndex(plan.batch, (index, offset) => {
    const ao = plan.aOffset(index);
    const bo = plan.bOffset(index);
    const oo = offset * rows * cols;
    for (let r = 0; r < rows; r++) {
      for (let i = 0; i < cols; i++) {
        let s = 0;
        for (let k = 0; k < inner; k++) s += a.data[ao + r * inner + k] * b.data[bo + k * cols + i];
        out.data[oo + r * cols + i] = s;
      }
    }
  });
  return out;
}

function batchedBackward(gradOut: Tensor, a: Tensor, b: Tensor): Tensor[] {
  const plan = batchPlan(a.shape, b.shape);
  const { rows, inner, cols } = plan;
  const gA = new Tensor(a.shape);
  const gB = new Tensor(b.shape);
  forEachIndex(plan.batch, (index, offset) => {
    const ao = plan.aOffset(index);
    const bo = plan.bOffset(index);
    const oo = offset * rows * cols;
    for (let r = 0; r < rows; r++) {
      for (let i = 0; i < cols; i++) {
        const g = gradOut.data[oo + r * cols + i];
        for (let k = 0; k < inner; k++) {
          gA.data[ao + r * inner + k] += g * b.data[bo + k * cols + i];
          gB.data[bo + k * cols + i] += g * a.data[ao + r * inner + k];
        }
      }
    }
  });
  return [gA, gB];
}

export const matmulOp: OpDef<NoAttrs> = {
  name: 'matmul',
  args: ['input', 'other'],
  tensors: ['input', 'other'],
  shape: ([a, b]) => matmulShape(a, b),
  forward: ([a, b]) => (isBatched(a.shape, b.shape) ? batchedForward(a, b) : multiply(a, b, matmulShape(a.shape, b.shape))),
  backward: (gradOut, [a, b]) => {
    if (isBatched(a.shape, b.shape)) return batchedBackward(gradOut, a, b);
    const G = lifted(gradOut, a.ndim === 1 ? 'left' : 'right');
    const gA = multiply(G, transposed2d(b, 'right'), [G.shape[0], as2d(b, 'right').rows]);
    const gB = multiply(transposed2d(a, 'left'), G, [as2d(a, 'left').cols, G.shape[1]]);
    return [new Tensor(a.shape, gA.data), new Tensor(b.shape, gB.data)];
  },
  cost: contraction(([a], out) => [numel(out), a[a.length - 1]]),
};

export interface SelectMatmulAttrs {
  index: number;
}

function headIndex(a: readonly number[], index: number): number {
  return index < 0 && a.length === 3 ? index + a[1] : index;
}

function selectMatmulShape(a: readonly number[], b: readonly number[], requested: number): number[] {
  const index = headIndex(a, requested);
  if (a.length !== 3 || b.length !== 2 || a[2] !== b[0] || !Number.isInteger(index) || index < 0 || index >= a[1]) {
    throw new Error(`select_matmul: cannot multiply ${formatShape(a)}[:, ${requested}, :] by ${formatShape(b)}`);
  }
  return [a[0], b[1]];
}

function slice(a: Tensor, index: number): Tensor {
  const [rows, heads, width] = a.shape;
  const out = new Tensor([rows, width]);
  for (let r = 0; r < rows; r++) out.data.set(a.data.subarray((r * heads + index) * width, (r * heads + index + 1) * width), r * width);
  return out;
}

export const selectMatmulOp: OpDef<SelectMatmulAttrs> = {
  name: 'select_matmul',
  args: ['input', 'other', 'index'],
  tensors: ['input', 'other'],
  shape: ([a, b], { index }) => selectMatmulShape(a, b, index),
  forward: ([a, b], { index }) => multiply(slice(a, headIndex(a.shape, index)), b, selectMatmulShape(a.shape, b.shape, index)),
  backward: (gradOut, [a, b], _out, attrs) => {
    const index = headIndex(a.shape, attrs.index);
    const [rows, heads, width] = a.shape;
    const gSlice = multiply(gradOut, transposed2d(b, 'right'), [rows, width]);
    const gA = new Tensor(a.shape);
    for (let r = 0; r < rows; r++) gA.data.set(gSlice.data.subarray(r * width, (r + 1) * width), (r * heads + index) * width);
    const gB = multiply(transposed2d(slice(a, index), 'left'), gradOut, [width, b.shape[1]]);
    return [gA, gB];
  },
  cost: contraction(([a, b]) => [a[0], b[0], b[1]]),
};

function groupedLinearShape(input: readonly number[], weight: readonly number[], group: readonly number[]): number[] {
  if (input.length !== 2 || weight.length !== 3 || group.length !== 1 || weight[2] !== input[1] || group[0] !== input[0]) {
    throw new Error(`grouped_linear: input ${formatShape(input)} must be (N, in), weight ${formatShape(weight)} must be (G, out, in) and group ${formatShape(group)} must be (N,)`);
  }
  return [input[0], weight[1]];
}

export function groupOf(weight: Tensor, group: Tensor, row: number): number {
  const g = Math.round(group.data[row]);
  if (g < 0 || g >= weight.shape[0]) throw new Error(`grouped_linear: group ${g} of row ${row} is out of range for weight ${formatShape(weight.shape)}`);
  return g;
}

export const groupedLinearOp: OpDef<NoAttrs> = {
  name: 'grouped_linear',
  args: ['input', 'weight', 'group'],
  tensors: ['input', 'weight', 'group'],
  differentiable: [true, true, false],
  shape: ([input, weight, group]) => groupedLinearShape(input, weight, group),
  forward: ([input, weight, group]) => {
    const [rows, outs] = groupedLinearShape(input.shape, weight.shape, group.shape);
    const width = input.shape[1];
    const out = new Tensor([rows, outs]);
    for (let r = 0; r < rows; r++) {
      const base = groupOf(weight, group, r) * outs * width;
      for (let i = 0; i < outs; i++) {
        let s = 0;
        for (let k = 0; k < width; k++) s += input.data[r * width + k] * weight.data[base + i * width + k];
        out.data[r * outs + i] = s;
      }
    }
    return out;
  },
  backward: (gradOut, [input, weight, group]) => {
    const [rows, outs] = groupedLinearShape(input.shape, weight.shape, group.shape);
    const width = input.shape[1];
    const gi = new Tensor(input.shape);
    const gw = new Tensor(weight.shape);
    for (let r = 0; r < rows; r++) {
      const base = groupOf(weight, group, r) * outs * width;
      for (let i = 0; i < outs; i++) {
        const g = gradOut.data[r * outs + i];
        for (let k = 0; k < width; k++) {
          gi.data[r * width + k] += g * weight.data[base + i * width + k];
          gw.data[base + i * width + k] += g * input.data[r * width + k];
        }
      }
    }
    return [gi, gw, null];
  },
  cost: { ...contraction(([input, weight]) => [input[0], input[1], weight[1]]), saves: [[1, 2], [0, 2]] },
};

export interface SolveTriangularAttrs {
  upper: boolean | null;
  unitriangular: boolean;
}

function triangularShape(a: readonly number[], b: readonly number[], attrs: SolveTriangularAttrs): number[] {
  if (typeof attrs.upper !== 'boolean') throw new Error('solve_triangular: upper must be given as true or false');
  const n = a[a.length - 1];
  const sameBatch = a.length === b.length && a.slice(0, -2).every((size, d) => size === b[d]);
  if (a.length < 2 || a[a.length - 2] !== n || !sameBatch || b[b.length - 2] !== n) {
    throw new Error(`solve_triangular: A ${formatShape(a)} must be (..., n, n) and B ${formatShape(b)} must be (..., n, k) with the same batch dimensions`);
  }
  return [...b];
}

export function inTriangle(row: number, col: number, attrs: SolveTriangularAttrs): boolean {
  if (attrs.unitriangular && row === col) return false;
  return attrs.upper ? col >= row : col <= row;
}

export function solveTriangular(a: Tensor, b: Tensor, attrs: SolveTriangularAttrs): Tensor {
  triangularShape(a.shape, b.shape, attrs);
  const n = a.shape[a.ndim - 1];
  const k = b.shape[b.ndim - 1];
  const out = new Tensor(b.shape);
  const batches = a.size / (n * n);
  for (let m = 0; m < batches; m++) {
    const ao = m * n * n;
    const bo = m * n * k;
    for (let step = 0; step < n; step++) {
      const i = attrs.upper ? n - 1 - step : step;
      for (let c = 0; c < k; c++) {
        let s = b.data[bo + i * k + c];
        for (let j = 0; j < n; j++) {
          if (j !== i && inTriangle(i, j, attrs)) s -= a.data[ao + i * n + j] * out.data[bo + j * k + c];
        }
        out.data[bo + i * k + c] = attrs.unitriangular ? s : s / a.data[ao + i * n + i];
      }
    }
  }
  return out;
}

function lastTransposed(t: Tensor): Tensor {
  const rows = t.shape[t.ndim - 2];
  const cols = t.shape[t.ndim - 1];
  const out = new Tensor([...t.shape.slice(0, -2), cols, rows]);
  const batches = t.size / (rows * cols);
  for (let m = 0; m < batches; m++) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) out.data[m * rows * cols + c * rows + r] = t.data[m * rows * cols + r * cols + c];
    }
  }
  return out;
}

export const solveTriangularOp: OpDef<SolveTriangularAttrs> = {
  name: 'solve_triangular',
  args: ['A', 'B'],
  tensors: ['A', 'B'],
  defaults: { upper: null, unitriangular: false },
  shape: ([a, b], attrs) => triangularShape(a, b, attrs),
  forward: ([a, b], attrs) => solveTriangular(a, b, attrs),
  backward: (gradOut, [a], out, attrs) => {
    const gB = solveTriangular(lastTransposed(a), gradOut, { ...attrs, upper: !attrs.upper });
    const n = a.shape[a.ndim - 1];
    const k = out.shape[out.ndim - 1];
    const gA = new Tensor(a.shape);
    const batches = a.size / (n * n);
    for (let m = 0; m < batches; m++) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (!inTriangle(i, j, attrs)) continue;
          let s = 0;
          for (let c = 0; c < k; c++) s += gB.data[m * n * k + i * k + c] * out.data[m * n * k + j * k + c];
          gA.data[m * n * n + i * n + j] = -s;
        }
      }
    }
    return [gA, gB];
  },
  cost: {
    saves: [[0, 'out'], [0]],
    flops: ([a], out) => {
      const count = [numel(out), a[a.length - 1]];
      return { forward: count, backward: [count, count] };
    },
  },
};
