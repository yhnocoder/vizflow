import { Tensor, formatShape } from '../../tensor';
import type { OpDef } from '../op';

type NoAttrs = Record<string, never>;

function matmulShape(a: readonly number[], b: readonly number[]): number[] {
  const fail = () => new Error(`matmul: cannot multiply ${formatShape(a)} by ${formatShape(b)}`);
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

export const matmulOp: OpDef<NoAttrs> = {
  name: 'matmul',
  args: ['input', 'other'],
  tensors: ['input', 'other'],
  shape: ([a, b]) => matmulShape(a, b),
  forward: ([a, b]) => multiply(a, b, matmulShape(a.shape, b.shape)),
  backward: (gradOut, [a, b]) => {
    const G = lifted(gradOut, a.ndim === 1 ? 'left' : 'right');
    const gA = multiply(G, transposed2d(b, 'right'), [G.shape[0], as2d(b, 'right').rows]);
    const gB = multiply(transposed2d(a, 'left'), G, [as2d(a, 'left').cols, G.shape[1]]);
    return [new Tensor(a.shape, gA.data), new Tensor(b.shape, gB.data)];
  },
};

export interface NarrowMatmulAttrs {
  start: number;
}

function narrowMatmulShape(a: readonly number[], b: readonly number[], start: number): number[] {
  if (a.length !== 2 || b.length !== 2 || !Number.isInteger(start) || start < 0 || start + b[0] > a[1]) {
    throw new Error(`narrow_matmul: cannot multiply columns ${start}..${start + (b[0] ?? 0) - 1} of ${formatShape(a)} by ${formatShape(b)}`);
  }
  return [a[0], b[1]];
}

function columns(a: Tensor, start: number, width: number): Tensor {
  const rows = a.shape[0];
  const out = new Tensor([rows, width]);
  for (let r = 0; r < rows; r++) out.data.set(a.data.subarray(r * a.shape[1] + start, r * a.shape[1] + start + width), r * width);
  return out;
}

export const narrowMatmulOp: OpDef<NarrowMatmulAttrs> = {
  name: 'narrow_matmul',
  args: ['input', 'other', 'start'],
  tensors: ['input', 'other'],
  shape: ([a, b], { start }) => narrowMatmulShape(a, b, start),
  forward: ([a, b], { start }) => multiply(columns(a, start, b.shape[0]), b, narrowMatmulShape(a.shape, b.shape, start)),
  backward: (gradOut, [a, b], _out, { start }) => {
    const width = b.shape[0];
    const gSlice = multiply(gradOut, transposed2d(b, 'right'), [a.shape[0], width]);
    const gA = new Tensor(a.shape);
    for (let r = 0; r < a.shape[0]; r++) gA.data.set(gSlice.data.subarray(r * width, (r + 1) * width), r * a.shape[1] + start);
    const gB = multiply(transposed2d(columns(a, start, width), 'left'), gradOut, [width, b.shape[1]]);
    return [gA, gB];
  },
};
