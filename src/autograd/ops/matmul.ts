import { Tensor, formatShape } from '../../tensor';
import type { OpDef } from '../op';
import { exponentOf } from '../rounding';

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
};

function groupedShape(x: readonly number[], w: readonly number[], group: readonly number[]): number[] {
  if (x.length !== 2 || w.length !== 3 || group.length !== 1 || w[2] !== x[1] || group[0] !== x[0]) {
    throw new Error(`grouped_mm: x ${formatShape(x)} must be (R, d), w ${formatShape(w)} must be (E, n, d) and group ${formatShape(group)} must be (R)`);
  }
  return [x[0], w[1]];
}

export function groupOf(group: Tensor, experts: number, row: number): number {
  const e = Math.round(group.data[row]);
  if (e < 0 || e >= experts) throw new Error(`grouped_mm: group ${e} at [${row}] is out of range for ${experts} experts`);
  return e;
}

export const groupedMmOp: OpDef<NoAttrs> = {
  name: 'grouped_mm',
  args: ['x', 'w', 'group'],
  tensors: ['x', 'w', 'group'],
  differentiable: [true, true, false],
  shape: ([x, w, group]) => groupedShape(x, w, group),
  forward: ([x, w, group]) => {
    const [R, n] = groupedShape(x.shape, w.shape, group.shape);
    const d = x.shape[1];
    const out = new Tensor([R, n]);
    for (let r = 0; r < R; r++) {
      const base = groupOf(group, w.shape[0], r) * n * d;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let k = 0; k < d; k++) s += w.data[base + i * d + k] * x.data[r * d + k];
        out.data[r * n + i] = s;
      }
    }
    return out;
  },
  backward: (gradOut, [x, w, group]) => {
    const [R, n] = gradOut.shape;
    const d = x.shape[1];
    const gx = new Tensor(x.shape);
    const gw = new Tensor(w.shape);
    for (let r = 0; r < R; r++) {
      const base = groupOf(group, w.shape[0], r) * n * d;
      for (let i = 0; i < n; i++) {
        const g = gradOut.data[r * n + i];
        for (let k = 0; k < d; k++) {
          gx.data[r * d + k] += g * w.data[base + i * d + k];
          gw.data[base + i * d + k] += g * x.data[r * d + k];
        }
      }
    }
    return [gx, gw, null];
  },
};

export const TC_ACC_BITS = 14;

export type ScaledMmAccum = 'fp32' | 'tc' | 'no_promote';

const ACCUMS: readonly ScaledMmAccum[] = ['fp32', 'tc', 'no_promote'];

export interface ScaledMmAttrs {
  block_k: number;
  block_n: number;
  accum: ScaledMmAccum;
}

function scaledMmShape(a: readonly number[], sfa: readonly number[], b: readonly number[], sfb: readonly number[], attrs: ScaledMmAttrs): number[] {
  if (!ACCUMS.includes(attrs.accum)) throw new Error(`scaled_mm: accum must be one of 'fp32', 'tc', 'no_promote', got ${JSON.stringify(attrs.accum)}`);
  for (const key of ['block_k', 'block_n'] as const) {
    if (!Number.isInteger(attrs[key]) || attrs[key] < 1) throw new Error(`scaled_mm: ${key} ${attrs[key]} must be a positive integer`);
  }
  if (a.length !== 2 || b.length !== 2 || a[1] !== b[1]) throw new Error(`scaled_mm: a ${formatShape(a)} must be (M, K) and b ${formatShape(b)} must be (N, K)`);
  const [M, K] = a;
  const N = b[0];
  const kBlocks = Math.ceil(K / attrs.block_k);
  const sfaShape = [M, kBlocks];
  const sfbShape = [Math.ceil(N / attrs.block_n), kBlocks];
  if (formatShape(sfa) !== formatShape(sfaShape)) {
    throw new Error(`scaled_mm: sfa ${formatShape(sfa)} must be (M, ceil(K / block_k)) = ${formatShape(sfaShape)} for a ${formatShape(a)} and block_k ${attrs.block_k}`);
  }
  if (formatShape(sfb) !== formatShape(sfbShape)) {
    throw new Error(
      `scaled_mm: sfb ${formatShape(sfb)} must be (ceil(N / block_n), ceil(K / block_k)) = ${formatShape(sfbShape)} for b ${formatShape(b)}, block_n ${attrs.block_n} and block_k ${attrs.block_k}`,
    );
  }
  return [M, N];
}

export function blockRange(K: number, j: number, blockK: number): [number, number] {
  return [j * blockK, Math.min(K, (j + 1) * blockK)];
}

export function blockDot(a: Tensor, b: Tensor, m: number, n: number, j: number, blockK: number): number {
  const K = a.shape[1];
  const [start, end] = blockRange(K, j, blockK);
  let s = 0;
  for (let k = start; k < end; k++) s += a.data[m * K + k] * b.data[n * K + k];
  return s;
}

export function tcAccumulate(acc: number, terms: readonly number[]): number {
  let top = acc === 0 ? -Infinity : exponentOf(Math.abs(acc));
  for (const t of terms) if (t !== 0) top = Math.max(top, exponentOf(Math.abs(t)));
  if (top === -Infinity) return acc;
  const quantum = 2 ** (top - TC_ACC_BITS + 1);
  let s = acc;
  for (const t of terms) s += Math.trunc(t / quantum) * quantum;
  return Math.fround(s);
}

export const scaledMmOp: OpDef<ScaledMmAttrs> = {
  name: 'scaled_mm',
  args: ['a', 'sfa', 'b', 'sfb', 'block_k', 'block_n', 'accum'],
  tensors: ['a', 'sfa', 'b', 'sfb'],
  sizes: ['block_k', 'block_n'],
  defaults: { block_k: 128, block_n: 128, accum: 'fp32' },
  shape: ([a, sfa, b, sfb], attrs) => scaledMmShape(a, sfa, b, sfb, attrs),
  forward: ([a, sfa, b, sfb], attrs) => {
    const [M, N] = scaledMmShape(a.shape, sfa.shape, b.shape, sfb.shape, attrs);
    const K = a.shape[1];
    const kBlocks = sfa.shape[1];
    const out = new Tensor([M, N]);
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        const nb = Math.floor(n / attrs.block_n);
        let final = 0;
        for (let j = 0; j < kBlocks; j++) {
          const sa = sfa.data[m * kBlocks + j];
          const sb = sfb.data[nb * kBlocks + j];
          if (attrs.accum === 'fp32') {
            final += sa * sb * blockDot(a, b, m, n, j, attrs.block_k);
            continue;
          }
          const scale = Math.fround(sa * sb);
          const [start, end] = blockRange(K, j, attrs.block_k);
          const products: number[] = [];
          for (let k = start; k < end; k++) products.push(a.data[m * K + k] * b.data[n * K + k]);
          final = attrs.accum === 'tc' ? Math.fround(final + scale * tcAccumulate(0, products)) : tcAccumulate(final, products.map((p) => scale * p));
        }
        out.data[m * N + n] = final;
      }
    }
    return out;
  },
  backward: (gradOut, [a, sfa, b, sfb], _out, attrs) => {
    const [M, K] = a.shape;
    const N = b.shape[0];
    const kBlocks = sfa.shape[1];
    const ga = new Tensor(a.shape);
    const gb = new Tensor(b.shape);
    const gsfa = new Tensor(sfa.shape);
    const gsfb = new Tensor(sfb.shape);
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        const g = gradOut.data[m * N + n];
        const nb = Math.floor(n / attrs.block_n);
        for (let j = 0; j < kBlocks; j++) {
          const sa = sfa.data[m * kBlocks + j];
          const sb = sfb.data[nb * kBlocks + j];
          const [start, end] = blockRange(K, j, attrs.block_k);
          for (let k = start; k < end; k++) {
            ga.data[m * K + k] += g * sa * sb * b.data[n * K + k];
            gb.data[n * K + k] += g * sa * sb * a.data[m * K + k];
          }
          const dot = blockDot(a, b, m, n, j, attrs.block_k);
          gsfa.data[m * kBlocks + j] += g * sb * dot;
          gsfb.data[nb * kBlocks + j] += g * sa * dot;
        }
      }
    }
    return [ga, gsfa, gb, gsfb];
  },
};
