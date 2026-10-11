import type { OpCtx, OpExplain } from '../../autograd';
import { type SolveTriangularAttrs, inTriangle, isBatched, solveTriangular } from '../../autograd/ops/matmul';
import { Tensor, broadcastIndex } from '../../tensor';

const headOf = (attrs: object) => (attrs as { index: number }).index;

export const matmulExplain: OpExplain = {
  pattern: 'contract',
  local: (ctx, out, arg, input) => {
    const [A, B] = ctx.inputs;
    if (isBatched(A.shape, B.shape)) {
      const batch = out.slice(0, -2);
      if (arg === 0) return B.get([...broadcastIndex(batch, B.shape.slice(0, -2)), input[input.length - 1], out[out.length - 1]]);
      return A.get([...broadcastIndex(batch, A.shape.slice(0, -2)), out[out.length - 2], input[input.length - 2]]);
    }
    if (arg === 0) {
      const k = input[input.length - 1];
      return B.ndim === 1 ? B.get([k]) : B.get([k, out[out.length - 1]]);
    }
    const k = input[0];
    return A.ndim === 1 ? A.get([k]) : A.get([out[0], k]);
  },
  tex: {
    forward: (s) => `${s.in[0]}${s.in[1]}`,
    local: (s) => (s.arg === 0 ? s.ins[1] : s.ins[0]),
  },
};

export const selectMatmulExplain: OpExplain = {
  pattern: 'contract',
  local: (ctx, out, arg, input) => {
    const [A, B] = ctx.inputs;
    if (arg === 0) return B.get([input[2], out[1]]);
    return A.get([out[0], headOf(ctx.attrs), input[0]]);
  },
  tex: {
    forward: (s) => `\\operatorname{select}(${s.in[0]},1,${headOf(s.attrs)})\\,${s.in[1]}`,
    local: (s) => (s.arg === 0 ? s.ins[1] : s.ins[0]),
  },
};

function triangularMatrixTex(a: string, attrs: SolveTriangularAttrs): string {
  if (attrs.unitriangular) return attrs.upper ? `(I+\\operatorname{triu}(${a},1))` : `(I+\\operatorname{tril}(${a},-1))`;
  return attrs.upper ? `\\operatorname{triu}(${a})` : `\\operatorname{tril}(${a})`;
}

function inverseOf(ctx: OpCtx): Tensor {
  const A = ctx.inputs[0];
  const n = A.shape[A.ndim - 1];
  const eye = new Tensor(A.shape);
  for (let m = 0; m < A.size / (n * n); m++) {
    for (let i = 0; i < n; i++) eye.data[m * n * n + i * n + i] = 1;
  }
  return solveTriangular(A, eye, ctx.attrs as SolveTriangularAttrs);
}

export const solveTriangularExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg, input) => {
    const inverse = inverseOf(ctx);
    const batch = out.slice(0, -2);
    const row = out[out.length - 2];
    if (arg === 1) return inverse.get([...batch, row, input[input.length - 2]]);
    const p = input[input.length - 2];
    const q = input[input.length - 1];
    return -inverse.get([...batch, row, p]) * ctx.out.get([...batch, q, out[out.length - 1]]);
  },
  tex: {
    forward: (s) => `\\operatorname{solve\\_triangular}(${s.in[0]},${s.in[1]})`,
    local: (s) => {
      const matrix = triangularMatrixTex(s.ins[0], s.attrs as SolveTriangularAttrs);
      const row = s.outIndex[s.outIndex.length - 2] + 1;
      const p = s.inIndex[s.inIndex.length - 2];
      if (s.arg === 1) return `\\left[${matrix}^{-1}\\right]_{${row},${p + 1}}`;
      const q = s.inIndex[s.inIndex.length - 1];
      const x = s.elem(-1, [...s.outIndex.slice(0, -2), q, s.outIndex[s.outIndex.length - 1]]);
      return `-\\left[${matrix}^{-1}\\right]_{${row},${p + 1}}${x}`;
    },
  },
  selected: (ctx, out) => {
    const attrs = ctx.attrs as SolveTriangularAttrs;
    const n = ctx.inputs[0].shape[ctx.inputs[0].ndim - 1];
    const batch = out.slice(0, -2);
    const row = out[out.length - 2];
    const col = out[out.length - 1];
    const reaches = (p: number) => (attrs.upper ? p >= row : p <= row);
    const picks: { arg: number; index: number[] }[] = [];
    for (let p = 0; p < n; p++) {
      if (!reaches(p)) continue;
      picks.push({ arg: 1, index: [...batch, p, col] });
      for (let q = 0; q < n; q++) {
        if (inTriangle(p, q, attrs)) picks.push({ arg: 0, index: [...batch, p, q] });
      }
    }
    return picks;
  },
};
