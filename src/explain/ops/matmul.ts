import type { LocalTexArgs, OpCtx, OpExplain } from '../../autograd';
import { blockDot, blockRange } from '../../autograd/ops/matmul';

const headOf = (attrs: object) => (attrs as { index: number }).index;

export const matmulExplain: OpExplain = {
  pattern: 'contract',
  local: (ctx, out, arg, input) => {
    const [A, B] = ctx.inputs;
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

const expertOf = (ctx: OpCtx, row: number) => Math.round(ctx.inputs[2].get([row]));

export const groupedMmExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg, input) => {
    const [x, w] = ctx.inputs;
    if (arg === 0) return w.get([expertOf(ctx, out[0]), out[1], input[1]]);
    return x.get([out[0], input[2]]);
  },
  tex: {
    forward: (s) => `\\operatorname{grouped\\_mm}(${s.in[0]},${s.in[1]},${s.in[2]})`,
    local: (s) => (s.arg === 0 ? s.elem(1, [expertOf(s.ctx, s.outIndex[0]), s.outIndex[1], s.inIndex[1]]) : s.elem(0, [s.outIndex[0], s.inIndex[2]])),
  },
  selected: (ctx, out) => {
    const [r, i] = out;
    const e = expertOf(ctx, r);
    return Array.from({ length: ctx.inputs[0].shape[1] }, (_, k) => [
      { arg: 0, index: [r, k] },
      { arg: 1, index: [e, i, k] },
    ]).flat();
  },
};

const blocksOf = (attrs: object) => attrs as { block_k: number; block_n: number };

function blockTermsTex(s: LocalTexArgs, m: number, n: number, j: number): string {
  const [start, end] = blockRange(s.ctx.inputs[0].shape[1], j, blocksOf(s.attrs).block_k);
  const termOf = (k: number) => `${s.elem(0, [m, k])}\\,${s.elem(2, [n, k])}`;
  if (end - start === 1) return termOf(start);
  if (end - start === 2) return `${termOf(start)}+${termOf(start + 1)}`;
  return `${termOf(start)}+\\cdots+${termOf(end - 1)}`;
}

export const scaledMmExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg, input) => {
    const [a, sfa, b, sfb] = ctx.inputs;
    const { block_k, block_n } = blocksOf(ctx.attrs);
    const [m, n] = out;
    const nb = Math.floor(n / block_n);
    if (arg === 0 || arg === 2) {
      const k = input[1];
      const j = Math.floor(k / block_k);
      return sfa.get([m, j]) * sfb.get([nb, j]) * (arg === 0 ? b.get([n, k]) : a.get([m, k]));
    }
    const j = input[1];
    return blockDot(a, b, m, n, j, block_k) * (arg === 1 ? sfb.get([nb, j]) : sfa.get([m, j]));
  },
  tex: {
    forward: (s) => {
      const { block_k, block_n } = blocksOf(s.attrs);
      const [a, sfa, b, sfb] = s.in.map((t) => `{${t}}`);
      return `\\Big[\\sum_{j}${sfa}_{m,j}\\,${sfb}_{\\lceil n/${block_n}\\rceil,j}\\sum_{k=${block_k}(j-1)+1}^{${block_k}j}${a}_{m,k}\\,${b}_{n,k}\\Big]_{m,n}`;
    },
    local: (s) => {
      const { block_k, block_n } = blocksOf(s.attrs);
      const [m, n] = s.outIndex;
      const nb = Math.floor(n / block_n);
      if (s.arg === 0 || s.arg === 2) {
        const k = s.inIndex[1];
        const j = Math.floor(k / block_k);
        return `${s.elem(1, [m, j])}\\,${s.elem(3, [nb, j])}\\,${s.arg === 0 ? s.elem(2, [n, k]) : s.elem(0, [m, k])}`;
      }
      const j = s.inIndex[1];
      return `${s.arg === 1 ? s.elem(3, [nb, j]) : s.elem(1, [m, j])}\\left(${blockTermsTex(s, m, n, j)}\\right)`;
    },
  },
  selected: (ctx, out) => {
    const [a, sfa] = ctx.inputs;
    const [m, n] = out;
    const nb = Math.floor(n / blocksOf(ctx.attrs).block_n);
    const K = a.shape[1];
    const kBlocks = sfa.shape[1];
    return [
      ...Array.from({ length: K }, (_, k) => ({ arg: 0, index: [m, k] })),
      ...Array.from({ length: kBlocks }, (_, j) => ({ arg: 1, index: [m, j] })),
      ...Array.from({ length: K }, (_, k) => ({ arg: 2, index: [n, k] })),
      ...Array.from({ length: kBlocks }, (_, j) => ({ arg: 3, index: [nb, j] })),
    ];
  },
};
