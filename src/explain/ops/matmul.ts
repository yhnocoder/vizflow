import type { OpCtx, OpExplain } from '../../autograd';

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
