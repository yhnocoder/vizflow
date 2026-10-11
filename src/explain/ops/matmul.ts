import type { OpExplain } from '../../autograd';

const startOf = (attrs: object) => (attrs as { start: number }).start;

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

export const narrowMatmulExplain: OpExplain = {
  pattern: 'contract',
  local: (ctx, out, arg, input) => {
    const [A, B] = ctx.inputs;
    const start = startOf(ctx.attrs);
    if (arg === 0) return B.get([input[1] - start, out[1]]);
    return A.get([out[0], start + input[0]]);
  },
  tex: {
    forward: (s) => `\\operatorname{narrow}(${s.in[0]},-1,${startOf(s.attrs)})\\,${s.in[1]}`,
    local: (s) => (s.arg === 0 ? s.ins[1] : s.ins[0]),
  },
};
