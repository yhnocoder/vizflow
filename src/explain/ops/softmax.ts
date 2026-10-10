import type { OpExplain } from '../../autograd';

const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

export const softmaxExplain: OpExplain = {
  pattern: 'dense',
  local: (ctx, out, _arg, input) => ctx.out.get(out) * ((same(out, input) ? 1 : 0) - ctx.out.get(input)),
  tex: {
    forward: (s) => `\\operatorname{softmax}(${s.in[0]})`,
    local: (s) => (same(s.outIndex, s.inIndex) ? `${s.out}(1-${s.out})` : `-${s.out}${s.elem(-1, s.inIndex)}`),
  },
};

export const logSoftmaxExplain: OpExplain = {
  pattern: 'dense',
  local: (ctx, out, _arg, input) => (same(out, input) ? 1 : 0) - Math.exp(ctx.out.get(input)),
  tex: {
    forward: (s) => `\\log\\operatorname{softmax}(${s.in[0]})`,
    local: (s) =>
      same(s.outIndex, s.inIndex) ? `1-\\exp(${s.elem(-1, s.inIndex)})` : `-\\exp(${s.elem(-1, s.inIndex)})`,
  },
};
