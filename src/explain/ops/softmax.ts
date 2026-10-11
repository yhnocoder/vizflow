import type { OpCtx, OpExplain } from '../../autograd';
import { lseWeightPair } from '../../autograd/ops/softmax';
import { offsetOf } from '../../tensor';

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

function mergeWeight(ctx: OpCtx, out: number[], arg: number): number {
  const [, la, , lb] = ctx.inputs;
  const row = Math.floor(offsetOf(ctx.out.shape, out) / ctx.out.shape[ctx.out.ndim - 1]);
  return lseWeightPair(la.data[row], lb.data[row])[arg < 2 ? 0 : 1];
}

export const lseMergeExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg) => {
    const w = mergeWeight(ctx, out, arg);
    if (arg % 2 === 0 || w === 0) return w;
    return w * (ctx.inputs[arg - 1].get(out) - ctx.out.get(out));
  },
  tex: {
    forward: (s) => `\\operatorname{lse\\_merge}(${s.in.join(',')})`,
    local: (s) => {
      const w = `\\frac{e^{${s.ins[s.arg < 2 ? 1 : 3]}}}{e^{${s.ins[1]}}+e^{${s.ins[3]}}}`;
      return s.arg % 2 === 0 ? w : `${w}\\left(${s.ins[s.arg - 1]}-${s.out}\\right)`;
    },
  },
};
