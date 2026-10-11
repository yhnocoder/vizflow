import type { OpCtx, OpExplain } from '../../autograd';
import { type PadAttrs, convTaps, padSource } from '../../autograd/ops/index_ops';
import { indexOf, normDim, offsetOf } from '../../tensor';

export const gatherExplain: OpExplain = {
  pattern: 'select',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{gather}(${s.in[0]},${s.in[1]})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const dim = normDim((ctx.attrs as { dim: number }).dim, out.length);
    const index = [...out];
    index[dim] = Math.round(ctx.inputs[1].get(out));
    return [{ arg: 0, index }];
  },
};

export const embeddingExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `${s.in[1]}[${s.in[0]}]`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const row = Math.round(ctx.inputs[0].get(out.slice(0, -1)));
    return [{ arg: 1, index: [row, out[out.length - 1]] }];
  },
};

export const narrowExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => {
      const { dim, start, length } = s.attrs as { dim: number; start: number; length: number };
      return `\\operatorname{narrow}(${s.in[0]},${dim},${start},${length})`;
    },
    local: () => '1',
  },
  selected: (ctx, out) => {
    const attrs = ctx.attrs as { dim: number; start: number };
    const dim = normDim(attrs.dim, out.length);
    const index = [...out];
    index[dim] += attrs.start < 0 ? attrs.start + ctx.inputs[0].shape[dim] : attrs.start;
    return [{ arg: 0, index }];
  },
};

export const catExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{cat}(${s.in.join(',')})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const dim = normDim((ctx.attrs as { dim: number }).dim, out.length);
    const index = [...out];
    let arg = 0;
    while (index[dim] >= ctx.inputs[arg].shape[dim]) {
      index[dim] -= ctx.inputs[arg].shape[dim];
      arg++;
    }
    return [{ arg, index }];
  },
};

export const viewExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{view}(${s.in[0]})`,
    local: () => '1',
  },
  selected: (ctx, out) => [{ arg: 0, index: indexOf(ctx.inputs[0].shape, offsetOf(ctx.out.shape, out)) }],
};

export const selectExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => {
      const { dim, index } = s.attrs as { dim: number; index: number };
      return `\\operatorname{select}(${s.in[0]},${dim},${index})`;
    },
    local: () => '1',
  },
  selected: (ctx, out) => {
    const attrs = ctx.attrs as { dim: number; index: number };
    const dim = normDim(attrs.dim, ctx.inputs[0].shape.length);
    const index = attrs.index < 0 ? attrs.index + ctx.inputs[0].shape[dim] : attrs.index;
    return [{ arg: 0, index: [...out.slice(0, dim), index, ...out.slice(dim)] }];
  },
};

export const repeatInterleaveExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => {
      const { repeats, dim } = s.attrs as { repeats: number; dim: number };
      return `\\operatorname{repeat\\_interleave}(${s.in[0]},${repeats},${dim})`;
    },
    local: () => '1',
  },
  selected: (ctx, out) => {
    const attrs = ctx.attrs as { repeats: number; dim: number };
    const dim = normDim(attrs.dim, out.length);
    const index = [...out];
    index[dim] = Math.floor(out[dim] / attrs.repeats);
    return [{ arg: 0, index }];
  },
};

function convTap(ctx: OpCtx, out: number[], arg: number, input: number[]): { s: number; j: number } {
  const K = ctx.inputs[1].shape[1];
  const t = out[0];
  return arg === 0 ? { s: input[0], j: input[0] - t + K - 1 } : { s: t - K + 1 + input[1], j: input[1] };
}

export const causalConv1dExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg, input) => {
    const { s, j } = convTap(ctx, out, arg, input);
    const c = out[1];
    return arg === 0 ? ctx.inputs[1].get([c, j]) : ctx.inputs[0].get([s, c]);
  },
  tex: {
    forward: (s) => `\\operatorname{conv1d}(${s.in[0]},${s.in[1]})`,
    local: (s) => {
      const { s: row, j } = convTap(s.ctx, s.outIndex, s.arg, s.inIndex);
      const c = s.outIndex[1];
      return s.arg === 0 ? s.elem(1, [c, j]) : s.elem(0, [row, c]);
    },
  },
  selected: (ctx, out) => {
    const [T] = ctx.inputs[0].shape;
    const K = ctx.inputs[1].shape[1];
    const c = out[1];
    return convTaps(T, K, out[0]).flatMap(({ s, j }) => [
      { arg: 0, index: [s, c] },
      { arg: 1, index: [c, j] },
    ]);
  },
};

export const topkMaskExplain: OpExplain = {
  pattern: 'dense',
  local: () => 0,
  tex: {
    forward: (s) => `\\operatorname{topk\\_mask}(${s.in[0]},${(s.attrs as { k: number }).k})`,
    local: () => '0',
  },
};

export const unsqueezeExplain: OpExplain = {
  ...viewExplain,
  tex: {
    forward: (s) => `\\operatorname{unsqueeze}(${s.in[0]},${(s.attrs as { dim: number }).dim})`,
    local: () => '1',
  },
};

export const padExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{pad}(${s.in[0]})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const source = padSource(ctx.inputs[0].shape, ctx.attrs as PadAttrs, out);
    return source === undefined ? [] : [{ arg: 0, index: source }];
  },
};
