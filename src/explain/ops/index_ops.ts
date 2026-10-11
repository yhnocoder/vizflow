import type { OpExplain } from '../../autograd';
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
