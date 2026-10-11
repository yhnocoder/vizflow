import type { OpCtx, OpExplain } from '../../autograd';

interface NllLossAttrs {
  reduction: 'mean' | 'sum' | 'none';
  ignore_index: number;
}

function targets(ctx: OpCtx): number[] {
  return Array.from(ctx.inputs[1].data, (v) => Math.round(v));
}

function counted(ctx: OpCtx): number {
  const { ignore_index } = ctx.attrs as NllLossAttrs;
  return targets(ctx).filter((t) => t !== ignore_index).length;
}

export const nllLossExplain: OpExplain = {
  pattern: 'select',
  local: (ctx) => ((ctx.attrs as NllLossAttrs).reduction === 'mean' ? -1 / counted(ctx) : -1),
  tex: {
    forward: (s) => `\\operatorname{nll\\_loss}(${s.in[0]},${s.in[1]})`,
    local: (s) => ((s.attrs as NllLossAttrs).reduction === 'mean' ? `-\\frac{1}{${counted(s.ctx)}}` : '-1'),
  },
  selected: (ctx, out) => {
    const { reduction, ignore_index } = ctx.attrs as NllLossAttrs;
    const picks = targets(ctx);
    if (ctx.inputs[0].ndim === 1) {
      return picks[0] === ignore_index ? [] : [{ arg: 0, index: [picks[0]] }];
    }
    const rows = reduction === 'none' ? [out[0]] : picks.map((_, row) => row);
    return rows.filter((row) => picks[row] !== ignore_index).map((row) => ({ arg: 0, index: [row, picks[row]] }));
  },
};
