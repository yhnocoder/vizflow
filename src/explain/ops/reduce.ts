import type { OpCtx, OpExplain } from '../../autograd';
import { normDims, offsetOf, reduceMap } from '../../tensor';

interface ReduceAttrs {
  dim: number | number[] | null;
  keepdim: boolean;
}

function groupSize(ctx: OpCtx): number {
  return ctx.inputs[0].size / ctx.out.size;
}

function isFirstMax(ctx: OpCtx, out: number[], input: number[]): boolean {
  const attrs = ctx.attrs as ReduceAttrs;
  const x = ctx.inputs[0];
  const { map } = reduceMap(x.shape, normDims(attrs.dim, x.ndim), attrs.keepdim);
  const group = offsetOf(ctx.out.shape, out);
  const target = ctx.out.get(out);
  for (let offset = 0; offset < map.length; offset++) {
    if (map[offset] !== group || x.data[offset] !== target) continue;
    return offset === offsetOf(x.shape, input);
  }
  return false;
}

export const sumExplain: OpExplain = {
  pattern: 'reduce',
  local: () => 1,
  tex: { forward: (s) => `\\sum ${s.in[0]}`, local: () => '1' },
};

export const meanExplain: OpExplain = {
  pattern: 'reduce',
  local: (ctx) => 1 / groupSize(ctx),
  tex: {
    forward: (s) => `\\operatorname{mean}(${s.in[0]})`,
    local: (s) => `\\frac{1}{${groupSize(s.ctx)}}`,
  },
};

export const maxExplain: OpExplain = {
  pattern: 'reduce',
  local: (ctx, out, _arg, input) => (isFirstMax(ctx, out, input) ? 1 : 0),
  tex: {
    forward: (s) => `\\operatorname{max}(${s.in[0]})`,
    local: (s) => (isFirstMax(s.ctx, s.outIndex, s.inIndex) ? '1' : '0'),
  },
};

export const logsumexpExplain: OpExplain = {
  pattern: 'reduce',
  local: (ctx, out, _arg, input) => (ctx.out.get(out) === -Infinity ? 0 : Math.exp(ctx.inputs[0].get(input) - ctx.out.get(out))),
  tex: {
    forward: (s) => `\\log\\sum\\exp(${s.in[0]})`,
    local: (s) => `\\exp(${s.in}-${s.out})`,
  },
};
