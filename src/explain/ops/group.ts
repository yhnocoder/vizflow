import type { OpCtx, OpExplain } from '../../autograd';
import { groupMulSource, tileMulSource } from '../../autograd/ops/group';
import { indexOf, offsetOf } from '../../tensor';

const sizeOf = (attrs: object) => (attrs as { size: number }).size;

export const groupMeanExplain: OpExplain = {
  pattern: 'select',
  local: (ctx) => 1 / sizeOf(ctx.attrs),
  tex: {
    forward: (s) => `\\operatorname{mean}_{${sizeOf(s.attrs)}}(${s.in[0]})`,
    local: (s) => `\\frac{1}{${sizeOf(s.attrs)}}`,
  },
  selected: (ctx, out) => {
    const size = sizeOf(ctx.attrs);
    const first = offsetOf(ctx.out.shape, out) * size;
    return Array.from({ length: size }, (_, j) => ({ arg: 0, index: indexOf(ctx.inputs[0].shape, first + j) }));
  },
};

function scaledExplain(name: string, source: typeof groupMulSource): OpExplain {
  const otherIndex = (ctx: OpCtx, out: number[]) => indexOf(ctx.inputs[1].shape, source(ctx.inputs[0].shape, ctx.inputs[1].shape, offsetOf(ctx.out.shape, out)));
  return {
    pattern: 'elementwise',
    local: (ctx, out, arg) => (arg === 0 ? ctx.inputs[1].get(otherIndex(ctx, out)) : ctx.inputs[0].get(out)),
    tex: {
      forward: (s) => `\\operatorname{${name}}(${s.in[0]},${s.in[1]})`,
      local: (s) => (s.arg === 0 ? s.ins[1] : s.ins[0]),
    },
    selected: (ctx, out) => [
      { arg: 0, index: out },
      { arg: 1, index: otherIndex(ctx, out) },
    ],
  };
}

export const groupMulExplain = scaledExplain('group\\_mul', groupMulSource);
export const tileMulExplain = scaledExplain('tile\\_mul', tileMulSource);
