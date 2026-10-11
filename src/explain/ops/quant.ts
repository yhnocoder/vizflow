import type { OpCtx, OpExplain, QuantFormat } from '../../autograd';
import { type BlockAttrs, type BlockScaleAttrs, type QuantizeAttrs, blockOf } from '../../autograd/ops/quant';
import { quantMax } from '../../autograd/quant';
import { forEachIndex } from '../../tensor';

const FORMAT_TEX: Record<QuantFormat, string> = { fp8_e4m3: 'E4M3', fp8_e5m2: 'E5M2', fp4_e2m1: 'E2M1', int8: 'INT8', int4: 'INT4' };

function blockAbsMax(ctx: OpCtx, out: number[]): number[] {
  const attrs = ctx.attrs as BlockAttrs;
  const input = ctx.inputs[0];
  let best: number[] = [];
  let top = -1;
  forEachIndex(input.shape, (index, offset) => {
    const v = Math.abs(input.data[offset]);
    if (v > top && blockOf(index, attrs).every((b, d) => b === out[d])) {
      top = v;
      best = index;
    }
  });
  return best;
}

function sameAndBlock(ctx: OpCtx, out: number[]): { arg: number; index: number[] }[] {
  return [
    { arg: 0, index: out },
    { arg: 1, index: blockOf(out, ctx.attrs as BlockAttrs) },
  ];
}

function passes(ctx: OpCtx, index: number[]): boolean {
  const attrs = ctx.attrs as QuantizeAttrs;
  const scaled = Math.fround(ctx.inputs[0].get(index) / ctx.inputs[1].get(blockOf(index, attrs)));
  return Math.abs(scaled) <= quantMax(attrs.fmt);
}

export const blockScaleExplain: OpExplain = {
  pattern: 'select',
  local: () => 0,
  tex: {
    forward: (s) => `\\operatorname{scale}_{\\mathrm{${(s.attrs as BlockScaleAttrs).rule.replaceAll('_', '\\_')}}}(${s.in[0]})`,
    local: () => '0',
  },
  selected: (ctx, out) => [{ arg: 0, index: blockAbsMax(ctx, out) }],
};

export const quantizeExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg) => (arg === 0 && passes(ctx, out) ? 1 / ctx.inputs[1].get(blockOf(out, ctx.attrs as BlockAttrs)) : 0),
  tex: {
    forward: (s) => `\\operatorname{round}_{\\mathrm{${FORMAT_TEX[(s.attrs as QuantizeAttrs).fmt]}}}\\!\\big(${s.in[0]} / ${s.in[1]}\\big)`,
    local: (s) => (s.arg === 0 && passes(s.ctx, s.outIndex) ? `\\frac{1}{${s.elem(1, blockOf(s.outIndex, s.attrs as BlockAttrs))}}` : '0'),
  },
  selected: sameAndBlock,
};

export const dequantizeExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg) => (arg === 0 ? ctx.inputs[1].get(blockOf(out, ctx.attrs as BlockAttrs)) : 0),
  tex: {
    forward: (s) => `${s.in[0]}\\, ${s.in[1]}`,
    local: (s) => (s.arg === 0 ? s.elem(1, blockOf(s.outIndex, s.attrs as BlockAttrs)) : '0'),
  },
  selected: sameAndBlock,
};
