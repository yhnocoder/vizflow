import { Tensor, forEachIndex, formatShape, offsetOf } from '../../tensor';
import type { Dtype } from '../dtype';
import type { OpDef } from '../op';
import { QUANT_FORMATS, type QuantFormat, SCALE_RULES, type ScaleRule, quantMax, roundTo, scaleOf } from '../quant';

export interface BlockAttrs {
  br: number;
  bc: number;
}

export interface BlockScaleAttrs extends BlockAttrs {
  fmt: QuantFormat;
  rule: ScaleRule;
}

export interface QuantizeAttrs extends BlockAttrs {
  fmt: QuantFormat;
}

const SCALE_DTYPES: Record<ScaleRule, Dtype> = { amax: 'fp32', ue8m0: 'e8m0', mx: 'e8m0', e4m3: 'fp8_e4m3' };

function checkChoice<T extends string>(op: string, key: string, value: string, choices: readonly T[]): T {
  if (!(choices as readonly string[]).includes(value)) throw new Error(`${op}: unknown ${key} '${value}', expected one of ${choices.join(', ')}`);
  return value as T;
}

function blockShape(op: string, shape: readonly number[], attrs: BlockAttrs): number[] {
  const { br, bc } = attrs;
  if (shape.length === 0) throw new Error(`${op}: input must have at least one dimension`);
  if (!Number.isInteger(br) || !Number.isInteger(bc) || br < 1 || bc < 1) throw new Error(`${op}: block sizes br ${br} and bc ${bc} must be positive integers`);
  const cols = Math.ceil(shape[shape.length - 1] / bc);
  if (shape.length === 1) {
    if (br !== 1) throw new Error(`${op}: a one-dimensional input ${formatShape(shape)} needs br = 1, got ${br}`);
    return [cols];
  }
  return [...shape.slice(0, -2), Math.ceil(shape[shape.length - 2] / br), cols];
}

export function blockOf(index: readonly number[], attrs: BlockAttrs): number[] {
  const n = index.length;
  if (n === 1) return [Math.floor(index[0] / attrs.bc)];
  return [...index.slice(0, -2), Math.floor(index[n - 2] / attrs.br), Math.floor(index[n - 1] / attrs.bc)];
}

function checkScale(op: string, input: readonly number[], scale: readonly number[], attrs: BlockAttrs): number[] {
  const expected = blockShape(op, input, attrs);
  if (formatShape(expected) !== formatShape(scale)) {
    throw new Error(`${op}: scale ${formatShape(scale)} does not match the blocks ${formatShape(expected)} of input ${formatShape(input)}`);
  }
  return [...input];
}

export const blockScaleOp: OpDef<BlockScaleAttrs> = {
  name: 'block_scale',
  args: ['input', 'br', 'bc', 'fmt', 'rule'],
  tensors: ['input'],
  sizes: ['br', 'bc'],
  defaults: { fmt: 'fp8_e4m3', rule: 'amax' },
  differentiable: [false],
  dtype: (attrs) => SCALE_DTYPES[attrs.rule],
  shape: ([input], attrs) => {
    checkChoice('block_scale', 'fmt', attrs.fmt, QUANT_FORMATS);
    checkChoice('block_scale', 'rule', attrs.rule, SCALE_RULES);
    if (attrs.rule === 'mx') scaleOf(attrs.fmt, attrs.rule, 0);
    return blockShape('block_scale', input, attrs);
  },
  forward: ([input], attrs) => {
    const shape = blockShape('block_scale', input.shape, attrs);
    const amax = new Float64Array(shape.reduce((a, b) => a * b, 1));
    forEachIndex(input.shape, (index, offset) => {
      const o = offsetOf(shape, blockOf(index, attrs));
      amax[o] = Math.max(amax[o], Math.abs(input.data[offset]));
    });
    return new Tensor(shape, amax.map((a) => scaleOf(attrs.fmt, attrs.rule, a)));
  },
  backward: () => [null],
};

function withScale(input: Tensor, scale: Tensor, attrs: BlockAttrs, fn: (value: number, s: number, offset: number) => number): Tensor {
  const out = new Tensor(input.shape);
  forEachIndex(input.shape, (index, offset) => {
    out.data[offset] = fn(input.data[offset], scale.get(blockOf(index, attrs)), offset);
  });
  return out;
}

export const quantizeOp: OpDef<QuantizeAttrs> = {
  name: 'quantize',
  args: ['input', 'scale', 'br', 'bc', 'fmt'],
  tensors: ['input', 'scale'],
  sizes: ['br', 'bc'],
  defaults: { fmt: 'fp8_e4m3' },
  differentiable: [true, false],
  ste: true,
  dtype: (attrs) => attrs.fmt,
  shape: ([input, scale], attrs) => {
    checkChoice('quantize', 'fmt', attrs.fmt, QUANT_FORMATS);
    return checkScale('quantize', input, scale, attrs);
  },
  forward: ([input, scale], attrs) => {
    checkScale('quantize', input.shape, scale.shape, attrs);
    return withScale(input, scale, attrs, (x, s) => roundTo(attrs.fmt, Math.fround(x / s)));
  },
  backward: (gradOut, [input, scale], _out, attrs) => {
    const qmax = quantMax(attrs.fmt);
    return [withScale(input, scale, attrs, (x, s, offset) => (Math.abs(Math.fround(x / s)) > qmax ? 0 : gradOut.data[offset] / s)), null];
  },
};

export const dequantizeOp: OpDef<BlockAttrs> = {
  name: 'dequantize',
  args: ['input', 'scale', 'br', 'bc'],
  tensors: ['input', 'scale'],
  sizes: ['br', 'bc'],
  differentiable: [true, false],
  shape: ([input, scale], attrs) => checkScale('dequantize', input, scale, attrs),
  forward: ([input, scale], attrs) => {
    checkScale('dequantize', input.shape, scale.shape, attrs);
    return withScale(input, scale, attrs, (q, s) => q * s);
  },
  backward: (gradOut, [input, scale], _out, attrs) => [withScale(input, scale, attrs, (_q, s, offset) => gradOut.data[offset] * s), null],
};
