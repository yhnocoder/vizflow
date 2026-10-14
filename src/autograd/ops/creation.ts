import { Tensor } from '../../tensor';
import { type Dtype, checkDtype, finfoMin, isIntegerDtype } from '../dtype';
import type { OpCost, OpDef } from '../op';
import { NO_FLOPS } from './cost';

const CREATION_COST: OpCost = {
  saves: [],
  flops: () => ({ forward: NO_FLOPS, backward: [] }),
};

export interface ArangeAttrs {
  start: number;
  end: number | null;
  step: number;
  dtype: Dtype | null;
}

function arangeRange({ start, end, step }: ArangeAttrs): { first: number; step: number; length: number } {
  const [first, last] = end === null ? [0, start] : [start, end];
  if (step === 0) throw new Error('arange: step must be nonzero');
  if (![first, last, step].every(Number.isFinite)) throw new Error('arange: start, end and step must be finite numbers');
  return { first, step, length: Math.max(0, Math.ceil((last - first) / step)) };
}

function arangeDtype(attrs: ArangeAttrs): Dtype {
  if (attrs.dtype !== null) return checkDtype(attrs.dtype, 'arange');
  const given = attrs.end === null ? [attrs.start, attrs.step] : [attrs.start, attrs.end, attrs.step];
  return given.every(Number.isInteger) ? 'int64' : 'fp32';
}

export const arangeOp: OpDef<ArangeAttrs> = {
  name: 'arange',
  args: ['start', 'end', 'step', 'dtype'],
  tensors: [],
  sizes: ['start', 'end'],
  defaults: { end: null, step: 1, dtype: null },
  differentiable: [],
  dtype: arangeDtype,
  int64: true,
  cost: CREATION_COST,
  shape: (_inputs, attrs) => [arangeRange(attrs).length],
  forward: (_inputs, attrs) => {
    const { first, step, length } = arangeRange(attrs);
    const values = Tensor.from(Array.from({ length }, (_, i) => first + i * step), [length]);
    return isIntegerDtype(arangeDtype(attrs)) ? values.asInt64() : values;
  },
  backward: () => [],
};

export interface CausalMaskAttrs {
  size: number;
  dtype: Dtype;
}

export const causalMaskOp: OpDef<CausalMaskAttrs> = {
  name: 'causal_mask',
  args: ['size', 'dtype'],
  tensors: [],
  sizes: ['size'],
  differentiable: [],
  dtype: (attrs) => checkDtype(attrs.dtype, 'causal_mask'),
  cost: CREATION_COST,
  shape: (_inputs, { size }) => [size, size],
  forward: (_inputs, { size, dtype }) => {
    const fill = finfoMin(checkDtype(dtype, 'causal_mask'), 'causal_mask');
    return new Tensor([size, size], Float64Array.from({ length: size * size }, (_, i) => (i % size > Math.floor(i / size) ? fill : 0)));
  },
  backward: () => [],
};

export interface FillAttrs {
  size: number[];
  dtype: Dtype | null;
}

function fillDtype(name: string) {
  return (attrs: { dtype: Dtype | null }): Dtype => (attrs.dtype === null ? 'fp32' : checkDtype(attrs.dtype, name));
}

function asDtype(values: Tensor, dtype: Dtype): Tensor {
  return isIntegerDtype(dtype) ? values.asInt64() : values;
}

function checkSize(name: string, size: readonly number[]): number[] {
  if (!size.every((d) => Number.isInteger(d) && d >= 0)) throw new Error(`${name}: size (${size.join(', ')}) must be non-negative integers`);
  return [...size];
}

function fillOp(name: 'zeros' | 'ones', value: number): OpDef<FillAttrs> {
  const dtype = fillDtype(name);
  return {
    name,
    args: ['size', 'dtype'],
    tensors: [],
    rest: 'size',
    sizes: ['size'],
    defaults: { dtype: null },
    differentiable: [],
    dtype,
    int64: true,
    cost: CREATION_COST,
    shape: (_inputs, attrs) => checkSize(name, attrs.size),
    forward: (_inputs, attrs) => {
      const shape = checkSize(name, attrs.size);
      const out = new Tensor(shape);
      out.data.fill(value);
      return asDtype(out, dtype(attrs));
    },
    backward: () => [],
  };
}

export interface FullAttrs {
  size: number[];
  fill_value: number;
  dtype: Dtype | null;
}

export const fullOp: OpDef<FullAttrs> = {
  name: 'full',
  args: ['size', 'fill_value', 'dtype'],
  tensors: [],
  sizes: ['size'],
  defaults: { dtype: null },
  differentiable: [],
  dtype: fillDtype('full'),
  int64: true,
  cost: CREATION_COST,
  shape: (_inputs, attrs) => checkSize('full', attrs.size),
  forward: (_inputs, attrs) => {
    const out = new Tensor(checkSize('full', attrs.size));
    out.data.fill(attrs.fill_value);
    return asDtype(out, fillDtype('full')(attrs));
  },
  backward: () => [],
};

export const zerosOp = fillOp('zeros', 0);
export const onesOp = fillOp('ones', 1);

export interface EyeAttrs {
  n: number;
  dtype: Dtype | null;
}

export const eyeOp: OpDef<EyeAttrs> = {
  name: 'eye',
  args: ['n', 'dtype'],
  tensors: [],
  sizes: ['n'],
  defaults: { dtype: null },
  differentiable: [],
  dtype: fillDtype('eye'),
  int64: true,
  cost: CREATION_COST,
  shape: (_inputs, { n }) => checkSize('eye', [n, n]),
  forward: (_inputs, attrs) => {
    const { n } = attrs;
    const out = new Tensor(checkSize('eye', [n, n]));
    for (let i = 0; i < n; i++) out.data[i * n + i] = 1;
    return asDtype(out, fillDtype('eye')(attrs));
  },
  backward: () => [],
};

export const zerosLikeOp: OpDef<Record<string, never>> = {
  name: 'zeros_like',
  args: ['input'],
  tensors: ['input'],
  differentiable: [false],
  int64: true,
  cost: { saves: [[]], flops: () => ({ forward: NO_FLOPS, backward: [NO_FLOPS] }) },
  shape: ([input]) => [...input],
  forward: ([input]) => (input.int64 === undefined ? new Tensor(input.shape) : new Tensor(input.shape).asInt64()),
  backward: () => [null],
};
