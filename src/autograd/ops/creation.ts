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
