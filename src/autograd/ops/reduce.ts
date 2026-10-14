import { Tensor, normDim, normDims, reduceMap, reducedShape } from '../../tensor';
import type { OpDef } from '../op';
import { numel, overInput, pointwise } from './cost';

const LOGSUMEXP_FORWARD = 4;
const LOGSUMEXP_BACKWARD = 3;

export type Dim = number | number[];

export interface ReduceAttrs {
  dim: Dim | null;
  keepdim: boolean;
}

export interface MaxAttrs {
  dim: number;
  keepdim: boolean;
}

function grouped(input: Tensor, attrs: ReduceAttrs): { shape: number[]; map: Int32Array; count: Int32Array } {
  const dims = normDims(attrs.dim, input.ndim);
  const { shape, map } = reduceMap(input.shape, dims, attrs.keepdim);
  const count = new Int32Array(shape.reduce((a, b) => a * b, 1));
  for (let i = 0; i < map.length; i++) count[map[i]]++;
  return { shape, map, count };
}

function reducedShapeOf(shape: number[], attrs: ReduceAttrs): number[] {
  return reducedShape(shape, normDims(attrs.dim, shape.length), attrs.keepdim);
}

function spread(gradOut: Tensor, inputShape: readonly number[], map: Int32Array, scale: (group: number) => number): Tensor {
  const grad = new Tensor(inputShape);
  for (let i = 0; i < map.length; i++) grad.data[i] = gradOut.data[map[i]] * scale(map[i]);
  return grad;
}

export const sumOp: OpDef<ReduceAttrs> = {
  name: 'sum',
  args: ['input', 'dim'],
  tensors: ['input'],
  integral: 'int64',
  defaults: { dim: null, keepdim: false },
  shape: ([a], attrs) => reducedShapeOf(a, attrs),
  forward: ([a], attrs) => {
    const { shape, map } = grouped(a, attrs);
    const out = new Tensor(shape);
    for (let i = 0; i < map.length; i++) out.data[map[i]] += a.data[i];
    return out;
  },
  backward: (gradOut, [a], _out, attrs) => [spread(gradOut, a.shape, grouped(a, attrs).map, () => 1)],
  cost: overInput(1, 0, [[]]),
};

export const meanOp: OpDef<ReduceAttrs> = {
  name: 'mean',
  args: ['input', 'dim'],
  tensors: ['input'],
  defaults: { dim: null, keepdim: false },
  shape: ([a], attrs) => reducedShapeOf(a, attrs),
  forward: ([a], attrs) => {
    const { shape, map, count } = grouped(a, attrs);
    const out = new Tensor(shape);
    for (let i = 0; i < map.length; i++) out.data[map[i]] += a.data[i];
    for (let j = 0; j < out.size; j++) out.data[j] /= count[j];
    return out;
  },
  backward: (gradOut, [a], _out, attrs) => {
    const { map, count } = grouped(a, attrs);
    return [spread(gradOut, a.shape, map, (group) => 1 / count[group])];
  },
  cost: overInput(1, 1, [[]]),
};

function argmaxGroups(input: Tensor, attrs: ReduceAttrs): { shape: number[]; map: Int32Array; argmax: Int32Array } {
  const { shape, map } = grouped(input, attrs);
  const argmax = new Int32Array(shape.reduce((a, b) => a * b, 1)).fill(-1);
  for (let i = 0; i < map.length; i++) {
    const group = map[i];
    if (argmax[group] < 0 || input.data[i] > input.data[argmax[group]]) argmax[group] = i;
  }
  return { shape, map, argmax };
}

export const maxOp: OpDef<MaxAttrs> = {
  name: 'max',
  args: ['input', 'dim'],
  tensors: ['input'],
  defaults: { keepdim: false },
  shape: ([a], attrs) => reducedShapeOf(a, attrs),
  forward: ([a], attrs) => {
    const { shape, argmax } = argmaxGroups(a, attrs);
    return new Tensor(shape, Float64Array.from(argmax, (i) => a.data[i]));
  },
  backward: (gradOut, [a], _out, attrs) => {
    const grad = new Tensor(a.shape);
    const { argmax } = argmaxGroups(a, attrs);
    for (let j = 0; j < argmax.length; j++) grad.data[argmax[j]] += gradOut.data[j];
    return [grad];
  },
  cost: overInput(1, 0, [[0]]),
};

export const logsumexpOp: OpDef<ReduceAttrs> = {
  name: 'logsumexp',
  args: ['input', 'dim'],
  tensors: ['input'],
  defaults: { keepdim: false },
  floating: true,
  shape: ([a], attrs) => reducedShapeOf(a, attrs),
  forward: ([a], attrs) => {
    const { shape, map, argmax } = argmaxGroups(a, attrs);
    const out = new Tensor(shape);
    for (let i = 0; i < map.length; i++) out.data[map[i]] += Math.exp(a.data[i] - a.data[argmax[map[i]]]);
    for (let j = 0; j < out.size; j++) out.data[j] = a.data[argmax[j]] + Math.log(out.data[j]);
    return out;
  },
  backward: (gradOut, [a], out, attrs) => {
    const { map } = grouped(a, attrs);
    const grad = new Tensor(a.shape);
    for (let i = 0; i < map.length; i++) {
      grad.data[i] = gradOut.data[map[i]] * Math.exp(a.data[i] - out.data[map[i]]);
    }
    return [grad];
  },
  cost: overInput(LOGSUMEXP_FORWARD, LOGSUMEXP_BACKWARD, [[0, 'out']]),
};

export interface CumsumAttrs {
  dim: number;
}

function scan(input: Tensor, dim: number, reverse: boolean): Tensor {
  const d = normDim(dim, input.ndim);
  const size = input.shape[d];
  const stride = input.shape.slice(d + 1).reduce((a, b) => a * b, 1);
  const out = new Tensor(input.shape);
  for (let offset = 0; offset < input.size; offset++) {
    const position = Math.floor(offset / stride) % size;
    if (position !== (reverse ? size - 1 : 0)) continue;
    let total = 0;
    for (let step = 0; step < size; step++) {
      const at = offset + (reverse ? -step : step) * stride;
      total += input.data[at];
      out.data[at] = total;
    }
  }
  return out;
}

export const cumsumOp: OpDef<CumsumAttrs> = {
  name: 'cumsum',
  args: ['input', 'dim'],
  tensors: ['input'],
  shape: ([a], attrs) => {
    normDim(attrs.dim, a.length);
    return [...a];
  },
  forward: ([a], attrs) => scan(a, attrs.dim, false),
  backward: (gradOut, _inputs, _out, attrs) => [scan(gradOut, attrs.dim, true)],
  cost: pointwise(1, 1, [[]]),
};

export interface RrmsAttrs {
  eps: number;
}

export const rrmsOp: OpDef<RrmsAttrs> = {
  name: 'rrms',
  args: ['input', 'eps'],
  tensors: ['input'],
  floating: true,
  shape: ([a], { eps }) => {
    if (a.length === 0 || a[a.length - 1] === 0) throw new Error('rrms: input needs a non-empty last dimension');
    if (typeof eps !== 'number' || !Number.isFinite(eps) || eps < 0) throw new Error('rrms: eps must be a finite non-negative number');
    return [...a.slice(0, -1), 1];
  },
  forward: ([a], { eps }) => {
    const width = a.shape[a.ndim - 1];
    const out = new Tensor([...a.shape.slice(0, -1), 1]);
    for (let row = 0; row < out.size; row++) {
      let sum = 0;
      for (let col = 0; col < width; col++) sum += a.data[row * width + col] ** 2;
      out.data[row] = 1 / Math.sqrt(sum / width + eps);
    }
    return out;
  },
  backward: (gradOut, [a], out) => {
    const width = a.shape[a.ndim - 1];
    const grad = new Tensor(a.shape);
    for (let row = 0; row < out.size; row++) {
      const factor = -gradOut.data[row] * out.data[row] ** 3 / width;
      for (let col = 0; col < width; col++) grad.data[row * width + col] = factor * a.data[row * width + col];
    }
    return [grad];
  },
  cost: {
    saves: [[0, 'out']],
    flops: ([input], out) => ({ forward: [2 * numel(input) + 3 * numel(out)], backward: [[numel(input) + 4 * numel(out)]] }),
  },
};
