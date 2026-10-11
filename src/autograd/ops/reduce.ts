import { Tensor, normDims, reduceMap, reducedShape } from '../../tensor';
import type { OpDef } from '../op';
import { overInput } from './cost';

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
