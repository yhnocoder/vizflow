import { Tensor, normDim, reduceMap } from '../../tensor';
import type { OpDef } from '../op';
import { overInput } from './cost';

const SOFTMAX_FORWARD = 5;
const SOFTMAX_BACKWARD = 4;

export interface SoftmaxAttrs {
  dim: number;
}

function slices(input: Tensor, attrs: SoftmaxAttrs): { map: Int32Array; groups: number } {
  const { shape, map } = reduceMap(input.shape, [normDim(attrs.dim, input.ndim)], false);
  return { map, groups: shape.reduce((a, b) => a * b, 1) };
}

function logSoftmax(input: Tensor, attrs: SoftmaxAttrs): Tensor {
  const { map, groups } = slices(input, attrs);
  const max = new Float64Array(groups).fill(-Infinity);
  for (let i = 0; i < map.length; i++) max[map[i]] = Math.max(max[map[i]], input.data[i]);
  const sum = new Float64Array(groups);
  for (let i = 0; i < map.length; i++) sum[map[i]] += Math.exp(input.data[i] - max[map[i]]);
  const out = new Tensor(input.shape);
  for (let i = 0; i < map.length; i++) out.data[i] = input.data[i] - max[map[i]] - Math.log(sum[map[i]]);
  return out;
}

function groupDot(a: Tensor, b: Tensor, map: Int32Array, groups: number): Float64Array {
  const dot = new Float64Array(groups);
  for (let i = 0; i < map.length; i++) dot[map[i]] += a.data[i] * b.data[i];
  return dot;
}

export const softmaxOp: OpDef<SoftmaxAttrs> = {
  name: 'softmax',
  args: ['input', 'dim'],
  tensors: ['input'],
  shape: ([a]) => [...a],
  forward: ([a], attrs) => {
    const out = logSoftmax(a, attrs);
    for (let i = 0; i < out.size; i++) out.data[i] = Math.exp(out.data[i]);
    return out;
  },
  backward: (gradOut, [a], out, attrs) => {
    const { map, groups } = slices(a, attrs);
    const dot = groupDot(gradOut, out, map, groups);
    const grad = new Tensor(a.shape);
    for (let i = 0; i < map.length; i++) grad.data[i] = out.data[i] * (gradOut.data[i] - dot[map[i]]);
    return [grad];
  },
  cost: overInput(SOFTMAX_FORWARD, SOFTMAX_BACKWARD, [['out']]),
};

export const logSoftmaxOp: OpDef<SoftmaxAttrs> = {
  name: 'log_softmax',
  args: ['input', 'dim'],
  tensors: ['input'],
  shape: ([a]) => [...a],
  forward: ([a], attrs) => logSoftmax(a, attrs),
  backward: (gradOut, [a], out, attrs) => {
    const { map, groups } = slices(a, attrs);
    const sum = new Float64Array(groups);
    for (let i = 0; i < map.length; i++) sum[map[i]] += gradOut.data[i];
    const grad = new Tensor(a.shape);
    for (let i = 0; i < map.length; i++) grad.data[i] = gradOut.data[i] - Math.exp(out.data[i]) * sum[map[i]];
    return [grad];
  },
  cost: overInput(SOFTMAX_FORWARD, SOFTMAX_BACKWARD, [['out']]),
};
