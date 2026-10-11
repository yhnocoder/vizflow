import { Tensor, formatShape, normDim, reduceMap } from '../../tensor';
import type { OpDef } from '../op';

export interface SoftmaxAttrs {
  dim: number;
}

function slices(input: Tensor, attrs: SoftmaxAttrs): { map: Int32Array; groups: number } {
  const { shape, map } = reduceMap(input.shape, [normDim(attrs.dim, input.ndim)], false);
  return { map, groups: shape.reduce((a, b) => a * b, 1) };
}

function groupMax(input: Tensor, map: Int32Array, groups: number): Float64Array {
  const max = new Float64Array(groups).fill(-Infinity);
  for (let i = 0; i < map.length; i++) max[map[i]] = Math.max(max[map[i]], input.data[i]);
  return max;
}

function logSoftmax(input: Tensor, attrs: SoftmaxAttrs): Tensor {
  const { map, groups } = slices(input, attrs);
  const max = groupMax(input, map, groups);
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
    const { map, groups } = slices(a, attrs);
    const max = groupMax(a, map, groups);
    const out = logSoftmax(a, attrs);
    for (let i = 0; i < out.size; i++) out.data[i] = max[map[i]] === -Infinity ? 0 : Math.exp(out.data[i]);
    return out;
  },
  backward: (gradOut, [a], out, attrs) => {
    const { map, groups } = slices(a, attrs);
    const dot = groupDot(gradOut, out, map, groups);
    const grad = new Tensor(a.shape);
    for (let i = 0; i < map.length; i++) grad.data[i] = out.data[i] * (gradOut.data[i] - dot[map[i]]);
    return [grad];
  },
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
};

function lseMergeShape(oa: readonly number[], la: readonly number[], ob: readonly number[], lb: readonly number[]): number[] {
  const rows = [...oa.slice(0, -1), 1];
  const same = (x: readonly number[], y: readonly number[]) => x.length === y.length && x.every((v, i) => v === y[i]);
  if (oa.length === 0 || !same(oa, ob) || !same(la, rows) || !same(lb, rows)) {
    throw new Error(`lse_merge: o_a ${formatShape(oa)} and o_b ${formatShape(ob)} must have the same shape, and lse_a ${formatShape(la)}, lse_b ${formatShape(lb)} must be ${formatShape(rows)}`);
  }
  return [...oa];
}

export function lseWeightPair(a: number, b: number): [number, number] {
  const max = Math.max(a, b);
  if (max === -Infinity) return [0, 0];
  const total = max + Math.log(Math.exp(a - max) + Math.exp(b - max));
  return [a === -Infinity ? 0 : Math.exp(a - total), b === -Infinity ? 0 : Math.exp(b - total)];
}

function lseWeights(la: Tensor, lb: Tensor): { wa: Float64Array; wb: Float64Array } {
  const wa = new Float64Array(la.size);
  const wb = new Float64Array(la.size);
  for (let r = 0; r < la.size; r++) [wa[r], wb[r]] = lseWeightPair(la.data[r], lb.data[r]);
  return { wa, wb };
}

export const lseMergeOp: OpDef<Record<string, never>> = {
  name: 'lse_merge',
  args: ['o_a', 'lse_a', 'o_b', 'lse_b'],
  tensors: ['o_a', 'lse_a', 'o_b', 'lse_b'],
  shape: ([oa, la, ob, lb]) => lseMergeShape(oa, la, ob, lb),
  forward: ([oa, la, ob, lb]) => {
    const out = new Tensor(lseMergeShape(oa.shape, la.shape, ob.shape, lb.shape));
    const { wa, wb } = lseWeights(la, lb);
    const d = oa.shape[oa.ndim - 1];
    for (let i = 0; i < out.size; i++) {
      const r = Math.floor(i / d);
      out.data[i] = (wa[r] === 0 ? 0 : wa[r] * oa.data[i]) + (wb[r] === 0 ? 0 : wb[r] * ob.data[i]);
    }
    return out;
  },
  backward: (gradOut, [oa, la, ob, lb], out) => {
    const { wa, wb } = lseWeights(la, lb);
    const d = oa.shape[oa.ndim - 1];
    const goa = new Tensor(oa.shape);
    const gob = new Tensor(ob.shape);
    const gla = new Tensor(la.shape);
    const glb = new Tensor(lb.shape);
    for (let i = 0; i < out.size; i++) {
      const r = Math.floor(i / d);
      const g = gradOut.data[i];
      goa.data[i] = wa[r] * g;
      gob.data[i] = wb[r] * g;
      if (wa[r] !== 0) gla.data[r] += wa[r] * g * (oa.data[i] - out.data[i]);
      if (wb[r] !== 0) glb.data[r] += wb[r] * g * (ob.data[i] - out.data[i]);
    }
    return [goa, gla, gob, glb];
  },
};
