import { Tensor, formatShape } from '../../tensor';
import type { OpDef } from '../op';

type NoAttrs = Record<string, never>;

export interface GroupMeanAttrs {
  size: number;
}

const last = (shape: readonly number[]) => shape[shape.length - 1];

function groupMeanShape(shape: readonly number[], size: number): number[] {
  if (shape.length === 0 || !Number.isInteger(size) || size <= 0 || last(shape) % size !== 0) {
    throw new Error(`group_mean: last dimension of ${formatShape(shape)} is not a multiple of ${size}`);
  }
  return [...shape.slice(0, -1), last(shape) / size];
}

export const groupMeanOp: OpDef<GroupMeanAttrs> = {
  name: 'group_mean',
  args: ['input', 'size'],
  tensors: ['input'],
  sizes: ['size'],
  shape: ([a], { size }) => groupMeanShape(a, size),
  forward: ([a], { size }) => {
    const out = new Tensor(groupMeanShape(a.shape, size));
    for (let i = 0; i < a.size; i++) out.data[Math.floor(i / size)] += a.data[i] / size;
    return out;
  },
  backward: (gradOut, [a], _out, { size }) => [new Tensor(a.shape, a.data.map((_, i) => gradOut.data[Math.floor(i / size)] / size))],
};

function sameLeading(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.slice(0, -1).every((d, i) => d === b[i]);
}

function groupMulShape(a: readonly number[], b: readonly number[]): number[] {
  if (a.length === 0 || !sameLeading(a, b) || last(a) % last(b) !== 0) {
    throw new Error(`group_mul: cannot scale ${formatShape(a)} by one value per group from ${formatShape(b)}`);
  }
  return [...a];
}

export function groupMulSource(input: readonly number[], other: readonly number[], offset: number): number {
  const width = last(input);
  const size = width / last(other);
  return Math.floor(offset / width) * last(other) + Math.floor((offset % width) / size);
}

function tileMulShape(a: readonly number[], b: readonly number[]): number[] {
  const leadingOk = b.length === 1 || sameLeading(a, b);
  if (a.length === 0 || b.length === 0 || !leadingOk || last(a) % last(b) !== 0) {
    throw new Error(`tile_mul: cannot tile ${formatShape(b)} across ${formatShape(a)}`);
  }
  return [...a];
}

export function tileMulSource(input: readonly number[], other: readonly number[], offset: number): number {
  const width = last(input);
  const row = other.length === 1 ? 0 : Math.floor(offset / width);
  return row * last(other) + ((offset % width) % last(other));
}

function scaled(name: string, shape: (a: readonly number[], b: readonly number[]) => number[], source: typeof groupMulSource): OpDef<NoAttrs> {
  return {
    name,
    args: ['input', 'other'],
    tensors: ['input', 'other'],
    shape: ([a, b]) => shape(a, b),
    forward: ([a, b]) => {
      shape(a.shape, b.shape);
      return new Tensor(a.shape, a.data.map((x, i) => x * b.data[source(a.shape, b.shape, i)]));
    },
    backward: (gradOut, [a, b]) => {
      const ga = new Tensor(a.shape);
      const gb = new Tensor(b.shape);
      for (let i = 0; i < a.size; i++) {
        const j = source(a.shape, b.shape, i);
        ga.data[i] = gradOut.data[i] * b.data[j];
        gb.data[j] += gradOut.data[i] * a.data[i];
      }
      return [ga, gb];
    },
  };
}

export const groupMulOp = scaled('group_mul', groupMulShape, groupMulSource);
export const tileMulOp = scaled('tile_mul', tileMulShape, tileMulSource);
