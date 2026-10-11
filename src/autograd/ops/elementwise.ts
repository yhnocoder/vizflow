import { Tensor, broadcastIndex, broadcastShapes, forEachIndex, normDim, offsetOf, unbroadcast } from '../../tensor';
import type { OpDef } from '../op';

type NoAttrs = Record<string, never>;

function zipWith(a: Tensor, b: Tensor, fn: (x: number, y: number) => number): Tensor {
  const shape = broadcastShapes(a.shape, b.shape);
  const out = new Tensor(shape);
  forEachIndex(shape, (index, offset) => {
    const x = a.data[offsetOf(a.shape, broadcastIndex(index, a.shape))];
    const y = b.data[offsetOf(b.shape, broadcastIndex(index, b.shape))];
    out.data[offset] = fn(x, y);
  });
  return out;
}

function mapWith(a: Tensor, fn: (x: number) => number): Tensor {
  return new Tensor(a.shape, a.data.map(fn));
}

interface ScalarAttrs {
  other?: number;
}

function binary(
  name: string,
  forward: (x: number, y: number) => number,
  local: (x: number, y: number) => [number, number],
): OpDef<ScalarAttrs> {
  return {
    name,
    args: ['input', 'other'],
    tensors: ['input', 'other'],
    scalars: ['other'],
    shape: ([a, b]) => (b === undefined ? [...a] : broadcastShapes(a, b)),
    forward: ([a, b], { other }) => (b === undefined ? mapWith(a, (x) => forward(x, other as number)) : zipWith(a, b, forward)),
    backward: (gradOut, [a, b], _out, { other }) => {
      if (b === undefined) {
        return [new Tensor(a.shape, gradOut.data.map((g, i) => g * local(a.data[i], other as number)[0]))];
      }
      const ga = zipWith(a, b, (x, y) => local(x, y)[0]);
      const gb = zipWith(a, b, (x, y) => local(x, y)[1]);
      for (let i = 0; i < gradOut.size; i++) {
        ga.data[i] *= gradOut.data[i];
        gb.data[i] *= gradOut.data[i];
      }
      return [unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)];
    },
  };
}

function unary<A extends object = NoAttrs>(
  name: string,
  forward: (x: number, attrs: A) => number,
  local: (x: number, y: number, attrs: A) => number,
  args: string[] = ['input'],
): OpDef<A> {
  return {
    name,
    args,
    tensors: ['input'],
    shape: ([a]) => [...a],
    forward: ([a], attrs) => mapWith(a, (x) => forward(x, attrs)),
    backward: (gradOut, [a], out, attrs) =>
      [new Tensor(a.shape, gradOut.data.map((g, i) => g * local(a.data[i], out.data[i], attrs)))],
  };
}

export function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

export const addOp = binary('add', (x, y) => x + y, () => [1, 1]);
export const subOp = binary('sub', (x, y) => x - y, () => [1, -1]);
export const mulOp = binary('mul', (x, y) => x * y, (x, y) => [y, x]);
export const divOp = binary('div', (x, y) => x / y, (x, y) => [1 / y, -x / (y * y)]);
export const negOp = unary('neg', (x) => -x, () => -1);
export const expOp = unary('exp', Math.exp, (_x, y) => y);
export const logOp = unary('log', Math.log, (x) => 1 / x);
export const powOp = unary<{ exponent: number }>(
  'pow',
  (x, { exponent }) => x ** exponent,
  (x, _y, { exponent }) => exponent * x ** (exponent - 1),
  ['input', 'exponent'],
);
export const rsqrtOp = unary('rsqrt', (x) => 1 / Math.sqrt(x), (_x, y) => -0.5 * y ** 3);
export const sqrtOp = unary('sqrt', Math.sqrt, (_x, y) => 0.5 / y);
export const sigmoidOp = unary('sigmoid', sigmoid, (_x, y) => y * (1 - y));
export const siluOp = unary('silu', (x) => x * sigmoid(x), (x) => sigmoid(x) * (1 + x * (1 - sigmoid(x))));

interface TransposeAttrs {
  dim0: number;
  dim1: number;
}

function swapped(shape: readonly number[], attrs: TransposeAttrs): { d0: number; d1: number; shape: number[] } {
  const d0 = normDim(attrs.dim0, shape.length);
  const d1 = normDim(attrs.dim1, shape.length);
  const out = [...shape];
  out[d0] = shape[d1];
  out[d1] = shape[d0];
  return { d0, d1, shape: out };
}

function transposeTensor(input: Tensor, attrs: TransposeAttrs): Tensor {
  const { d0, d1, shape } = swapped(input.shape, attrs);
  const out = new Tensor(shape);
  forEachIndex(shape, (index, offset) => {
    const source = [...index];
    source[d0] = index[d1];
    source[d1] = index[d0];
    out.data[offset] = input.data[offsetOf(input.shape, source)];
  });
  return out;
}

export const transposeOp: OpDef<TransposeAttrs> = {
  name: 'transpose',
  args: ['input', 'dim0', 'dim1'],
  tensors: ['input'],
  shape: ([a], attrs) => swapped(a, attrs).shape,
  forward: ([a], attrs) => transposeTensor(a, attrs),
  backward: (gradOut, _inputs, _out, attrs) => [transposeTensor(gradOut, attrs)],
};
