import { Tensor, broadcastIndex, broadcastShapes, forEachIndex, normDim, offsetOf, unbroadcast } from '../../tensor';
import type { OpCost, OpDef } from '../op';
import { NO_FLOPS, VIEW_COST, movement, numel, pointwise } from './cost';

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
  cost: OpCost,
): OpDef<ScalarAttrs> {
  return {
    name,
    cost,
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

export const whereOp: OpDef<{ other: number }> = {
  name: 'where',
  args: ['condition', 'input', 'other'],
  tensors: ['condition', 'input'],
  differentiable: [false, true],
  shape: ([condition, input]) => broadcastShapes(condition, input),
  forward: ([condition, input], { other }) => zipWith(condition, input, (keep, x) => (keep !== 0 ? x : other)),
  backward: (gradOut, [condition, input]) => [null, unbroadcast(zipWith(condition, gradOut, (keep, g) => (keep !== 0 ? g : 0)), input.shape)],
  cost: movement(false, [[], [0]]),
};

function unary<A extends object = NoAttrs>(
  name: string,
  forward: (x: number, attrs: A) => number,
  local: (x: number, y: number, attrs: A) => number,
  cost: OpCost,
  args: string[] = ['input'],
): OpDef<A> {
  return {
    name,
    cost,
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

function additive(negated: boolean[]): OpCost {
  return {
    saves: [[], []],
    flops: (inputs, out) => ({
      forward: [numel(out)],
      backward: inputs.map((shape, k) => (negated[k] || numel(shape) < numel(out) ? [numel(out)] : NO_FLOPS)),
    }),
  };
}

const NONLINEAR_FORWARD = 1;
const NONLINEAR_BACKWARD = 2;
const DIV_OTHER_BACKWARD = 3;

const nonlinear = (saved: 0 | 'out') => pointwise(NONLINEAR_FORWARD, NONLINEAR_BACKWARD, [[saved]]);

export const addOp = binary('add', (x, y) => x + y, () => [1, 1], additive([false, false]));
export const subOp = binary('sub', (x, y) => x - y, () => [1, -1], additive([false, true]));
export const mulOp = binary('mul', (x, y) => x * y, (x, y) => [y, x], pointwise(1, 1, [[1], [0]]));
export const divOp = binary('div', (x, y) => x / y, (x, y) => [1 / y, -x / (y * y)], {
  saves: [[1], [0, 1]],
  flops: (inputs, out) => ({ forward: [numel(out)], backward: inputs.map((_, k) => (k === 0 ? [numel(out)] : [DIV_OTHER_BACKWARD, numel(out)])) }),
});
export const negOp = unary('neg', (x) => -x, () => -1, pointwise(1, 1, [[]]));
export const expOp = unary('exp', Math.exp, (_x, y) => y, nonlinear('out'));
export const logOp = unary('log', Math.log, (x) => 1 / x, nonlinear(0));
export const powOp = unary<{ exponent: number }>(
  'pow',
  (x, { exponent }) => x ** exponent,
  (x, _y, { exponent }) => exponent * x ** (exponent - 1),
  nonlinear(0),
  ['input', 'exponent'],
);
export const rsqrtOp: OpDef<{ eps: number }> = {
  ...unary<{ eps: number }>('rsqrt', (x, { eps }) => 1 / Math.sqrt(x + eps), (_x, y) => -0.5 * y ** 3, nonlinear('out')),
  defaults: { eps: 0 },
};
export const sqrtOp = unary('sqrt', Math.sqrt, (_x, y) => 0.5 / y, nonlinear('out'));
export const reciprocalOp = unary('reciprocal', (x) => 1 / x, (_x, y) => -y * y, nonlinear('out'));
export const sigmoidOp = unary('sigmoid', sigmoid, (_x, y) => y * (1 - y), nonlinear('out'));
export const SOFTPLUS_THRESHOLD = 20;
export const softplusOp = unary(
  'softplus',
  (x) => (x > SOFTPLUS_THRESHOLD ? x : Math.log1p(Math.exp(x))),
  (x) => (x > SOFTPLUS_THRESHOLD ? 1 : sigmoid(x)),
  nonlinear(0),
);
export const siluOp = unary('silu', (x) => x * sigmoid(x), (x) => sigmoid(x) * (1 + x * (1 - sigmoid(x))), nonlinear(0));

export function erf(x: number): number {
  const a = Math.abs(x);
  if (a >= 6) return Math.sign(x);
  let term = a;
  let sum = a;
  for (let n = 1; term > 1e-17 * sum; n++) {
    term *= (2 * a * a) / (2 * n + 1);
    sum += term;
  }
  return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * Math.exp(-a * a) * sum;
}

export const normalCdf = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));
export const normalPdf = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

export const reluOp = unary('relu', (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0), nonlinear('out'));
export const erfOp = unary('erf', erf, (x) => (2 / Math.sqrt(Math.PI)) * Math.exp(-x * x), nonlinear(0));
export const geluOp = unary('gelu', (x) => x * normalCdf(x), (x) => normalCdf(x) + x * normalPdf(x), nonlinear(0));
export const identityOp = unary('identity', (x) => x, () => 1, VIEW_COST);

function rotateHalf(input: Tensor, sign: number): Tensor {
  const width = input.shape[input.ndim - 1];
  const half = width / 2;
  const out = new Tensor(input.shape);
  for (let i = 0; i < input.size; i++) {
    out.data[i] = i % width < half ? -sign * input.data[i + half] : sign * input.data[i - half];
  }
  return out;
}

export const rotateHalfOp: OpDef<NoAttrs> = {
  name: 'rotate_half',
  args: ['input'],
  tensors: ['input'],
  shape: ([a]) => {
    if (a.length === 0 || a[a.length - 1] % 2 !== 0) throw new Error(`rotate_half: last dimension of (${a.join(', ')}) must be even`);
    return [...a];
  },
  forward: ([a]) => rotateHalf(a, 1),
  backward: (gradOut) => [rotateHalf(gradOut, -1)],
  cost: pointwise(1, 1, [[]]),
};

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
  cost: VIEW_COST,
};
