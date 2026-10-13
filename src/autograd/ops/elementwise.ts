import { Tensor, broadcastIndex, broadcastShapes, forEachIndex, formatShape, normDim, offsetOf, unbroadcast } from '../../tensor';
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

function whereShape([condition, input, other]: number[][]): number[] {
  const kept = broadcastShapes(condition, input);
  return other === undefined ? kept : broadcastShapes(kept, other);
}

function whereValue(shape: number[], [condition, input, other]: Tensor[], fallback: number | undefined): Tensor {
  const out = new Tensor(shape);
  forEachIndex(shape, (index, offset) => {
    const keep = condition.data[offsetOf(condition.shape, broadcastIndex(index, condition.shape))] !== 0;
    if (keep) out.data[offset] = input.data[offsetOf(input.shape, broadcastIndex(index, input.shape))];
    else out.data[offset] = other === undefined ? (fallback as number) : other.data[offsetOf(other.shape, broadcastIndex(index, other.shape))];
  });
  return out;
}

export const whereOp: OpDef<{ other?: number }> = {
  name: 'where',
  args: ['condition', 'input', 'other'],
  tensors: ['condition', 'input', 'other'],
  scalars: ['other'],
  differentiable: [false, true, true],
  shape: (shapes) => whereShape(shapes),
  forward: (inputs, { other }) => whereValue(whereShape(inputs.map((t) => [...t.shape])), inputs, other),
  backward: (gradOut, [condition, input, other]) => {
    const keep = zipWith(condition, gradOut, (c, g) => (c !== 0 ? g : 0));
    const grads: (Tensor | null)[] = [null, unbroadcast(keep, input.shape)];
    if (other !== undefined) grads.push(unbroadcast(zipWith(condition, gradOut, (c, g) => (c !== 0 ? 0 : g)), other.shape));
    return grads;
  },
  cost: movement(false, [[], [0], [0]]),
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

function integerOperands(a: Tensor, b: Tensor | undefined, other: number | undefined): boolean {
  const integral = (t: Tensor) => t.int64 !== undefined || t.data.every(Number.isInteger);
  return integral(a) && (b === undefined ? Number.isInteger(other) : integral(b));
}

function exactZip(a: Tensor, b: Tensor | undefined, other: number | undefined, fn: (x: bigint, y: bigint) => bigint): Tensor {
  const shape = b === undefined ? [...a.shape] : broadcastShapes(a.shape, b.shape);
  const values: bigint[] = [];
  forEachIndex(shape, (index) => {
    const x = a.exact(offsetOf(a.shape, broadcastIndex(index, a.shape)));
    const y = b === undefined ? BigInt(other as number) : b.exact(offsetOf(b.shape, broadcastIndex(index, b.shape)));
    values.push(BigInt.asIntN(64, fn(x, y)));
  });
  return Tensor.fromInt64(shape, values);
}

function withInt64(def: OpDef<ScalarAttrs>, exact: (x: bigint, y: bigint) => bigint): OpDef<ScalarAttrs> {
  return {
    ...def,
    int64: true,
    forward: (inputs, attrs) => {
      const [a, b] = inputs;
      const int64 = a.int64 !== undefined || b?.int64 !== undefined;
      if (int64 && integerOperands(a, b, attrs.other)) return exactZip(a, b, attrs.other, exact);
      const unsafe = inputs.find((t) => t.beyondExact());
      if (unsafe !== undefined) throw new Error(`${def.name}: int64 values beyond 2^53 can only combine with integers`);
      return def.forward(inputs, attrs);
    },
  };
}

function floorMod(x: number, y: number): number {
  return x - Math.floor(x / y) * y;
}

function floorModExact(x: bigint, y: bigint): bigint {
  if (y === 0n) throw new Error('remainder: integer division by zero');
  const r = x % y;
  return r !== 0n && r < 0n !== y < 0n ? r + y : r;
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
export const rsubOp = binary('rsub', (x, y) => y - x, () => [-1, 1], additive([true, false]));
export const mulOp = withInt64(binary('mul', (x, y) => x * y, (x, y) => [y, x], pointwise(1, 1, [[1], [0]])), (x, y) => x * y);
export const remainderOp = withInt64(binary('remainder', floorMod, (x, y) => [1, -Math.floor(x / y)], pointwise(1, 1, [[], [0, 1]])), floorModExact);
export const divOp = binary('div', (x, y) => x / y, (x, y) => [1 / y, -x / (y * y)], {
  saves: [[1], [0, 1]],
  flops: (inputs, out) => ({ forward: [numel(out)], backward: inputs.map((_, k) => (k === 0 ? [numel(out)] : [DIV_OTHER_BACKWARD, numel(out)])) }),
});
export const negOp = unary('neg', (x) => -x, () => -1, pointwise(1, 1, [[]]));
export const expOp = unary('exp', Math.exp, (_x, y) => y, nonlinear('out'));
export const logOp = unary('log', Math.log, (x) => 1 / x, nonlinear(0));
export const cosOp = unary('cos', Math.cos, (x) => -Math.sin(x), nonlinear(0));
export const sinOp = unary('sin', Math.sin, (x) => Math.cos(x), nonlinear(0));

const COMPARISON_COST: OpCost = pointwise(1, 0, [[], []]);

function integerArgs(name: string, a: Tensor, b: Tensor | undefined, other: number | undefined): void {
  if (!integerOperands(a, b, other)) throw new Error(`${name}: inputs must be integers or bools`);
}

function bitwise(name: string, fn: (x: bigint, y: bigint) => bigint): OpDef<ScalarAttrs> {
  return {
    name,
    args: ['input', 'other'],
    tensors: ['input', 'other'],
    scalars: ['other'],
    differentiable: [false, false],
    int64: true,
    cost: COMPARISON_COST,
    shape: ([a, b]) => (b === undefined ? [...a] : broadcastShapes(a, b)),
    forward: ([a, b], { other }) => {
      integerArgs(name, a, b, other);
      const out = exactZip(a, b, other, fn);
      return a.int64 !== undefined || b?.int64 !== undefined ? out : new Tensor(out.shape, out.data);
    },
    backward: () => [null, null],
  };
}

export const bitwiseXorOp = bitwise('bitwise_xor', (x, y) => x ^ y);
export const bitwiseAndOp = bitwise('bitwise_and', (x, y) => x & y);
export const bitwiseOrOp = bitwise('bitwise_or', (x, y) => x | y);

export function signBit(y: number): number {
  return y < 0 || Object.is(y, -0) ? -1 : 1;
}

export function copysignLocal(x: number, y: number): number {
  return x === 0 ? 0 : Math.sign(x) * signBit(y);
}

export const copysignOp: OpDef<ScalarAttrs> = {
  name: 'copysign',
  args: ['input', 'other'],
  tensors: ['input', 'other'],
  scalars: ['other'],
  differentiable: [true, false],
  cost: {
    saves: [[0, 1], []],
    flops: (inputs, out) => ({ forward: [numel(out)], backward: inputs.map((_, k) => (k === 0 ? [numel(out)] : NO_FLOPS)) }),
  },
  shape: ([a, b]) => (b === undefined ? [...a] : broadcastShapes(a, b)),
  forward: ([a, b], { other }) =>
    b === undefined ? mapWith(a, (x) => Math.abs(x) * signBit(other as number)) : zipWith(a, b, (x, y) => Math.abs(x) * signBit(y)),
  backward: (gradOut, [a, b], _out, { other }) => {
    const local = b === undefined ? mapWith(a, (x) => copysignLocal(x, other as number)) : zipWith(a, b, copysignLocal);
    return [unbroadcast(zipWith(local, gradOut, (l, g) => l * g), a.shape), null];
  },
};

function compare(name: string, fn: (x: number, y: number) => boolean): OpDef<ScalarAttrs> {
  return {
    name,
    args: ['input', 'other'],
    tensors: ['input', 'other'],
    scalars: ['other'],
    differentiable: [false, false],
    dtype: 'bool',
    cost: COMPARISON_COST,
    shape: ([a, b]) => (b === undefined ? [...a] : broadcastShapes(a, b)),
    forward: ([a, b], { other }) =>
      b === undefined ? mapWith(a, (x) => (fn(x, other as number) ? 1 : 0)) : zipWith(a, b, (x, y) => (fn(x, y) ? 1 : 0)),
    backward: () => [null, null],
  };
}

export const logicalNotOp: OpDef<NoAttrs> = {
  name: 'logical_not',
  args: ['input'],
  tensors: ['input'],
  differentiable: [false],
  dtype: 'bool',
  cost: COMPARISON_COST,
  shape: ([a]) => [...a],
  forward: ([a]) => mapWith(a, (x) => (x === 0 ? 1 : 0)),
  backward: () => [null],
};
export const eqOp = compare('eq', (x, y) => x === y);
export const geOp = compare('ge', (x, y) => x >= y);
export const ltOp = compare('lt', (x, y) => x < y);
export const greaterOp = compare('greater', (x, y) => x > y);

function floorDivExact(x: bigint, y: bigint): bigint {
  if (y === 0n) throw new Error('floor_divide: integer division by zero');
  const q = x / y;
  return x % y !== 0n && x < 0n !== y < 0n ? q - 1n : q;
}

export const floorDivideOp = withInt64(
  { ...binary('floor_divide', (x, y) => Math.floor(x / y), () => [0, 0], COMPARISON_COST), differentiable: [false, false], backward: () => [null, null] },
  floorDivExact,
);
export interface PowAttrs {
  input?: number;
  exponent?: number;
}

export function powOperands<T>(inputs: readonly T[], attrs: PowAttrs): { base: T | number; exponent: T | number } {
  let k = 0;
  const base = attrs.input ?? inputs[k++];
  const exponent = attrs.exponent ?? inputs[k++];
  return { base, exponent };
}

const asTensor = (value: Tensor | number) => (typeof value === 'number' ? new Tensor([], [value]) : value);

export const powOp: OpDef<PowAttrs> = {
  name: 'pow',
  args: ['input', 'exponent'],
  tensors: ['input', 'exponent'],
  scalars: ['input', 'exponent'],
  cost: pointwise(NONLINEAR_FORWARD, NONLINEAR_BACKWARD, [[0, 1], [0, 1]]),
  shape: (shapes, attrs) => {
    const { base, exponent } = powOperands(shapes, attrs);
    if (typeof base === 'number' && typeof exponent === 'number') throw new Error('pow: needs a tensor as the base or the exponent');
    return typeof base === 'number' ? [...(exponent as number[])] : typeof exponent === 'number' ? [...base] : broadcastShapes(base, exponent);
  },
  forward: (inputs, attrs) => {
    const { base, exponent } = powOperands(inputs, attrs);
    return zipWith(asTensor(base), asTensor(exponent), (x, y) => x ** y);
  },
  backward: (gradOut, inputs, _out, attrs) => {
    const { base, exponent } = powOperands(inputs, attrs);
    const b = asTensor(base);
    const e = asTensor(exponent);
    const grads: Tensor[] = [];
    const scaled = (local: Tensor, shape: readonly number[]) => unbroadcast(zipWith(local, gradOut, (l, g) => l * g), shape);
    if (typeof base !== 'number') grads.push(scaled(zipWith(b, e, (x, y) => y * x ** (y - 1)), base.shape));
    if (typeof exponent !== 'number') grads.push(scaled(zipWith(b, e, (x, y) => x ** y * Math.log(x)), exponent.shape));
    return grads;
  },
};
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
export const absOp = unary('abs', Math.abs, (x) => Math.sign(x), pointwise(1, 1, [[0]]));
export const signOp = unary('sign', Math.sign, () => 0, pointwise(1, 0, [[]]));
export const clampMinOp = unary<{ min: number }>('clamp_min', (x, { min }) => Math.max(x, min), (x, _y, { min }) => (x > min ? 1 : 0), pointwise(1, 1, [[0]]), ['input', 'min']);

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

export interface ClampAttrs {
  min: number | null;
  max: number | null;
}

export function clampInside(x: number, { min, max }: ClampAttrs): boolean {
  return (min === null || x > min) && (max === null || x < max);
}

export const clampOp: OpDef<ClampAttrs> = {
  name: 'clamp',
  args: ['input', 'min', 'max'],
  tensors: ['input'],
  defaults: { min: null, max: null },
  shape: ([a], { min, max }) => {
    if (min === null && max === null) throw new Error('clamp: at least one of min and max must be given');
    if (min !== null && max !== null && min > max) throw new Error(`clamp: min ${min} is larger than max ${max}`);
    return [...a];
  },
  forward: ([a], attrs) => mapWith(a, (x) => Math.min(attrs.max ?? Infinity, Math.max(attrs.min ?? -Infinity, x))),
  backward: (gradOut, [a], _out, attrs) => [new Tensor(a.shape, gradOut.data.map((g, i) => (clampInside(a.data[i], attrs) ? g : 0)))],
  cost: nonlinear(0),
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

export interface TriuAttrs {
  diagonal: number;
}

export function triuKeeps(index: readonly number[], attrs: TriuAttrs): boolean {
  return index[index.length - 1] - index[index.length - 2] >= attrs.diagonal;
}

function triuOf(input: Tensor, attrs: TriuAttrs): Tensor {
  const out = new Tensor(input.shape);
  forEachIndex(input.shape, (index, offset) => {
    if (triuKeeps(index, attrs)) out.data[offset] = input.data[offset];
  });
  return out;
}

export const triuOp: OpDef<TriuAttrs> = {
  name: 'triu',
  args: ['input', 'diagonal'],
  tensors: ['input'],
  defaults: { diagonal: 0 },
  shape: ([a]) => {
    if (a.length < 2) throw new Error(`triu: input ${formatShape(a)} must have at least two dimensions`);
    return [...a];
  },
  forward: ([a], attrs) => triuOf(a, attrs),
  backward: (gradOut, _inputs, _out, attrs) => [triuOf(gradOut, attrs)],
  cost: movement(false, [[]]),
};

function maskedShape(input: readonly number[], mask: readonly number[]): number[] {
  const shape = broadcastShapes(input, mask);
  if (shape.length !== input.length || shape.some((size, d) => size !== input[d])) {
    throw new Error(`masked_fill: mask ${formatShape(mask)} must broadcast to input ${formatShape(input)}`);
  }
  return [...input];
}

export const maskedFillOp: OpDef<{ value: number }> = {
  name: 'masked_fill',
  args: ['input', 'mask', 'value'],
  tensors: ['input', 'mask'],
  differentiable: [true, false],
  shape: ([input, mask]) => maskedShape(input, mask),
  forward: ([input, mask], { value }) => zipWith(input, mask, (x, fill) => (fill !== 0 ? value : x)),
  backward: (gradOut, [, mask]) => [zipWith(gradOut, mask, (g, fill) => (fill !== 0 ? 0 : g)), null],
  cost: movement(false, [[1], []]),
};
