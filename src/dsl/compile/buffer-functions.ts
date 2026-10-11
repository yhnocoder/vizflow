import type { Nested } from '../../tensor';
import type { Num } from './types';

export const BUFFER_FUNCTIONS: Record<string, (args: Num[]) => { shape: Num[]; values: Nested }> = {
  rope_cos: ([T, dh, theta]) => ({ shape: [T, dh], values: ropeTable(T.toy, dh.toy, theta.toy, Math.cos) }),
  rope_sin: ([T, dh, theta]) => ({ shape: [T, dh], values: ropeTable(T.toy, dh.toy, theta.toy, Math.sin) }),
  zeros: (shape) => ({ shape, values: zerosOf(shape.map((n) => n.toy)) }),
  full: (args) => ({ shape: args.slice(0, -1), values: fullOf(args.slice(0, -1).map((n) => n.toy), args[args.length - 1].toy) }),
  arange: ([n]) => ({ shape: [n], values: [...Array(n.toy).keys()] }),
  causal_mask: ([T]) => ({
    shape: [T, T],
    values: [...Array(T.toy).keys()].map((t) => [...Array(T.toy).keys()].map((s) => (s > t ? -Infinity : 0))),
  }),
};

export const BUFFER_ARITY: Record<string, number | { min: number }> = { rope_cos: 3, rope_sin: 3, zeros: { min: 1 }, full: { min: 2 }, arange: 1, causal_mask: 1 };

function fullOf(shape: number[], value: number): Nested {
  return shape.length === 0 ? value : Array.from({ length: shape[0] }, () => fullOf(shape.slice(1), value));
}

function zerosOf(shape: number[]): Nested {
  return fullOf(shape, 0);
}

function ropeTable(T: number, dh: number, theta: number, fn: (x: number) => number): number[][] {
  const inv = [...Array(dh / 2).keys()].map((i) => 1 / theta ** ((2 * i) / dh));
  return [...Array(T).keys()].map((t) => [...Array(dh).keys()].map((c) => fn(t * inv[c % (dh / 2)])));
}

export function reshapeNested(values: Nested, shape: number[]): Nested {
  const flat: number[] = [];
  const collect = (v: Nested): void => {
    if (Array.isArray(v)) v.forEach(collect);
    else flat.push(v as number);
  };
  collect(values);
  const nest = (offset: number, dims: number[]): Nested => {
    if (dims.length === 0) return flat[offset];
    const stride = dims.slice(1).reduce((a, b) => a * b, 1);
    return Array.from({ length: dims[0] }, (_, i) => nest(offset + i * stride, dims.slice(1)));
  };
  return nest(0, shape);
}
