import type { Nested } from '../../tensor';
import type { Num } from './types';

export const BUFFER_FUNCTIONS: Record<string, (args: Num[]) => { shape: Num[]; values: Nested }> = {
  rope_cos: ([T, dh, theta]) => ({ shape: [T, dh], values: ropeTable(T.toy, dh.toy, theta.toy, Math.cos) }),
  rope_sin: ([T, dh, theta]) => ({ shape: [T, dh], values: ropeTable(T.toy, dh.toy, theta.toy, Math.sin) }),
  causal_mask: ([T]) => ({
    shape: [T, T],
    values: [...Array(T.toy).keys()].map((t) => [...Array(T.toy).keys()].map((s) => (s > t ? -Infinity : 0))),
  }),
};

export const BUFFER_ARITY: Record<string, number> = { rope_cos: 3, rope_sin: 3, causal_mask: 1 };

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
