import type { Nested } from '../../tensor';
import type { Num } from './types';

type BufferOptions = Record<string, Num | boolean>;

interface BufferFunction {
  args: [number, number];
  options: Record<string, 'number' | 'bool'>;
  make(args: Num[], options: BufferOptions): { shape: Num[]; values: Nested };
}

const ROPE_OPTIONS: Record<string, 'number' | 'bool'> = { stride: 'number', interleaved: 'bool' };

function rope(fn: (x: number) => number): BufferFunction {
  return {
    args: [3, 3],
    options: ROPE_OPTIONS,
    make: ([n, dh, theta], options) => ({
      shape: [n, dh],
      values: ropeTable(n.toy, dh.toy, theta.toy, fn, (options.stride as Num | undefined)?.toy ?? 1, options.interleaved === true),
    }),
  };
}

export const BUFFER_FUNCTIONS: Record<string, BufferFunction> = {
  rope_cos: rope(Math.cos),
  rope_sin: rope(Math.sin),
  zeros: { args: [1, Infinity], options: {}, make: (shape) => ({ shape, values: filled(shape.map((n) => n.toy), 0) }) },
  full: { args: [2, Infinity], options: {}, make: ([value, ...shape]) => ({ shape, values: filled(shape.map((n) => n.toy), value.toy) }) },
  causal_mask: {
    args: [1, 2],
    options: {},
    make: ([T, window]) => ({
      shape: [T, T],
      values: grid(T.toy, T.toy, (t, s) => s > t || (window !== undefined && s <= t - window.toy)),
    }),
  },
  block_causal_mask: {
    args: [2, 2],
    options: {},
    make: ([T, m]) => {
      const blocks = { toy: Math.floor(T.toy / m.toy), real: Math.floor(T.real / m.real) };
      return { shape: [T, blocks], values: grid(T.toy, blocks.toy, (t, w) => (w + 1) * m.toy > t + 1) };
    },
  },
};

function grid(rows: number, cols: number, masked: (r: number, c: number) => boolean): number[][] {
  return [...Array(rows).keys()].map((r) => [...Array(cols).keys()].map((c) => (masked(r, c) ? -Infinity : 0)));
}

function filled(shape: number[], value: number): Nested {
  return shape.length === 0 ? value : Array.from({ length: shape[0] }, () => filled(shape.slice(1), value));
}

function ropeTable(n: number, dh: number, theta: number, fn: (x: number) => number, stride: number, interleaved: boolean): number[][] {
  const inv = [...Array(dh / 2).keys()].map((i) => 1 / theta ** ((2 * i) / dh));
  const pair = (c: number) => (interleaved ? Math.floor(c / 2) : c % (dh / 2));
  return [...Array(n).keys()].map((t) => [...Array(dh).keys()].map((c) => fn(t * stride * inv[pair(c)])));
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
