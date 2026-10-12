import type { Nested } from '../../tensor';
import type { Num } from './types';

type BufferOptions = Record<string, Num | boolean>;

interface BufferFunction {
  args: [number, number];
  options: Record<string, 'number' | 'bool'>;
  make(args: Num[], options: BufferOptions): { shape: Num[]; values: Nested };
}

const ROPE_OPTIONS: Record<string, 'number' | 'bool'> = {
  stride: 'number',
  interleaved: 'bool',
  half: 'bool',
  yarn_factor: 'number',
  yarn_original: 'number',
  beta_fast: 'number',
  beta_slow: 'number',
};

interface Yarn {
  factor: number;
  original: number;
  betaFast: number;
  betaSlow: number;
}

function yarnOf(options: BufferOptions): Yarn | undefined {
  const toy = (key: string, fallback: number) => (options[key] as Num | undefined)?.toy ?? fallback;
  const original = toy('yarn_original', 0);
  if (original <= 0) return undefined;
  return { factor: toy('yarn_factor', 1), original, betaFast: toy('beta_fast', 32), betaSlow: toy('beta_slow', 1) };
}

function rope(fn: (x: number) => number): BufferFunction {
  return {
    args: [3, 3],
    options: ROPE_OPTIONS,
    make: ([n, dh, theta], options) => {
      const half = options.half === true;
      const stride = (options.stride as Num | undefined)?.toy ?? 1;
      const freqs = ropeFrequencies(dh.toy, theta.toy, yarnOf(options));
      return {
        shape: half ? [n, { toy: dh.toy / 2, real: dh.real / 2 }] : [n, dh],
        values: half
          ? [...Array(n.toy).keys()].map((t) => freqs.map((f) => fn(t * stride * f)))
          : ropeTable(n.toy, dh.toy, freqs, fn, stride, options.interleaved === true),
      };
    },
  };
}

function rope2d(fn: (x: number) => number): BufferFunction {
  return {
    args: [4, 4],
    options: {},
    make: ([nh, nw, dim, theta]) => {
      const inv = [...Array(dim.toy / 2).keys()].map((i) => 1 / theta.toy ** ((2 * i) / dim.toy));
      const values: number[][] = [];
      for (let i = 0; i < nh.toy; i++) for (let j = 0; j < nw.toy; j++) values.push([...inv.map((f) => fn(i * f)), ...inv.map((f) => fn(j * f))]);
      return { shape: [{ toy: nh.toy * nw.toy, real: nh.real * nw.real }, dim], values };
    },
  };
}

export const BUFFER_FUNCTIONS: Record<string, BufferFunction> = {
  rope_cos: rope(Math.cos),
  rope_sin: rope(Math.sin),
  rope2d_cos: rope2d(Math.cos),
  rope2d_sin: rope2d(Math.sin),
  window_topk_idxs: {
    args: [2, 2],
    options: {},
    make: ([T, W]) => {
      const width = (n: number, w: number) => Math.min(n, w);
      return {
        shape: [T, { toy: width(T.toy, W.toy), real: width(T.real, W.real) }],
        values: [...Array(T.toy).keys()].map((t) =>
          [...Array(width(T.toy, W.toy)).keys()].map((k) => {
            const s = Math.max(t - W.toy + 1, 0) + k;
            return s > t ? -1 : s;
          }),
        ),
      };
    },
  },
  zeros: { args: [1, Infinity], options: {}, make: (shape) => ({ shape, values: filled(shape.map((n) => n.toy), 0) }) },
  ones: { args: [1, Infinity], options: {}, make: (shape) => ({ shape, values: filled(shape.map((n) => n.toy), 1) }) },
  full: { args: [2, Infinity], options: {}, make: (args) => ({ shape: args.slice(0, -1), values: filled(args.slice(0, -1).map((n) => n.toy), args[args.length - 1].toy) }) },
  arange: { args: [1, 1], options: {}, make: ([n]) => ({ shape: [n], values: [...Array(n.toy).keys()] }) },
  causal_mask: {
    args: [1, 2],
    options: {},
    make: ([T, window]) => ({
      shape: [T, T],
      values: grid(T.toy, T.toy, (t, s) => s > t || (window !== undefined && s <= t - window.toy)),
    }),
  },
};

function grid(rows: number, cols: number, masked: (r: number, c: number) => boolean): number[][] {
  return [...Array(rows).keys()].map((r) => [...Array(cols).keys()].map((c) => (masked(r, c) ? -Infinity : 0)));
}

function filled(shape: number[], value: number): Nested {
  return shape.length === 0 ? value : Array.from({ length: shape[0] }, () => filled(shape.slice(1), value));
}

function ropeFrequencies(dh: number, theta: number, yarn: Yarn | undefined): number[] {
  const inv = [...Array(dh / 2).keys()].map((i) => 1 / theta ** ((2 * i) / dh));
  if (yarn === undefined) return inv;
  const corrected = (rotations: number) => (dh * Math.log(yarn.original / (rotations * 2 * Math.PI))) / (2 * Math.log(theta));
  const low = Math.max(Math.floor(corrected(yarn.betaFast)), 0);
  const high = Math.min(Math.ceil(corrected(yarn.betaSlow)), dh - 1);
  return inv.map((f, i) => {
    const smooth = 1 - Math.min(1, Math.max(0, (i - low) / Math.max(high - low, 1e-3)));
    return (f / yarn.factor) * (1 - smooth) + f * smooth;
  });
}

function ropeTable(n: number, dh: number, inv: number[], fn: (x: number) => number, stride: number, interleaved: boolean): number[][] {
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
