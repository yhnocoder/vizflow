import type { Nested } from '../../tensor';
import type { Num } from './types';

type BufferArg = Num | Num[];

export const POSITION_LISTS: ReadonlySet<string> = new Set(['rope_cos', 'rope_sin']);

const positionsOf = (T: BufferArg): number[] => (Array.isArray(T) ? T.map((n) => n.toy) : [...Array(T.toy).keys()]);
const rowsOf = (T: BufferArg): Num => (Array.isArray(T) ? { toy: T.length, real: T.length } : T);

const ROPE_FUNCTIONS: Record<string, (args: BufferArg[]) => { shape: Num[]; values: Nested }> = {
  rope_cos: ([T, dh, theta]) => ({ shape: [rowsOf(T), dh as Num], values: ropeTable(positionsOf(T), (dh as Num).toy, (theta as Num).toy, Math.cos) }),
  rope_sin: ([T, dh, theta]) => ({ shape: [rowsOf(T), dh as Num], values: ropeTable(positionsOf(T), (dh as Num).toy, (theta as Num).toy, Math.sin) }),
};

const NUMBER_FUNCTIONS: Record<string, (args: Num[]) => { shape: Num[]; values: Nested }> = {
  zeros: (shape) => ({ shape, values: zerosOf(shape.map((n) => n.toy)) }),
  causal_mask: ([T]) => ({
    shape: [T, T],
    values: [...Array(T.toy).keys()].map((t) => [...Array(T.toy).keys()].map((s) => (s > t ? -Infinity : 0))),
  }),
};

export const BUFFER_ARITY: Record<string, number | 'any' | { lists: number; optional: number; usage: string }> = {
  rope_cos: 3,
  rope_sin: 3,
  zeros: 'any',
  causal_mask: 1,
  packed_mask: { lists: 1, optional: 2, usage: 'a list of new-token counts, then optionally a list of cached lengths and a window size' },
  packed_positions: { lists: 1, optional: 1, usage: 'a list of new-token counts, then optionally a list of cached lengths' },
  packed_segments: { lists: 1, optional: 0, usage: 'a list of new-token counts' },
};

interface Packed {
  seq: number;
  pos: number;
}

function packedLayout(args: BufferArg[]): { queries: Packed[]; keys: Packed[] } {
  const lens = (args[0] as Num[]).map((n) => n.toy);
  const cachedArg = args[1];
  if (cachedArg !== undefined && !Array.isArray(cachedArg)) throw new Error('the cached lengths must be a list');
  const cached = cachedArg === undefined ? lens.map(() => 0) : cachedArg.map((n) => n.toy);
  if (cached.length !== lens.length) throw new Error(`got ${lens.length} new-token counts but ${cached.length} cached lengths`);
  const counts = [...lens, ...cached];
  if (counts.some((n) => !Number.isInteger(n) || n < 0) || lens.some((n) => n === 0)) throw new Error('new-token counts must be positive integers and cached lengths non-negative integers');
  const queries: Packed[] = [];
  const keys: Packed[] = [];
  lens.forEach((n, seq) => {
    for (let j = 0; j < cached[seq] + n; j++) keys.push({ seq, pos: j });
    for (let j = 0; j < n; j++) queries.push({ seq, pos: cached[seq] + j });
  });
  return { queries, keys };
}

const fixed = (n: number): Num => ({ toy: n, real: n });

const PACKED_FUNCTIONS = {
  packed_mask(args: BufferArg[]): { shape: Num[]; values: Nested } {
    const { queries, keys } = packedLayout(args);
    const windowArg = args[2];
    if (windowArg !== undefined && (Array.isArray(windowArg) || !Number.isInteger(windowArg.toy) || windowArg.toy < 1)) throw new Error('the window size must be a positive integer');
    const window = windowArg === undefined ? Infinity : (windowArg as Num).toy;
    const values = queries.map((q) => keys.map((k) => (k.seq === q.seq && k.pos <= q.pos && q.pos - k.pos < window ? 0 : -Infinity)));
    return { shape: [fixed(queries.length), fixed(keys.length)], values };
  },
  packed_positions(args: BufferArg[]): { shape: Num[]; values: Nested } {
    const { queries } = packedLayout(args);
    return { shape: [fixed(queries.length)], values: queries.map((q) => q.pos) };
  },
  packed_segments(args: BufferArg[]): { shape: Num[]; values: Nested } {
    const { queries } = packedLayout(args);
    return { shape: [fixed(queries.length)], values: queries.map((q) => q.seq) };
  },
};

export const BUFFER_FUNCTIONS: Record<string, (args: BufferArg[]) => { shape: Num[]; values: Nested }> = {
  ...(NUMBER_FUNCTIONS as Record<string, (args: BufferArg[]) => { shape: Num[]; values: Nested }>),
  ...ROPE_FUNCTIONS,
  ...PACKED_FUNCTIONS,
};

function zerosOf(shape: number[]): Nested {
  return shape.length === 0 ? 0 : Array.from({ length: shape[0] }, () => zerosOf(shape.slice(1)));
}

function ropeTable(positions: number[], dh: number, theta: number, fn: (x: number) => number): number[][] {
  const inv = [...Array(dh / 2).keys()].map((i) => 1 / theta ** ((2 * i) / dh));
  return positions.map((t) => [...Array(dh).keys()].map((c) => fn(t * inv[c % (dh / 2)])));
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
