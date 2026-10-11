import type { Dtype } from './dtype';

export type Overflow = 'inf' | 'saturate';

export interface FloatFormat {
  name: string;
  mantissa: number;
  minExponent: number;
  max: number;
  overflow: Overflow;
}

function format(name: string, exponentBits: number, mantissa: number, overflow: Overflow, maxMantissa?: number): FloatFormat {
  const bias = 2 ** (exponentBits - 1) - 1;
  const top = overflow === 'inf' ? 2 ** exponentBits - 2 - bias : 2 ** exponentBits - 1 - bias;
  const fraction = maxMantissa ?? 2 ** mantissa - 1;
  return { name, mantissa, minExponent: 1 - bias, max: (1 + fraction / 2 ** mantissa) * 2 ** top, overflow };
}

export const FLOAT_FORMATS = {
  fp32: format('fp32', 8, 23, 'inf'),
  bf16: format('bf16', 8, 7, 'inf'),
  fp16: format('fp16', 5, 10, 'inf'),
  fp8_e5m2: format('fp8_e5m2', 5, 2, 'inf'),
  fp8_e4m3: format('fp8_e4m3', 4, 3, 'saturate', 6),
  fp4_e2m1: format('fp4_e2m1', 2, 1, 'saturate'),
} as const satisfies Record<string, FloatFormat>;

export type FloatFormatName = keyof typeof FLOAT_FORMATS;

export function exponentOf(a: number): number {
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  else if (2 ** (e + 1) <= a) e++;
  return e;
}

function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function roundToFormat(x: number, f: FloatFormat): number {
  if (Number.isNaN(x) || x === 0 || !Number.isFinite(x)) {
    if (!Number.isFinite(x) && !Number.isNaN(x) && f.overflow === 'saturate') return Math.sign(x) * f.max;
    return x;
  }
  const a = Math.abs(x);
  const quantum = 2 ** (Math.max(exponentOf(a), f.minExponent) - f.mantissa);
  const r = roundHalfEven(a / quantum) * quantum;
  if (r > f.max) {
    return Math.sign(x) * (f.overflow === 'saturate' ? f.max : Infinity);
  }
  return r === 0 ? Math.sign(x) * 0 : Math.sign(x) * r;
}

export function roundToDtype(x: number, dtype: Dtype): number {
  if (dtype === 'fp64') return x;
  if (dtype === 'fp32') return Math.fround(x);
  if (dtype === 'bf16' || dtype === 'fp16' || dtype === 'fp8_e4m3' || dtype === 'fp8_e5m2' || dtype === 'fp4_e2m1') return roundToFormat(Math.fround(x), FLOAT_FORMATS[dtype]);
  if (dtype === 'e8m0') return roundToE8m0(x);
  if (dtype === 'bool') return x !== 0 ? 1 : 0;
  if (dtype === 'int4') return Math.min(Math.max(Math.trunc(x), INT4_MIN), INT4_MAX) + 0;
  return Math.trunc(x);
}

const INT4_MIN = -8;
const INT4_MAX = 7;
const E8M0_MIN_EXPONENT = -127;
const E8M0_MAX_EXPONENT = 127;

function roundToE8m0(x: number): number {
  const a = Math.abs(Math.fround(x));
  if (!Number.isFinite(a)) return NaN;
  if (a < 2 ** E8M0_MIN_EXPONENT) return 2 ** E8M0_MIN_EXPONENT;
  const e = exponentOf(a);
  const up = a / 2 ** e >= 1.5 ? e + 1 : e;
  return up > E8M0_MAX_EXPONENT ? NaN : 2 ** up;
}

export function roundUpToPowerOfTwo(x: number): number {
  if (!(x > 0) || !Number.isFinite(x)) throw new Error(`roundUpToPowerOfTwo: expected a positive finite number, got ${x}`);
  const e = exponentOf(x);
  return 2 ** e === x ? x : 2 ** (e + 1);
}
