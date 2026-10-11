import { FLOAT_FORMATS, type FloatFormat, exponentOf, roundHalfEven, roundToFormat, roundUpToPowerOfTwo } from './rounding';

export const QUANT_FORMATS = ['fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1', 'int8', 'int4'] as const;
export type QuantFormat = (typeof QUANT_FORMATS)[number];
export const SCALE_RULES = ['amax', 'ue8m0', 'mx', 'e4m3'] as const;
export type ScaleRule = (typeof SCALE_RULES)[number];

const AMAX_FLOOR = 1e-4;
const E4M3_MIN_SUBNORMAL = 2 ** -9;
const INT_MAX: Partial<Record<QuantFormat, number>> = { int8: 127, int4: 7 };

function floatFormat(fmt: QuantFormat): FloatFormat | undefined {
  return fmt === 'fp8_e4m3' || fmt === 'fp8_e5m2' || fmt === 'fp4_e2m1' ? FLOAT_FORMATS[fmt] : undefined;
}

export function quantMax(fmt: QuantFormat): number {
  return floatFormat(fmt)?.max ?? INT_MAX[fmt]!;
}

export function roundTo(fmt: QuantFormat, v: number): number {
  const float = floatFormat(fmt);
  if (float !== undefined) return roundToFormat(Math.min(Math.max(v, -float.max), float.max), float) + 0;
  if (Number.isNaN(v)) return v;
  const qmax = INT_MAX[fmt]!;
  return Math.min(Math.max(roundHalfEven(v), -qmax), qmax) + 0;
}

export function scaleOf(fmt: QuantFormat, rule: ScaleRule, amax: number): number {
  const qmax = quantMax(fmt);
  switch (rule) {
    case 'amax':
      return Math.fround(Math.max(Math.fround(amax), Math.fround(AMAX_FLOOR)) / qmax);
    case 'ue8m0':
      return roundUpToPowerOfTwo(scaleOf(fmt, 'amax', amax));
    case 'mx':
      if (floatFormat(fmt) === undefined) throw new Error(`block_scale: rule 'mx' needs a float format, got '${fmt}'`);
      return amax === 0 ? 1 : 2 ** (exponentOf(amax) - exponentOf(qmax));
    case 'e4m3':
      if (amax === 0) return 1;
      return roundTo('fp8_e4m3', Math.fround(amax / qmax)) || E4M3_MIN_SUBNORMAL;
  }
}
