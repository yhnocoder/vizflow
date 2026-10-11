export const DTYPES = ['bool', 'int4', 'uint8', 'int8', 'int16', 'int32', 'int64', 'fp4_e2m1', 'e8m0', 'fp8_e4m3', 'fp8_e5m2', 'fp16', 'bf16', 'fp32', 'fp64'] as const;
export type Dtype = (typeof DTYPES)[number];

export const DTYPE_BITS: Readonly<Record<Dtype, number>> = {
  bool: 8,
  int4: 4,
  uint8: 8,
  int8: 8,
  int16: 16,
  int32: 32,
  int64: 64,
  fp4_e2m1: 4,
  e8m0: 8,
  fp8_e4m3: 8,
  fp8_e5m2: 8,
  fp16: 16,
  bf16: 16,
  fp32: 32,
  fp64: 64,
};

export const DEFAULT_DTYPE: Dtype = 'fp32';
export const LABEL_DTYPE: Dtype = 'int64';

const FLOAT_RANK: Partial<Record<Dtype, number>> = { fp4_e2m1: 1, e8m0: 1, fp8_e4m3: 2, fp8_e5m2: 2, fp16: 3, bf16: 3, fp32: 4, fp64: 5 };
const FP8_RANK = 2;
const INT_RANK: Partial<Record<Dtype, number>> = { bool: 0, int4: 1, uint8: 2, int8: 2, int16: 3, int32: 4, int64: 5 };

export function checkDtype(dtype: string, where: string): Dtype {
  if (!(DTYPES as readonly string[]).includes(dtype)) throw new Error(`${where}: unknown dtype '${dtype}', expected one of ${DTYPES.join(', ')}`);
  return dtype as Dtype;
}

function widest(dtypes: readonly Dtype[], ranks: Partial<Record<Dtype, number>>, where: string): Dtype {
  const top = Math.max(...dtypes.map((d) => ranks[d]!));
  const tied = new Set(dtypes.filter((d) => ranks[d] === top));
  if (tied.size === 1) return [...tied][0];
  if (ranks === FLOAT_RANK && top <= FP8_RANK) {
    const names = DTYPES.filter((d) => tied.has(d));
    throw new Error(`${where}: cannot promote ${names.join(' with ')}, give the dtype explicitly`);
  }
  return Object.entries(ranks).find(([, rank]) => rank === top + 1)![0] as Dtype;
}

export function promoteDtypes(dtypes: readonly Dtype[], fallback: Dtype, where: string): Dtype {
  if (dtypes.length === 0) return fallback;
  const floats = dtypes.filter((d) => FLOAT_RANK[d] !== undefined);
  return floats.length > 0 ? widest(floats, FLOAT_RANK, where) : widest(dtypes, INT_RANK, where);
}
