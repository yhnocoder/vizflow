export const DTYPES = ['bool', 'uint8', 'int8', 'int16', 'int32', 'int64', 'fp8_e4m3', 'fp8_e5m2', 'fp16', 'bf16', 'fp32', 'fp64'] as const;
export type Dtype = (typeof DTYPES)[number];

export const DTYPE_BYTES: Readonly<Record<Dtype, number>> = {
  bool: 1,
  uint8: 1,
  int8: 1,
  int16: 2,
  int32: 4,
  int64: 8,
  fp8_e4m3: 1,
  fp8_e5m2: 1,
  fp16: 2,
  bf16: 2,
  fp32: 4,
  fp64: 8,
};

export const DEFAULT_DTYPE: Dtype = 'fp32';
export const LABEL_DTYPE: Dtype = 'int64';

const FLOAT_RANK: Partial<Record<Dtype, number>> = { fp8_e4m3: 1, fp8_e5m2: 1, fp16: 2, bf16: 2, fp32: 3, fp64: 4 };
const INT_RANK: Partial<Record<Dtype, number>> = { bool: 0, uint8: 1, int8: 1, int16: 2, int32: 3, int64: 4 };

export function checkDtype(dtype: string, where: string): Dtype {
  if (!(DTYPES as readonly string[]).includes(dtype)) throw new Error(`${where}: unknown dtype '${dtype}', expected one of ${DTYPES.join(', ')}`);
  return dtype as Dtype;
}

function widest(dtypes: readonly Dtype[], ranks: Partial<Record<Dtype, number>>, where: string): Dtype {
  const top = Math.max(...dtypes.map((d) => ranks[d]!));
  const tied = new Set(dtypes.filter((d) => ranks[d] === top));
  if (tied.size === 1) return [...tied][0];
  if (tied.has('fp8_e4m3') && tied.has('fp8_e5m2')) throw new Error(`${where}: cannot promote fp8_e4m3 with fp8_e5m2, give the dtype explicitly`);
  return Object.entries(ranks).find(([, rank]) => rank === top + 1)![0] as Dtype;
}

export function promoteDtypes(dtypes: readonly Dtype[], fallback: Dtype, where: string): Dtype {
  if (dtypes.length === 0) return fallback;
  const floats = dtypes.filter((d) => FLOAT_RANK[d] !== undefined);
  return floats.length > 0 ? widest(floats, FLOAT_RANK, where) : widest(dtypes, INT_RANK, where);
}
