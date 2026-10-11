export const DTYPES = ['bool', 'uint8', 'int8', 'int16', 'int32', 'int64', 'fp8_e4m3', 'fp8_e5m2', 'fp16', 'bf16', 'fp32', 'fp64'] as const;
export type Dtype = (typeof DTYPES)[number];

export const DEFAULT_DTYPE: Dtype = 'fp32';
export const LABEL_DTYPE: Dtype = 'int64';

const FLOAT_RANK: Partial<Record<Dtype, number>> = { fp8_e4m3: 1, fp8_e5m2: 1, fp16: 2, bf16: 2, fp32: 3, fp64: 4 };

export function checkDtype(dtype: string, where: string): Dtype {
  if (!(DTYPES as readonly string[]).includes(dtype)) throw new Error(`${where}: unknown dtype '${dtype}', expected one of ${DTYPES.join(', ')}`);
  return dtype as Dtype;
}

export function promoteDtypes(dtypes: readonly Dtype[], fallback: Dtype): Dtype {
  const floats = dtypes.filter((d) => FLOAT_RANK[d] !== undefined);
  if (floats.length === 0) return dtypes[0] ?? fallback;
  const top = Math.max(...floats.map((d) => FLOAT_RANK[d]!));
  const widest = new Set(floats.filter((d) => FLOAT_RANK[d] === top));
  if (widest.size === 1) return [...widest][0];
  return Object.entries(FLOAT_RANK).find(([, rank]) => rank === top + 1)![0] as Dtype;
}
