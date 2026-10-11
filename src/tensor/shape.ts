export type Shape = readonly number[];

export function numel(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

export function stridesOf(shape: Shape): number[] {
  const strides = new Array<number>(shape.length);
  let step = 1;
  for (let d = shape.length - 1; d >= 0; d--) {
    strides[d] = step;
    step *= shape[d];
  }
  return strides;
}

export function offsetOf(shape: Shape, index: readonly number[]): number {
  if (index.length !== shape.length) {
    throw new Error(`index ${formatShape(index)} does not match shape ${formatShape(shape)}`);
  }
  let offset = 0;
  for (let d = 0; d < shape.length; d++) {
    const i = index[d];
    if (!Number.isInteger(i) || i < 0 || i >= shape[d]) {
      throw new Error(`index ${formatShape(index)} is out of range for shape ${formatShape(shape)}`);
    }
    offset = offset * shape[d] + i;
  }
  return offset;
}

export function indexOf(shape: Shape, offset: number): number[] {
  const index = new Array<number>(shape.length);
  let rest = offset;
  for (let d = shape.length - 1; d >= 0; d--) {
    index[d] = rest % shape[d];
    rest = Math.floor(rest / shape[d]);
  }
  return index;
}

export function forEachIndex(shape: Shape, fn: (index: number[], offset: number) => void): void {
  const total = numel(shape);
  const index = new Array<number>(shape.length).fill(0);
  for (let offset = 0; offset < total; offset++) {
    fn(index.slice(), offset);
    for (let d = shape.length - 1; d >= 0; d--) {
      index[d]++;
      if (index[d] < shape[d]) break;
      index[d] = 0;
    }
  }
}

export function normDim(dim: number, ndim: number): number {
  if (!Number.isInteger(dim) || dim < -ndim || dim >= ndim) {
    throw new Error(`dim ${dim} is out of range for a tensor with ${ndim} dimension(s)`);
  }
  return dim < 0 ? dim + ndim : dim;
}

export function normDims(dims: number | readonly number[] | null | undefined, ndim: number): number[] {
  if (dims === null || dims === undefined) return Array.from({ length: ndim }, (_, d) => d);
  const list = typeof dims === 'number' ? [dims] : dims;
  const normalized = list.map((d) => normDim(d, ndim));
  const unique = [...new Set(normalized)].sort((a, b) => a - b);
  if (unique.length !== normalized.length) {
    throw new Error(`dim ${JSON.stringify(dims)} contains a repeated dimension`);
  }
  return unique;
}

export function reducedShape(shape: Shape, dims: number[], keepdim: boolean): number[] {
  const reduced = new Set(dims.map((d) => normDim(d, shape.length)));
  const out: number[] = [];
  shape.forEach((size, d) => {
    if (!reduced.has(d)) out.push(size);
    else if (keepdim) out.push(1);
  });
  return out;
}

export function reduceMap(shape: Shape, dims: number[], keepdim: boolean): { shape: number[]; map: Int32Array } {
  const reduced = new Set(dims.map((d) => normDim(d, shape.length)));
  const outShape = reducedShape(shape, dims, keepdim);
  const map = new Int32Array(numel(shape));
  forEachIndex(shape, (index, offset) => {
    const outIndex: number[] = [];
    index.forEach((i, d) => {
      if (!reduced.has(d)) outIndex.push(i);
      else if (keepdim) outIndex.push(0);
    });
    map[offset] = offsetOf(outShape, outIndex);
  });
  return { shape: outShape, map };
}

export function formatShape(shape: Shape): string {
  if (shape.length === 0) return '()';
  if (shape.length === 1) return `(${shape[0]},)`;
  return `(${shape.join(', ')})`;
}
