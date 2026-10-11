import { type Shape, formatShape, numel, offsetOf, stridesOf } from './shape';

export type Nested = number | bigint | Nested[];

export const EXACT_LIMIT = 2 ** 53;

export class Tensor {
  readonly shape: Shape;
  readonly data: Float64Array;
  readonly int64: BigInt64Array | undefined;

  constructor(shape: Shape, data?: Float64Array | number[], int64?: BigInt64Array) {
    const size = numel(shape);
    this.shape = Object.freeze([...shape]);
    if (int64 !== undefined) {
      if (int64.length !== size) throw new Error(`expected ${size} values for shape ${formatShape(shape)}, got ${int64.length}`);
      this.int64 = int64;
      this.data = Float64Array.from(int64, Number);
      return;
    }
    this.int64 = undefined;
    if (data === undefined) {
      this.data = new Float64Array(size);
      return;
    }
    if (data.length !== size) {
      throw new Error(`expected ${size} values for shape ${formatShape(shape)}, got ${data.length}`);
    }
    this.data = data instanceof Float64Array ? data : Float64Array.from(data);
  }

  static fromInt64(shape: Shape, values: readonly bigint[] | BigInt64Array): Tensor {
    const exact = new BigInt64Array(values.length);
    values.forEach((v, i) => {
      if (BigInt.asIntN(64, v) !== v) throw new Error(`${v} is outside the int64 range`);
      exact[i] = v;
    });
    return new Tensor(shape, undefined, exact);
  }

  asInt64(): Tensor {
    if (this.int64 !== undefined) return this;
    return Tensor.fromInt64(this.shape, Array.from(this.data, (v) => exactInteger(v)));
  }

  exact(offset: number): bigint {
    return this.int64 !== undefined ? this.int64[offset] : exactInteger(this.data[offset]);
  }

  beyondExact(): boolean {
    return this.int64 !== undefined && this.int64.some((v) => v >= EXACT_LIMIT_BIG || v <= -EXACT_LIMIT_BIG);
  }

  static zeros(shape: Shape): Tensor {
    return new Tensor(shape);
  }

  static from(values: Nested, shape?: Shape): Tensor {
    const nested = Array.isArray(values) && values.some(Array.isArray);
    if (shape !== undefined && nested) {
      const inferred = inferShape(values);
      if (formatShape(inferred) !== formatShape(shape)) {
        throw new Error(`expected shape ${formatShape(shape)}, got nested values of shape ${formatShape(inferred)}`);
      }
    }
    const resolved = shape ?? inferShape(values);
    const flat: (number | bigint)[] = [];
    flatten(values, flat);
    if (flat.some((v) => typeof v === 'bigint')) return Tensor.fromInt64(resolved, flat.map((v) => (typeof v === 'bigint' ? v : exactInteger(v))));
    return new Tensor(resolved, flat as number[]);
  }

  get size(): number {
    return this.data.length;
  }

  get ndim(): number {
    return this.shape.length;
  }

  get(index: readonly number[]): number {
    return this.data[offsetOf(this.shape, index)];
  }

  set(index: readonly number[], value: number): void {
    this.data[offsetOf(this.shape, index)] = value;
  }

  clone(): Tensor {
    if (this.int64 !== undefined) return new Tensor(this.shape, undefined, new BigInt64Array(this.int64));
    return new Tensor(this.shape, new Float64Array(this.data));
  }

  toNested(): Nested {
    if (this.shape.length === 0) return this.data[0];
    const strides = stridesOf(this.shape);
    const build = (dim: number, base: number): Nested => {
      const items: Nested[] = [];
      for (let i = 0; i < this.shape[dim]; i++) {
        const offset = base + i * strides[dim];
        items.push(dim === this.shape.length - 1 ? this.data[offset] : build(dim + 1, offset));
      }
      return items;
    };
    return build(0, 0);
  }
}

const EXACT_LIMIT_BIG = BigInt(EXACT_LIMIT);

export function exactInteger(v: number): bigint {
  if (!Number.isInteger(v) || Math.abs(v) > EXACT_LIMIT) throw new Error(`${v} is not an integer that a float64 holds exactly`);
  return BigInt(v);
}

export function inferShape(values: Nested): number[] {
  if (typeof values === 'number' || typeof values === 'bigint') return [];
  if (!Array.isArray(values)) throw new Error(`cannot build a tensor from ${String(values)}`);
  if (values.length === 0) return [0];
  const inner = inferShape(values[0]);
  for (const item of values) {
    const other = inferShape(item);
    if (other.length !== inner.length || other.some((d, i) => d !== inner[i])) {
      throw new Error('nested values are ragged');
    }
  }
  return [values.length, ...inner];
}

function flatten(values: Nested, out: (number | bigint)[]): void {
  if (typeof values === 'number' || typeof values === 'bigint') {
    out.push(values);
    return;
  }
  if (!Array.isArray(values)) throw new Error(`cannot build a tensor from ${String(values)}`);
  for (const item of values) flatten(item, out);
}
