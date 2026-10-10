import { type Shape, formatShape, numel, offsetOf, stridesOf } from './shape';

export type Nested = number | Nested[];

export class Tensor {
  readonly shape: Shape;
  readonly data: Float64Array;

  constructor(shape: Shape, data?: Float64Array | number[]) {
    const size = numel(shape);
    this.shape = Object.freeze([...shape]);
    if (data === undefined) {
      this.data = new Float64Array(size);
      return;
    }
    if (data.length !== size) {
      throw new Error(`expected ${size} values for shape ${formatShape(shape)}, got ${data.length}`);
    }
    this.data = data instanceof Float64Array ? data : Float64Array.from(data);
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
    const flat: number[] = [];
    flatten(values, flat);
    return new Tensor(resolved, flat);
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

export function inferShape(values: Nested): number[] {
  if (typeof values === 'number') return [];
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

function flatten(values: Nested, out: number[]): void {
  if (typeof values === 'number') {
    out.push(values);
    return;
  }
  if (!Array.isArray(values)) throw new Error(`cannot build a tensor from ${String(values)}`);
  for (const item of values) flatten(item, out);
}
