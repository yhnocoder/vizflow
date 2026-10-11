import { Tensor, forEachIndex, formatShape, normDim, offsetOf } from '../../tensor';
import type { OpDef } from '../op';

export interface GatherAttrs {
  dim: number;
}

function gatherShape(input: readonly number[], index: readonly number[], attrs: GatherAttrs): number[] {
  if (input.length !== index.length) {
    throw new Error(`gather: index ${formatShape(index)} must have the same number of dimensions as input ${formatShape(input)}`);
  }
  const dim = normDim(attrs.dim, input.length);
  index.forEach((size, d) => {
    if (d !== dim && size > input[d]) {
      throw new Error(`gather: index ${formatShape(index)} is larger than input ${formatShape(input)} along dim ${d}`);
    }
  });
  return [...index];
}

function sourceOffsets(input: Tensor, index: Tensor, attrs: GatherAttrs): Int32Array {
  const dim = normDim(attrs.dim, input.ndim);
  const offsets = new Int32Array(index.size);
  forEachIndex(index.shape, (position, offset) => {
    const picked = Math.round(index.data[offset]);
    if (picked < 0 || picked >= input.shape[dim]) {
      throw new Error(`gather: index ${picked} at [${position.join(', ')}] is out of range for input ${formatShape(input.shape)} along dim ${dim}`);
    }
    const source = [...position];
    source[dim] = picked;
    offsets[offset] = offsetOf(input.shape, source);
  });
  return offsets;
}

export const gatherOp: OpDef<GatherAttrs> = {
  name: 'gather',
  args: ['input', 'dim', 'index'],
  tensors: ['input', 'index'],
  differentiable: [true, false],
  shape: ([input, index], attrs) => gatherShape(input, index, attrs),
  forward: ([input, index], attrs) => {
    gatherShape(input.shape, index.shape, attrs);
    const offsets = sourceOffsets(input, index, attrs);
    return new Tensor(index.shape, Float64Array.from(offsets, (o) => input.data[o]));
  },
  backward: (gradOut, [input, index], _out, attrs) => {
    const grad = new Tensor(input.shape);
    const offsets = sourceOffsets(input, index, attrs);
    for (let i = 0; i < offsets.length; i++) grad.data[offsets[i]] += gradOut.data[i];
    return [grad, null];
  },
};

function embeddingRow(weight: Tensor, id: number, position: number[]): number {
  const row = Math.round(id);
  if (row < 0 || row >= weight.shape[0]) {
    throw new Error(`embedding: index ${row} at [${position.join(', ')}] is out of range for weight ${formatShape(weight.shape)}`);
  }
  return row;
}

function embeddingShape(input: readonly number[], weight: readonly number[]): number[] {
  if (weight.length !== 2) throw new Error(`embedding: weight ${formatShape(weight)} must be two-dimensional`);
  return [...input, weight[1]];
}

export const embeddingOp: OpDef<Record<string, never>> = {
  name: 'embedding',
  args: ['input', 'weight'],
  tensors: ['input', 'weight'],
  differentiable: [false, true],
  shape: ([input, weight]) => embeddingShape(input, weight),
  forward: ([input, weight]) => {
    const dim = weight.shape[1];
    const out = new Tensor(embeddingShape(input.shape, weight.shape));
    forEachIndex(input.shape, (position, offset) => {
      const row = embeddingRow(weight, input.data[offset], position);
      out.data.set(weight.data.subarray(row * dim, (row + 1) * dim), offset * dim);
    });
    return out;
  },
  backward: (gradOut, [input, weight]) => {
    const dim = weight.shape[1];
    const grad = new Tensor(weight.shape);
    forEachIndex(input.shape, (position, offset) => {
      const row = embeddingRow(weight, input.data[offset], position);
      for (let c = 0; c < dim; c++) grad.data[row * dim + c] += gradOut.data[offset * dim + c];
    });
    return [null, grad];
  },
};

export interface NarrowAttrs {
  dim: number;
  start: number;
  length: number;
}

function narrowed(shape: readonly number[], attrs: NarrowAttrs): { dim: number; start: number; shape: number[] } {
  if (shape.length === 0) throw new Error('narrow: input must have at least one dimension');
  const dim = normDim(attrs.dim, shape.length);
  const size = shape[dim];
  const start = attrs.start < 0 ? attrs.start + size : attrs.start;
  const { length } = attrs;
  if (!Number.isInteger(start) || !Number.isInteger(length) || length < 0 || start < 0 || start + length > size) {
    throw new Error(`narrow: start ${attrs.start} and length ${length} are out of range for input ${formatShape(shape)} along dim ${dim}`);
  }
  const out = [...shape];
  out[dim] = length;
  return { dim, start, shape: out };
}

export function narrowSource(inputShape: readonly number[], attrs: NarrowAttrs, out: readonly number[]): number[] {
  const { dim, start } = narrowed(inputShape, attrs);
  const index = [...out];
  index[dim] += start;
  return index;
}

export const narrowOp: OpDef<NarrowAttrs> = {
  name: 'narrow',
  args: ['input', 'dim', 'start', 'length'],
  tensors: ['input'],
  sizes: ['length'],
  shape: ([input], attrs) => narrowed(input, attrs).shape,
  forward: ([input], attrs) => {
    const out = new Tensor(narrowed(input.shape, attrs).shape);
    forEachIndex(out.shape, (index, offset) => {
      out.data[offset] = input.get(narrowSource(input.shape, attrs, index));
    });
    return out;
  },
  backward: (gradOut, [input], _out, attrs) => {
    const grad = new Tensor(input.shape);
    forEachIndex(gradOut.shape, (index, offset) => {
      grad.data[offsetOf(input.shape, narrowSource(input.shape, attrs, index))] = gradOut.data[offset];
    });
    return [grad];
  },
};

export interface CatAttrs {
  dim: number;
}

function catShape(shapes: readonly (readonly number[])[], attrs: CatAttrs): number[] {
  const first = shapes[0];
  if (first.length === 0) throw new Error('cat: zero-dimensional tensors cannot be concatenated');
  const dim = normDim(attrs.dim, first.length);
  const out = [...first];
  out[dim] = 0;
  for (const shape of shapes) {
    if (shape.length !== first.length || shape.some((size, d) => d !== dim && size !== first[d])) {
      throw new Error(`cat: tensors ${formatShape(first)} and ${formatShape(shape)} differ outside dim ${dim}`);
    }
    out[dim] += shape[dim];
  }
  return out;
}

export function catSource(shapes: readonly (readonly number[])[], attrs: CatAttrs, out: readonly number[]): { arg: number; index: number[] } {
  const dim = normDim(attrs.dim, out.length);
  let position = out[dim];
  let arg = 0;
  while (position >= shapes[arg][dim]) {
    position -= shapes[arg][dim];
    arg++;
  }
  const index = [...out];
  index[dim] = position;
  return { arg, index };
}

export const catOp: OpDef<CatAttrs> = {
  name: 'cat',
  args: ['tensors', 'dim'],
  tensors: ['tensors'],
  list: 'tensors',
  defaults: { dim: 0 },
  shape: (shapes, attrs) => catShape(shapes, attrs),
  forward: (inputs, attrs) => {
    const shapes = inputs.map((t) => t.shape);
    const out = new Tensor(catShape(shapes, attrs));
    forEachIndex(out.shape, (index, offset) => {
      const source = catSource(shapes, attrs, index);
      out.data[offset] = inputs[source.arg].get(source.index);
    });
    return out;
  },
  backward: (gradOut, inputs, _out, attrs) => {
    const shapes = inputs.map((t) => t.shape);
    const grads = inputs.map((t) => new Tensor(t.shape));
    forEachIndex(gradOut.shape, (index, offset) => {
      const source = catSource(shapes, attrs, index);
      grads[source.arg].set(source.index, gradOut.data[offset]);
    });
    return grads;
  },
};
