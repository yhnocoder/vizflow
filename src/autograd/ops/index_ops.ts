import { Tensor, broadcastIndex, forEachIndex, formatShape, normDim, offsetOf, reduceMap, unbroadcast } from '../../tensor';
import type { OpDef } from '../op';
import { NO_FLOPS, VIEW_COST, movement, numel, pointwise } from './cost';

const CONV_MULTIPLY_ADD = 2;

export interface GatherAttrs {
  dim: number;
}

function gatherShape(input: readonly number[], index: readonly number[], attrs: GatherAttrs, name = 'gather'): number[] {
  if (input.length !== index.length) {
    throw new Error(`${name}: index ${formatShape(index)} must have the same number of dimensions as input ${formatShape(input)}`);
  }
  const dim = normDim(attrs.dim, input.length);
  index.forEach((size, d) => {
    if (d !== dim && size > input[d]) {
      throw new Error(`${name}: index ${formatShape(index)} is larger than input ${formatShape(input)} along dim ${d}`);
    }
  });
  return [...index];
}

function sourceOffsets(input: Tensor, index: Tensor, attrs: GatherAttrs, name = 'gather'): Int32Array {
  const dim = normDim(attrs.dim, input.ndim);
  const offsets = new Int32Array(index.size);
  forEachIndex(index.shape, (position, offset) => {
    const picked = Math.round(index.data[offset]);
    if (picked < 0 || picked >= input.shape[dim]) {
      throw new Error(`${name}: index ${picked} at [${position.join(', ')}] is out of range for input ${formatShape(input.shape)} along dim ${dim}`);
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
  cost: movement(true, [[1], []]),
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
  cost: movement(true, [[], [0]]),
};

export interface NarrowAttrs {
  dim: number;
  start: number;
  length: number;
  step?: number;
}

function narrowed(shape: readonly number[], attrs: NarrowAttrs): { dim: number; start: number; step: number; shape: number[] } {
  if (shape.length === 0) throw new Error('narrow: input must have at least one dimension');
  const dim = normDim(attrs.dim, shape.length);
  const size = shape[dim];
  const start = attrs.start < 0 ? attrs.start + size : attrs.start;
  const { length } = attrs;
  const step = attrs.step ?? 1;
  if (!Number.isInteger(step) || step < 1) throw new Error(`narrow: step ${step} must be a positive integer`);
  if (!Number.isInteger(start) || !Number.isInteger(length) || length < 0 || start < 0 || (length > 0 && start + (length - 1) * step >= size)) {
    const given = step === 1 ? `start ${attrs.start} and length ${length}` : `start ${attrs.start}, length ${length} and step ${step}`;
    throw new Error(`narrow: ${given} are out of range for input ${formatShape(shape)} along dim ${dim}`);
  }
  const out = [...shape];
  out[dim] = length;
  return { dim, start, step, shape: out };
}

export function narrowSource(inputShape: readonly number[], attrs: NarrowAttrs, out: readonly number[]): number[] {
  const { dim, start, step } = narrowed(inputShape, attrs);
  const index = [...out];
  index[dim] = start + index[dim] * step;
  return index;
}

export const narrowOp: OpDef<NarrowAttrs> = {
  name: 'narrow',
  args: ['input', 'dim', 'start', 'length', 'step'],
  tensors: ['input'],
  sizes: ['length'],
  defaults: { step: 1 },
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
  cost: VIEW_COST,
};

export interface CatAttrs {
  dim: number;
  pieces: number | null;
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
  if (attrs.pieces === null) return out;
  if (!Number.isInteger(attrs.pieces) || attrs.pieces < 1 || shapes.some((shape) => shape[dim] !== first[dim])) {
    throw new Error(`cat: pieces ${attrs.pieces} needs a positive integer and tensors of the same size along dim ${dim}`);
  }
  out[dim] = first[dim] * attrs.pieces;
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
  sizes: ['pieces'],
  defaults: { dim: 0, pieces: null },
  shape: (shapes, attrs) => catShape(shapes, attrs),
  forward: (inputs, attrs) => {
    if (attrs.pieces !== null && attrs.pieces !== inputs.length) throw new Error(`cat: pieces is ${attrs.pieces}, but ${inputs.length} tensors are given`);
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
  cost: movement(false, []),
};

export interface ViewAttrs {
  shape: number[];
}

export function viewShape(input: readonly number[], attrs: ViewAttrs, name = 'view'): number[] {
  const size = input.reduce((a, b) => a * b, 1);
  const free = attrs.shape.filter((d) => d === -1).length;
  const known = attrs.shape.filter((d) => d !== -1).reduce((a, b) => a * b, 1);
  const fail = () => new Error(`${name}: shape (${attrs.shape.join(', ')}) is invalid for input ${formatShape(input)} of size ${size}`);
  if (free > 1 || attrs.shape.some((d) => !Number.isInteger(d) || (d < 0 && d !== -1))) throw fail();
  if (free === 0 ? known !== size : known === 0 || size % known !== 0) throw fail();
  return attrs.shape.map((d) => (d === -1 ? size / known : d));
}

export const viewOp: OpDef<ViewAttrs> = {
  name: 'view',
  args: ['input', 'shape'],
  tensors: ['input'],
  rest: 'shape',
  sizes: ['shape'],
  shape: ([input], attrs) => viewShape(input, attrs),
  int64: true,
  forward: ([input], attrs) => new Tensor(viewShape(input.shape, attrs), input.data.slice(), input.int64?.slice()),
  backward: (gradOut, [input]) => [new Tensor(input.shape, gradOut.data.slice())],
  cost: VIEW_COST,
};

export const reshapeOp: OpDef<ViewAttrs> = {
  ...viewOp,
  name: 'reshape',
  shape: ([input], attrs) => viewShape(input, attrs, 'reshape'),
  forward: ([input], attrs) => new Tensor(viewShape(input.shape, attrs, 'reshape'), input.data.slice(), input.int64?.slice()),
};

export interface SelectAttrs {
  dim: number;
  index: number;
}

function selected(shape: readonly number[], attrs: SelectAttrs): { dim: number; index: number; shape: number[] } {
  if (shape.length === 0) throw new Error('select: input must have at least one dimension');
  const dim = normDim(attrs.dim, shape.length);
  const index = attrs.index < 0 ? attrs.index + shape[dim] : attrs.index;
  if (!Number.isInteger(index) || index < 0 || index >= shape[dim]) {
    throw new Error(`select: index ${attrs.index} is out of range for input ${formatShape(shape)} along dim ${dim}`);
  }
  return { dim, index, shape: shape.filter((_, d) => d !== dim) };
}

export function selectSource(inputShape: readonly number[], attrs: SelectAttrs, out: readonly number[]): number[] {
  const { dim, index } = selected(inputShape, attrs);
  return [...out.slice(0, dim), index, ...out.slice(dim)];
}

export const selectOp: OpDef<SelectAttrs> = {
  name: 'select',
  args: ['input', 'dim', 'index'],
  tensors: ['input'],
  shape: ([input], attrs) => {
    if (input.length === 0) throw new Error('select: input must have at least one dimension');
    const dim = normDim(attrs.dim, input.length);
    return input.filter((_, d) => d !== dim);
  },
  int64: true,
  forward: ([input], attrs) => {
    const shape = selected(input.shape, attrs).shape;
    const offsets: number[] = [];
    forEachIndex(shape, (index) => offsets.push(offsetOf(input.shape, selectSource(input.shape, attrs, index))));
    if (input.int64 !== undefined) return Tensor.fromInt64(shape, offsets.map((o) => input.exact(o)));
    return new Tensor(shape, offsets.map((o) => input.data[o]));
  },
  backward: (gradOut, [input], _out, attrs) => {
    const grad = new Tensor(input.shape);
    forEachIndex(gradOut.shape, (index, offset) => {
      grad.data[offsetOf(input.shape, selectSource(input.shape, attrs, index))] = gradOut.data[offset];
    });
    return [grad];
  },
  cost: VIEW_COST,
};

export interface RepeatInterleaveAttrs {
  repeats: number;
  dim: number;
}

function repeated(shape: readonly number[], attrs: RepeatInterleaveAttrs): { dim: number; shape: number[] } {
  if (shape.length === 0) throw new Error('repeat_interleave: input must have at least one dimension');
  if (!Number.isInteger(attrs.repeats) || attrs.repeats < 1) throw new Error(`repeat_interleave: repeats ${attrs.repeats} must be a positive integer`);
  const dim = normDim(attrs.dim, shape.length);
  const out = [...shape];
  out[dim] *= attrs.repeats;
  return { dim, shape: out };
}

export function repeatSource(inputShape: readonly number[], attrs: RepeatInterleaveAttrs, out: readonly number[]): number[] {
  const { dim } = repeated(inputShape, attrs);
  const index = [...out];
  index[dim] = Math.floor(out[dim] / attrs.repeats);
  return index;
}

export const repeatInterleaveOp: OpDef<RepeatInterleaveAttrs> = {
  name: 'repeat_interleave',
  args: ['input', 'repeats', 'dim'],
  tensors: ['input'],
  sizes: ['repeats'],
  shape: ([input], attrs) => repeated(input, attrs).shape,
  forward: ([input], attrs) => {
    const out = new Tensor(repeated(input.shape, attrs).shape);
    forEachIndex(out.shape, (index, offset) => {
      out.data[offset] = input.get(repeatSource(input.shape, attrs, index));
    });
    return out;
  },
  backward: (gradOut, [input], _out, attrs) => {
    const grad = new Tensor(input.shape);
    forEachIndex(gradOut.shape, (index, offset) => {
      grad.data[offsetOf(input.shape, repeatSource(input.shape, attrs, index))] += gradOut.data[offset];
    });
    return [grad];
  },
  cost: movement(true, [[]]),
};

export interface RepeatKvAttrs {
  n_rep: number;
}

export function repeatKvAsInterleave(inputShape: readonly number[], attrs: RepeatKvAttrs): RepeatInterleaveAttrs {
  if (inputShape.length < 3) throw new Error(`repeat_kv: input ${formatShape(inputShape)} must have a head dimension before (T, d)`);
  return { repeats: attrs.n_rep, dim: inputShape.length - 3 };
}

export const repeatKvOp: OpDef<RepeatKvAttrs> = {
  name: 'repeat_kv',
  args: ['input', 'n_rep'],
  tensors: ['input'],
  sizes: ['n_rep'],
  shape: ([input], attrs) => repeatInterleaveOp.shape([input], repeatKvAsInterleave(input, attrs)),
  forward: ([input], attrs) => repeatInterleaveOp.forward([input], repeatKvAsInterleave(input.shape, attrs)),
  backward: (gradOut, [input], out, attrs) => repeatInterleaveOp.backward(gradOut, [input], out, repeatKvAsInterleave(input.shape, attrs)),
  cost: movement(true, [[]]),
};

export const contiguousOp: OpDef<Record<string, never>> = {
  name: 'contiguous',
  args: ['input'],
  tensors: ['input'],
  shape: ([input]) => [...input],
  int64: true,
  forward: ([input]) => input.clone(),
  backward: (gradOut) => [gradOut.clone()],
  cost: movement(false, [[]]),
};

function convShape(input: readonly number[], weight: readonly number[]): number[] {
  if (input.length !== 2 || weight.length !== 2 || input[1] !== weight[0]) {
    throw new Error(`causal_conv1d: input ${formatShape(input)} must be (T, C) and weight ${formatShape(weight)} must be (C, K)`);
  }
  return [...input];
}

export interface ConvAttrs {
  dilation: number;
}

export function convTaps(T: number, K: number, t: number, dilation = 1): { s: number; j: number }[] {
  const taps: { s: number; j: number }[] = [];
  for (let j = 0; j < K; j++) {
    const s = t - (K - 1 - j) * dilation;
    if (s >= 0 && s < T) taps.push({ s, j });
  }
  return taps;
}

function checkDilation(attrs: ConvAttrs): number {
  if (!Number.isInteger(attrs.dilation) || attrs.dilation < 1) throw new Error(`causal_conv1d: dilation ${attrs.dilation} must be a positive integer`);
  return attrs.dilation;
}

export const causalConv1dOp: OpDef<ConvAttrs> = {
  name: 'causal_conv1d',
  args: ['input', 'weight'],
  tensors: ['input', 'weight'],
  sameDtype: ['input', 'weight'],
  sizes: ['dilation'],
  defaults: { dilation: 1 },
  shape: ([input, weight], attrs) => {
    checkDilation(attrs);
    return convShape(input, weight);
  },
  forward: ([input, weight], attrs) => {
    const [T, C] = convShape(input.shape, weight.shape);
    const K = weight.shape[1];
    const out = new Tensor([T, C]);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        let total = 0;
        for (const { s, j } of convTaps(T, K, t, attrs.dilation)) total += weight.data[c * K + j] * input.data[s * C + c];
        out.data[t * C + c] = total;
      }
    }
    return out;
  },
  backward: (gradOut, [input, weight], _out, attrs) => {
    const [T, C] = input.shape;
    const K = weight.shape[1];
    const gx = new Tensor(input.shape);
    const gw = new Tensor(weight.shape);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        const g = gradOut.data[t * C + c];
        for (const { s, j } of convTaps(T, K, t, attrs.dilation)) {
          gx.data[s * C + c] += g * weight.data[c * K + j];
          gw.data[c * K + j] += g * input.data[s * C + c];
        }
      }
    }
    return [gx, gw];
  },
  cost: {
    saves: [[1], [0]],
    flops: ([input, weight]) => {
      const count = [CONV_MULTIPLY_ADD, input[0], input[1], weight[1]];
      return { forward: count, backward: [count, count] };
    },
  },
};

export interface TopkMaskAttrs {
  k: number;
  dim: number;
}

function topkDim(shape: readonly number[], attrs: TopkMaskAttrs): number {
  if (shape.length === 0) throw new Error('topk_mask: input must have at least one dimension');
  const dim = normDim(attrs.dim, shape.length);
  if (!Number.isInteger(attrs.k) || attrs.k < 1 || attrs.k > shape[dim]) {
    throw new Error(`topk_mask: k ${attrs.k} is out of range for input ${formatShape(shape)} along dim ${dim}`);
  }
  return dim;
}

export const topkMaskOp: OpDef<TopkMaskAttrs> = {
  name: 'topk_mask',
  args: ['input', 'k', 'dim'],
  tensors: ['input'],
  sizes: ['k'],
  defaults: { dim: -1 },
  differentiable: [false],
  shape: ([input], attrs) => {
    topkDim(input, attrs);
    return [...input];
  },
  forward: ([input], attrs) => {
    const { map } = reduceMap(input.shape, [topkDim(input.shape, attrs)], false);
    const groups = new Map<number, number[]>();
    map.forEach((group, offset) => {
      const members = groups.get(group);
      if (members === undefined) groups.set(group, [offset]);
      else members.push(offset);
    });
    const out = new Tensor(input.shape);
    for (const members of groups.values()) {
      const ranked = [...members].sort((a, b) => input.data[b] - input.data[a] || a - b);
      for (const offset of ranked.slice(0, attrs.k)) out.data[offset] = 1;
    }
    return out;
  },
  backward: () => [null],
  cost: movement(false, [[]]),
};

export interface TopkIndicesAttrs {
  k: number;
  dim: number;
}

function sliceLayout(shape: readonly number[], dim: number): { outer: number; size: number; inner: number } {
  return { outer: shape.slice(0, dim).reduce((a, b) => a * b, 1), size: shape[dim], inner: shape.slice(dim + 1).reduce((a, b) => a * b, 1) };
}

function topkIndicesShape(shape: readonly number[], attrs: TopkIndicesAttrs): { dim: number; shape: number[] } {
  if (shape.length === 0) throw new Error('topk_indices: input must have at least one dimension');
  const dim = normDim(attrs.dim, shape.length);
  if (!Number.isInteger(attrs.k) || attrs.k < 1 || attrs.k > shape[dim]) {
    throw new Error(`topk_indices: k ${attrs.k} is out of range for input ${formatShape(shape)} along dim ${dim}`);
  }
  const out = [...shape];
  out[dim] = attrs.k;
  return { dim, shape: out };
}

export const topkIndicesOp: OpDef<TopkIndicesAttrs> = {
  name: 'topk_indices',
  args: ['input', 'k', 'dim'],
  tensors: ['input'],
  sizes: ['k'],
  defaults: { dim: -1 },
  differentiable: [false],
  dtype: 'int64',
  shape: ([input], attrs) => topkIndicesShape(input, attrs).shape,
  forward: ([input], attrs) => {
    const { dim, shape } = topkIndicesShape(input.shape, attrs);
    const { outer, size, inner } = sliceLayout(input.shape, dim);
    const out = new Tensor(shape);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        const at = (j: number) => (o * size + j) * inner + i;
        const ranked = Array.from({ length: size }, (_, j) => j).sort((a, b) => input.data[at(b)] - input.data[at(a)] || a - b);
        for (let j = 0; j < attrs.k; j++) out.data[(o * attrs.k + j) * inner + i] = ranked[j];
      }
    }
    return out;
  },
  backward: () => [null],
  cost: movement(false, [[]]),
};

export interface OneHotAttrs {
  num_classes: number;
}

export const oneHotOp: OpDef<OneHotAttrs> = {
  name: 'one_hot',
  args: ['input', 'num_classes'],
  tensors: ['input'],
  sizes: ['num_classes'],
  differentiable: [false],
  dtype: 'int64',
  shape: ([input], { num_classes }) => {
    if (!Number.isInteger(num_classes) || num_classes < 1) throw new Error(`one_hot: num_classes ${num_classes} must be a positive integer`);
    return [...input, num_classes];
  },
  forward: ([input], { num_classes }) => {
    const out = new Tensor([...input.shape, num_classes]);
    forEachIndex(input.shape, (position, offset) => {
      const c = Math.round(input.data[offset]);
      if (c < 0 || c >= num_classes) throw new Error(`one_hot: class ${c} at [${position.join(', ')}] is not in 0..${num_classes - 1}`);
      out.data[offset * num_classes + c] = 1;
    });
    return out;
  },
  backward: () => [null],
  cost: movement(false, [[]]),
};

export interface PermuteAttrs {
  dims: number[];
}

function permuted(shape: readonly number[], attrs: PermuteAttrs): { dims: number[]; shape: number[] } {
  if (attrs.dims.length !== shape.length) throw new Error(`permute: dims (${attrs.dims.join(', ')}) do not match input ${formatShape(shape)}`);
  const dims = attrs.dims.map((d) => normDim(d, shape.length));
  if (new Set(dims).size !== dims.length) throw new Error(`permute: dims (${attrs.dims.join(', ')}) repeat a dimension`);
  return { dims, shape: dims.map((d) => shape[d]) };
}

export function permuteSource(inputShape: readonly number[], attrs: PermuteAttrs, out: readonly number[]): number[] {
  const { dims } = permuted(inputShape, attrs);
  const source = new Array<number>(dims.length);
  dims.forEach((d, i) => {
    source[d] = out[i];
  });
  return source;
}

export const permuteOp: OpDef<PermuteAttrs> = {
  name: 'permute',
  args: ['input', 'dims'],
  tensors: ['input'],
  rest: 'dims',
  shape: ([input], attrs) => permuted(input, attrs).shape,
  forward: ([input], attrs) => {
    const out = new Tensor(permuted(input.shape, attrs).shape);
    forEachIndex(out.shape, (index, offset) => {
      out.data[offset] = input.get(permuteSource(input.shape, attrs, index));
    });
    return out;
  },
  backward: (gradOut, [input], _out, attrs) => {
    const grad = new Tensor(input.shape);
    forEachIndex(gradOut.shape, (index, offset) => {
      grad.set(permuteSource(input.shape, attrs, index), gradOut.data[offset]);
    });
    return [grad];
  },
  cost: VIEW_COST,
};

export interface NonzeroStaticAttrs {
  size: number;
  fill_value: number;
}

export const nonzeroStaticOp: OpDef<NonzeroStaticAttrs> = {
  name: 'nonzero_static',
  args: ['input', 'size'],
  tensors: ['input'],
  sizes: ['size'],
  defaults: { fill_value: -1 },
  differentiable: [false],
  dtype: 'int64',
  shape: ([input], { size }) => {
    if (!Number.isInteger(size) || size < 0) throw new Error(`nonzero_static: size ${size} must be a non-negative integer`);
    return [size, input.length];
  },
  forward: ([input], { size, fill_value }) => {
    const out = new Tensor([size, input.ndim]);
    out.data.fill(fill_value);
    let row = 0;
    forEachIndex(input.shape, (index, offset) => {
      if (input.data[offset] === 0 || row >= size) return;
      out.data.set(index, row * input.ndim);
      row++;
    });
    return out;
  },
  backward: () => [null],
  cost: movement(false, [[]]),
};

function indexShape(input: readonly number[], indices: readonly (readonly number[])[]): number[] {
  if (indices.length > input.length) throw new Error(`index: ${indices.length} index tensors are too many for input ${formatShape(input)}`);
  if (indices.some((shape) => shape.length !== indices[0].length || shape.some((n, d) => n !== indices[0][d]))) {
    throw new Error(`index: index tensors ${indices.map((shape) => formatShape(shape)).join(', ')} must have the same shape`);
  }
  return [...indices[0], ...input.slice(indices.length)];
}

export function indexSource(input: readonly number[], indices: readonly Tensor[], row: number): number[] {
  return indices.map((t, d) => {
    const raw = Math.round(t.data[row]);
    const picked = raw < 0 ? raw + input[d] : raw;
    if (picked < 0 || picked >= input[d]) throw new Error(`index: index ${raw} at [${row}] is out of range for input ${formatShape(input)} along dim ${d}`);
    return picked;
  });
}

function indexRows(input: Tensor, indices: readonly Tensor[]): { starts: number[]; width: number } {
  const width = input.shape.slice(indices.length).reduce((a, b) => a * b, 1);
  const lead = input.shape.slice(0, indices.length);
  const starts = Array.from({ length: indices[0].size }, (_, row) => offsetOf(lead, indexSource(input.shape, indices, row)) * width);
  return { starts, width };
}

export const indexOp: OpDef<Record<string, never>> = {
  name: 'index',
  args: ['input', 'indices'],
  tensors: ['input', 'indices'],
  list: 'indices',
  differentiable: [true, false],
  shape: ([input, ...indices]) => indexShape(input, indices),
  forward: ([input, ...indices]) => {
    const out = new Tensor(indexShape(input.shape, indices.map((t) => t.shape)));
    const { starts, width } = indexRows(input, indices);
    starts.forEach((start, row) => out.data.set(input.data.subarray(start, start + width), row * width));
    return out;
  },
  backward: (gradOut, [input, ...indices]) => {
    const grad = new Tensor(input.shape);
    const { starts, width } = indexRows(input, indices);
    starts.forEach((start, row) => {
      for (let c = 0; c < width; c++) grad.data[start + c] += gradOut.data[row * width + c];
    });
    return [grad, ...indices.map(() => null)];
  },
  cost: movement(true, [[1]]),
};

export interface IndexAddAttrs {
  dim: number;
}

function indexAddLayout(input: readonly number[], index: readonly number[], source: readonly number[], attrs: IndexAddAttrs): { dim: number; outer: number; inner: number } {
  if (input.length === 0) throw new Error('index_add: input must have at least one dimension');
  const dim = normDim(attrs.dim, input.length);
  const fits = index.length === 1 && source.length === input.length && index[0] === source[dim] && source.every((size, d) => d === dim || size === input[d]);
  if (!fits) throw new Error(`index_add: index ${formatShape(index)} and source ${formatShape(source)} do not fit input ${formatShape(input)} along dim ${dim}`);
  return { dim, outer: input.slice(0, dim).reduce((a, b) => a * b, 1), inner: input.slice(dim + 1).reduce((a, b) => a * b, 1) };
}

export function indexAddTarget(input: readonly number[], index: Tensor, dim: number, i: number): number {
  const raw = Math.round(index.data[i]);
  const picked = raw < 0 ? raw + input[dim] : raw;
  if (picked < 0 || picked >= input[dim]) throw new Error(`index_add: index ${raw} at [${i}] is out of range for input ${formatShape(input)} along dim ${dim}`);
  return picked;
}

function indexAddPairs(input: Tensor, index: Tensor, source: Tensor, attrs: IndexAddAttrs): [number, number][] {
  const { dim, outer, inner } = indexAddLayout(input.shape, index.shape, source.shape, attrs);
  const pairs: [number, number][] = [];
  for (let i = 0; i < index.size; i++) {
    const target = indexAddTarget(input.shape, index, dim, i);
    for (let o = 0; o < outer; o++) {
      for (let c = 0; c < inner; c++) pairs.push([(o * input.shape[dim] + target) * inner + c, (o * source.shape[dim] + i) * inner + c]);
    }
  }
  return pairs;
}

export const indexAddOp: OpDef<IndexAddAttrs> = {
  name: 'index_add',
  args: ['input', 'dim', 'index', 'source'],
  tensors: ['input', 'index', 'source'],
  sameDtype: ['input', 'source'],
  differentiable: [true, false, true],
  shape: ([input, index, source], attrs) => {
    indexAddLayout(input, index, source, attrs);
    return [...input];
  },
  forward: ([input, index, source], attrs) => {
    const out = input.clone();
    for (const [to, from] of indexAddPairs(input, index, source, attrs)) out.data[to] += source.data[from];
    return out;
  },
  backward: (gradOut, [input, index, source], _out, attrs) => {
    const gs = new Tensor(source.shape);
    for (const [to, from] of indexAddPairs(input, index, source, attrs)) gs.data[from] = gradOut.data[to];
    return [gradOut.clone(), null, gs];
  },
  cost: {
    saves: [[], [], [1]],
    flops: (inputs) => ({ forward: [numel(inputs[2])], backward: inputs.map(() => NO_FLOPS) }),
  },
};

export interface UnsqueezeAttrs {
  dim: number;
}

export function unsqueezedShape(input: readonly number[], attrs: UnsqueezeAttrs): number[] {
  const dim = normDim(attrs.dim, input.length + 1);
  return [...input.slice(0, dim), 1, ...input.slice(dim)];
}

export const unsqueezeOp: OpDef<UnsqueezeAttrs> = {
  name: 'unsqueeze',
  args: ['input', 'dim'],
  tensors: ['input'],
  shape: ([input], attrs) => unsqueezedShape(input, attrs),
  int64: true,
  forward: ([input], attrs) => new Tensor(unsqueezedShape(input.shape, attrs), input.data.slice(), input.int64?.slice()),
  backward: (gradOut, [input]) => [new Tensor(input.shape, gradOut.data.slice())],
  cost: VIEW_COST,
};

export interface SqueezeAttrs {
  dim: number;
}

function squeezedShape(input: readonly number[], attrs: SqueezeAttrs): number[] {
  if (input.length === 0) return [];
  const dim = normDim(attrs.dim, input.length);
  return input[dim] === 1 ? input.filter((_, d) => d !== dim) : [...input];
}

export const squeezeOp: OpDef<SqueezeAttrs> = {
  name: 'squeeze',
  args: ['input', 'dim'],
  tensors: ['input'],
  shape: ([input], attrs) => squeezedShape(input, attrs),
  int64: true,
  forward: ([input], attrs) => new Tensor(squeezedShape(input.shape, attrs), input.data.slice(), input.int64?.slice()),
  backward: (gradOut, [input]) => [new Tensor(input.shape, gradOut.data.slice())],
  cost: VIEW_COST,
};

export interface PadAttrs {
  pad: number[];
  value: number;
}

function padding(shape: readonly number[], attrs: PadAttrs): { before: number[]; shape: number[] } {
  const { pad } = attrs;
  if (pad.length % 2 !== 0 || pad.length / 2 > shape.length || pad.some((p) => !Number.isInteger(p) || p < 0)) {
    throw new Error(`pad: (${pad.join(', ')}) must be pairs of non-negative integers, at most one pair per dimension of ${formatShape(shape)}`);
  }
  const before = shape.map(() => 0);
  const out = [...shape];
  for (let pair = 0; pair < pad.length / 2; pair++) {
    const d = shape.length - 1 - pair;
    before[d] = pad[2 * pair];
    out[d] += pad[2 * pair] + pad[2 * pair + 1];
  }
  return { before, shape: out };
}

export function padSource(inputShape: readonly number[], attrs: PadAttrs, out: readonly number[]): number[] | undefined {
  const { before } = padding(inputShape, attrs);
  const index = out.map((i, d) => i - before[d]);
  return index.every((i, d) => i >= 0 && i < inputShape[d]) ? index : undefined;
}

export const padOp: OpDef<PadAttrs> = {
  name: 'pad',
  args: ['input', 'pad'],
  tensors: ['input'],
  rest: 'pad',
  sizes: ['pad'],
  defaults: { value: 0 },
  shape: ([input], attrs) => padding(input, attrs).shape,
  forward: ([input], attrs) => {
    const out = new Tensor(padding(input.shape, attrs).shape);
    forEachIndex(out.shape, (index, offset) => {
      const source = padSource(input.shape, attrs, index);
      out.data[offset] = source === undefined ? attrs.value : input.get(source);
    });
    return out;
  },
  backward: (gradOut, [input], _out, attrs) => {
    const grad = new Tensor(input.shape);
    forEachIndex(gradOut.shape, (index, offset) => {
      const source = padSource(input.shape, attrs, index);
      if (source !== undefined) grad.set(source, gradOut.data[offset]);
    });
    return [grad];
  },
  cost: movement(false, [[]]),
};

export interface RepeatAttrs {
  sizes: number[];
}

function repeatedShape(input: readonly number[], attrs: RepeatAttrs): number[] {
  const { sizes } = attrs;
  if (sizes.length < input.length || sizes.some((n) => !Number.isInteger(n) || n < 1)) {
    throw new Error(`repeat: sizes (${sizes.join(', ')}) must be positive integers, at least one per dimension of ${formatShape(input)}`);
  }
  const padded = [...Array(sizes.length - input.length).fill(1), ...input];
  return padded.map((d, i) => d * sizes[i]);
}

export function repeatSourceIndex(inputShape: readonly number[], out: readonly number[]): number[] {
  const lead = out.length - inputShape.length;
  return inputShape.map((d, i) => out[lead + i] % d);
}

function tiled(input: Tensor, shape: number[]): Tensor {
  const out = new Tensor(shape);
  forEachIndex(out.shape, (index, offset) => {
    out.data[offset] = input.get(repeatSourceIndex(input.shape, index));
  });
  return out;
}

export const repeatOp: OpDef<RepeatAttrs> = {
  name: 'repeat',
  args: ['input', 'sizes'],
  tensors: ['input'],
  rest: 'sizes',
  sizes: ['sizes'],
  shape: ([input], attrs) => repeatedShape(input, attrs),
  forward: ([input], attrs) => tiled(input, repeatedShape(input.shape, attrs)),
  backward: (gradOut, [input]) => {
    const grad = new Tensor(input.shape);
    forEachIndex(gradOut.shape, (index, offset) => {
      grad.data[offsetOf(input.shape, repeatSourceIndex(input.shape, index))] += gradOut.data[offset];
    });
    return [grad];
  },
  cost: movement(true, [[]]),
};

export interface ExpandAttrs {
  sizes: number[];
}

function expandedShape(input: readonly number[], attrs: ExpandAttrs): number[] {
  const { sizes } = attrs;
  const lead = sizes.length - input.length;
  const fail = () => new Error(`expand: sizes (${sizes.join(', ')}) do not expand input ${formatShape(input)}`);
  if (lead < 0) throw fail();
  return sizes.map((size, i) => {
    const d = i - lead;
    if (size === -1 && d >= 0) return input[d];
    if (!Number.isInteger(size) || size < 1) throw fail();
    if (d >= 0 && input[d] !== 1 && input[d] !== size) throw fail();
    return size;
  });
}

export const expandOp: OpDef<ExpandAttrs> = {
  ...repeatOp,
  name: 'expand',
  shape: ([input], attrs) => expandedShape(input, attrs),
  forward: ([input], attrs) => tiled(input, expandedShape(input.shape, attrs)),
  cost: { ...movement(true, [[]]), view: true },
};

export interface CummaxAttrs {
  dim: number;
}

export function cummaxSources(input: Tensor, attrs: CummaxAttrs): Int32Array {
  const dim = normDim(attrs.dim, input.ndim);
  const sources = new Int32Array(input.size);
  forEachIndex(input.shape, (index, offset) => {
    if (index[dim] === 0) {
      sources[offset] = offset;
      return;
    }
    const before = [...index];
    before[dim] -= 1;
    const previous = sources[offsetOf(input.shape, before)];
    sources[offset] = input.data[offset] >= input.data[previous] ? offset : previous;
  });
  return sources;
}

export const cummaxOp: OpDef<CummaxAttrs> = {
  name: 'cummax',
  args: ['input', 'dim'],
  tensors: ['input'],
  shape: ([input], attrs) => {
    normDim(attrs.dim, input.length);
    return [...input];
  },
  forward: ([input], attrs) => new Tensor(input.shape, Array.from(cummaxSources(input, attrs), (o) => input.data[o])),
  backward: (gradOut, [input], _out, attrs) => {
    const grad = new Tensor(input.shape);
    cummaxSources(input, attrs).forEach((source, offset) => {
      grad.data[source] += gradOut.data[offset];
    });
    return [grad];
  },
  cost: pointwise(1, 1, [[0]]),
};

export interface SortAttrs {
  dim: number;
}

export function sortSources(input: Tensor, attrs: SortAttrs): Int32Array {
  const dim = normDim(attrs.dim, input.ndim);
  const { outer, size, inner } = sliceLayout(input.shape, dim);
  const sources = new Int32Array(input.size);
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      const at = (j: number) => (o * size + j) * inner + i;
      const order = Array.from({ length: size }, (_, j) => j).sort((a, b) => input.data[at(a)] - input.data[at(b)] || a - b);
      order.forEach((j, rank) => {
        sources[at(rank)] = at(j);
      });
    }
  }
  return sources;
}

export const sortOp: OpDef<SortAttrs> = {
  name: 'sort',
  args: ['input', 'dim'],
  tensors: ['input'],
  defaults: { dim: -1 },
  shape: ([input], attrs) => {
    if (input.length === 0) throw new Error('sort: input must have at least one dimension');
    normDim(attrs.dim, input.length);
    return [...input];
  },
  forward: ([input], attrs) => new Tensor(input.shape, Array.from(sortSources(input, attrs), (o) => input.data[o])),
  backward: (gradOut, [input], _out, attrs) => {
    const grad = new Tensor(input.shape);
    sortSources(input, attrs).forEach((source, offset) => {
      grad.data[source] += gradOut.data[offset];
    });
    return [grad];
  },
  cost: movement(true, [[]]),
};

export interface ScatterAttrs {
  dim: number;
  src?: number;
}

function scatterLayout(input: readonly number[], index: readonly number[], src: readonly number[] | undefined, attrs: ScatterAttrs): void {
  gatherShape(input, index, attrs, 'scatter');
  if (src === undefined) return;
  if (src.length !== input.length) throw new Error(`scatter: src ${formatShape(src)} must have the same number of dimensions as input ${formatShape(input)}`);
  index.forEach((n, d) => {
    if (n > src[d]) throw new Error(`scatter: index ${formatShape(index)} is larger than src ${formatShape(src)} along dim ${d}`);
  });
}

export function scatterPairs(input: Tensor, index: Tensor, src: Tensor | undefined, attrs: ScatterAttrs): [number, number][] {
  scatterLayout(input.shape, index.shape, src?.shape, attrs);
  const targets = sourceOffsets(input, index, attrs, 'scatter');
  const pairs: [number, number][] = [];
  forEachIndex(index.shape, (position, k) => pairs.push([targets[k], src === undefined ? k : offsetOf(src.shape, position)]));
  return pairs;
}

export const scatterOp: OpDef<ScatterAttrs> = {
  name: 'scatter',
  args: ['input', 'dim', 'index', 'src'],
  tensors: ['input', 'index', 'src'],
  sameDtype: ['input', 'src'],
  scalars: ['src'],
  differentiable: [true, false, true],
  shape: ([input, index, src], attrs) => {
    scatterLayout(input, index, src, attrs);
    return [...input];
  },
  forward: ([input, index, src], attrs) => {
    const out = input.clone();
    for (const [to, from] of scatterPairs(input, index, src, attrs)) out.data[to] = src === undefined ? attrs.src! : src.data[from];
    return out;
  },
  backward: (gradOut, [input, index, src], _out, attrs) => {
    const pairs = scatterPairs(input, index, src, attrs);
    const gi = gradOut.clone();
    for (const [to] of pairs) gi.data[to] = 0;
    if (src === undefined) return [gi, null];
    const gs = new Tensor(src.shape);
    for (const [to, from] of pairs) gs.data[from] = gradOut.data[to];
    return [gi, null, gs];
  },
  cost: movement(false, [[1], [], [1]]),
};

function indexPutLayout(input: readonly number[], mask: readonly number[], values: readonly number[]): number[] {
  if (mask.length === 0 || mask.length > input.length || mask.some((n, d) => n !== input[d])) {
    throw new Error(`index_put: mask ${formatShape(mask)} must match the leading dimensions of input ${formatShape(input)}`);
  }
  const rest = input.slice(mask.length);
  const tail = values.slice(Math.max(0, values.length - rest.length));
  const fits = values.length <= rest.length + 1 && tail.every((n, d) => n === 1 || n === rest[rest.length - tail.length + d]);
  if (!fits) throw new Error(`index_put: values ${formatShape(values)} cannot be broadcast to the selected rows of input ${formatShape(input)}`);
  return rest;
}

export function indexPutRows(mask: Tensor): number[] {
  const rows: number[] = [];
  for (let r = 0; r < mask.size; r++) if (mask.data[r] !== 0) rows.push(r);
  return rows;
}

export const indexPutOp: OpDef<Record<string, never>> = {
  name: 'index_put',
  args: ['input', 'mask', 'values'],
  tensors: ['input', 'mask', 'values'],
  sameDtype: ['input', 'values'],
  differentiable: [true, false, true],
  shape: ([input, mask, values]) => {
    indexPutLayout(input, mask, values);
    return [...input];
  },
  forward: ([input, mask, values]) => {
    const rest = indexPutLayout(input.shape, mask.shape, values.shape);
    const rows = indexPutRows(mask);
    const target = [rows.length, ...rest];
    if (values.shape.length === target.length && values.shape[0] !== 1 && values.shape[0] !== rows.length) {
      throw new Error(`index_put: values ${formatShape(values.shape)} has ${values.shape[0]} rows, but the mask selects ${rows.length}`);
    }
    const width = numel(rest);
    const out = input.clone();
    rows.forEach((r, k) => {
      forEachIndex(rest, (inner, c) => {
        out.data[r * width + c] = values.data[offsetOf(values.shape, broadcastIndex([k, ...inner], values.shape))];
      });
    });
    return out;
  },
  backward: (gradOut, [input, mask, values]) => {
    const rest = indexPutLayout(input.shape, mask.shape, values.shape);
    const rows = indexPutRows(mask);
    const width = numel(rest);
    const gi = gradOut.clone();
    const picked = new Tensor([rows.length, ...rest]);
    rows.forEach((r, k) => {
      for (let c = 0; c < width; c++) {
        picked.data[k * width + c] = gradOut.data[r * width + c];
        gi.data[r * width + c] = 0;
      }
    });
    return [gi, null, unbroadcast(picked, values.shape)];
  },
  cost: movement(true, [[], [], []]),
};

export function selectScatterTarget(input: readonly number[], src: readonly number[], attrs: SelectAttrs): { dim: number; index: number } {
  const { dim, index, shape } = selected(input, attrs);
  if (shape.length !== src.length || shape.some((size, d) => size !== src[d])) {
    throw new Error(`select_scatter: src ${formatShape(src)} must have the shape ${formatShape(shape)} of input ${formatShape(input)} without dim ${dim}`);
  }
  return { dim, index };
}

export const selectScatterOp: OpDef<SelectAttrs> = {
  name: 'select_scatter',
  args: ['input', 'src', 'dim', 'index'],
  tensors: ['input', 'src'],
  sameDtype: ['input', 'src'],
  shape: ([input, src], attrs) => {
    selectScatterTarget(input, src, attrs);
    return [...input];
  },
  forward: ([input, src], attrs) => {
    const out = input.clone();
    forEachIndex(src.shape, (index, offset) => {
      out.data[offsetOf(input.shape, selectSource(input.shape, attrs, index))] = src.data[offset];
    });
    return out;
  },
  backward: (gradOut, [input, src], _out, attrs) => {
    const gi = gradOut.clone();
    const gs = new Tensor(src.shape);
    forEachIndex(src.shape, (index, offset) => {
      const at = offsetOf(input.shape, selectSource(input.shape, attrs, index));
      gs.data[offset] = gradOut.data[at];
      gi.data[at] = 0;
    });
    return [gi, gs];
  },
  cost: movement(false, [[], []]),
};

export interface SliceScatterAttrs {
  dim: number;
  start: number | null;
  end: number | null;
  step: number;
}

export function sliceRange(size: number, attrs: SliceScatterAttrs): { start: number; step: number; length: number } {
  if (!Number.isInteger(attrs.step) || attrs.step <= 0) throw new Error('slice_scatter: step must be a positive integer');
  const clamp = (value: number | null, fallback: number) => (value === null ? fallback : Math.min(size, Math.max(0, value < 0 ? value + size : value)));
  const start = clamp(attrs.start, 0);
  const end = clamp(attrs.end, size);
  return { start, step: attrs.step, length: Math.max(0, Math.ceil((end - start) / attrs.step)) };
}

export function sliceScatterTarget(input: readonly number[], src: readonly number[], attrs: SliceScatterAttrs): { dim: number; start: number; step: number } {
  if (input.length === 0) throw new Error('slice_scatter: input must have at least one dimension');
  const dim = normDim(attrs.dim, input.length);
  const { start, step, length } = sliceRange(input[dim], attrs);
  const expected = input.map((size, d) => (d === dim ? length : size));
  if (src.length !== expected.length || src.some((size, d) => size !== expected[d])) {
    throw new Error(`slice_scatter: src ${formatShape(src)} must have the shape ${formatShape(expected)} of the slice of input ${formatShape(input)}`);
  }
  return { dim, start, step };
}

export function sliceScatterSource(input: readonly number[], src: readonly number[], attrs: SliceScatterAttrs, out: readonly number[]): { arg: number; index: number[] } {
  const { dim, start, step } = sliceScatterTarget(input, src, attrs);
  const k = (out[dim] - start) / step;
  if (Number.isInteger(k) && k >= 0 && k < src[dim]) return { arg: 1, index: out.map((v, d) => (d === dim ? k : v)) };
  return { arg: 0, index: [...out] };
}

export const sliceScatterOp: OpDef<SliceScatterAttrs> = {
  name: 'slice_scatter',
  args: ['input', 'src', 'dim', 'start', 'end', 'step'],
  tensors: ['input', 'src'],
  sizes: ['start', 'end'],
  defaults: { dim: 0, start: null, end: null, step: 1 },
  sameDtype: ['input', 'src'],
  shape: ([input, src], attrs) => {
    sliceScatterTarget(input, src, attrs);
    return [...input];
  },
  forward: ([input, src], attrs) => {
    const { dim, start, step } = sliceScatterTarget(input.shape, src.shape, attrs);
    const out = input.clone();
    forEachIndex(src.shape, (index, offset) => {
      out.data[offsetOf(input.shape, index.map((v, d) => (d === dim ? start + v * step : v)))] = src.data[offset];
    });
    return out;
  },
  backward: (gradOut, [input, src], _out, attrs) => {
    const { dim, start, step } = sliceScatterTarget(input.shape, src.shape, attrs);
    const gi = gradOut.clone();
    const gs = new Tensor(src.shape);
    forEachIndex(src.shape, (index, offset) => {
      const at = offsetOf(input.shape, index.map((v, d) => (d === dim ? start + v * step : v)));
      gs.data[offset] = gradOut.data[at];
      gi.data[at] = 0;
    });
    return [gi, gs];
  },
  cost: movement(false, [[], []]),
};
