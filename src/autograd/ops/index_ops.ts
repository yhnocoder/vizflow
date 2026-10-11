import { Tensor, forEachIndex, formatShape, normDim, offsetOf, reduceMap } from '../../tensor';
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
};

export interface ViewAttrs {
  shape: number[];
}

export function viewShape(input: readonly number[], attrs: ViewAttrs): number[] {
  const size = input.reduce((a, b) => a * b, 1);
  const free = attrs.shape.filter((d) => d === -1).length;
  const known = attrs.shape.filter((d) => d !== -1).reduce((a, b) => a * b, 1);
  const fail = () => new Error(`view: shape (${attrs.shape.join(', ')}) is invalid for input ${formatShape(input)} of size ${size}`);
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
  forward: ([input], attrs) => new Tensor(viewShape(input.shape, attrs), input.data.slice()),
  backward: (gradOut, [input]) => [new Tensor(input.shape, gradOut.data.slice())],
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
  shape: ([input], attrs) => selected(input, attrs).shape,
  forward: ([input], attrs) => {
    const out = new Tensor(selected(input.shape, attrs).shape);
    forEachIndex(out.shape, (index, offset) => {
      out.data[offset] = input.get(selectSource(input.shape, attrs, index));
    });
    return out;
  },
  backward: (gradOut, [input], _out, attrs) => {
    const grad = new Tensor(input.shape);
    forEachIndex(gradOut.shape, (index, offset) => {
      grad.data[offsetOf(input.shape, selectSource(input.shape, attrs, index))] = gradOut.data[offset];
    });
    return [grad];
  },
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
};

function convShape(input: readonly number[], weight: readonly number[]): number[] {
  if (input.length !== 2 || weight.length !== 2 || input[1] !== weight[0]) {
    throw new Error(`causal_conv1d: input ${formatShape(input)} must be (T, C) and weight ${formatShape(weight)} must be (C, K)`);
  }
  return [...input];
}

export function convTaps(T: number, K: number, t: number): { s: number; j: number }[] {
  const taps: { s: number; j: number }[] = [];
  for (let j = 0; j < K; j++) {
    const s = t - K + 1 + j;
    if (s >= 0 && s < T) taps.push({ s, j });
  }
  return taps;
}

export const causalConv1dOp: OpDef<Record<string, never>> = {
  name: 'causal_conv1d',
  args: ['input', 'weight'],
  tensors: ['input', 'weight'],
  shape: ([input, weight]) => convShape(input, weight),
  forward: ([input, weight]) => {
    const [T, C] = convShape(input.shape, weight.shape);
    const K = weight.shape[1];
    const out = new Tensor([T, C]);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        let total = 0;
        for (const { s, j } of convTaps(T, K, t)) total += weight.data[c * K + j] * input.data[s * C + c];
        out.data[t * C + c] = total;
      }
    }
    return out;
  },
  backward: (gradOut, [input, weight]) => {
    const [T, C] = input.shape;
    const K = weight.shape[1];
    const gx = new Tensor(input.shape);
    const gw = new Tensor(weight.shape);
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < C; c++) {
        const g = gradOut.data[t * C + c];
        for (const { s, j } of convTaps(T, K, t)) {
          gx.data[s * C + c] += g * weight.data[c * K + j];
          gw.data[c * K + j] += g * input.data[s * C + c];
        }
      }
    }
    return [gx, gw];
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
};

export interface RouteAttrs {
  k: number;
}

export interface RoutedPair {
  token: number;
  expert: number;
}

function routeShape(sel: readonly number[], name: string, attrs: RouteAttrs): number[] {
  if (sel.length !== 2) throw new Error(`${name}: sel ${formatShape(sel)} must be (T, E)`);
  if (!Number.isInteger(attrs.k) || attrs.k < 1 || attrs.k > sel[1]) {
    throw new Error(`${name}: k ${attrs.k} is out of range for sel ${formatShape(sel)}`);
  }
  return [sel[0] * attrs.k];
}

export function routeOrder(sel: Tensor, name: string, k: number): RoutedPair[] {
  const [T, E] = sel.shape;
  for (let t = 0; t < T; t++) {
    let count = 0;
    for (let e = 0; e < E; e++) {
      const v = sel.data[t * E + e];
      if (v !== 0 && v !== 1) throw new Error(`${name}: sel[${t}, ${e}] is ${v}, expected 0 or 1`);
      count += v;
    }
    if (count !== k) throw new Error(`${name}: row ${t} of sel has ${count} ones, expected k = ${k}`);
  }
  const pairs: RoutedPair[] = [];
  for (let e = 0; e < E; e++) {
    for (let t = 0; t < T; t++) if (sel.data[t * E + e] === 1) pairs.push({ token: t, expert: e });
  }
  return pairs;
}

function routeOp(name: string, pick: (pair: RoutedPair) => number): OpDef<RouteAttrs> {
  return {
    name,
    args: ['sel', 'k'],
    tensors: ['sel'],
    sizes: ['k'],
    differentiable: [false],
    shape: ([sel], attrs) => routeShape(sel, name, attrs),
    forward: ([sel], attrs) => new Tensor(routeShape(sel.shape, name, attrs), Float64Array.from(routeOrder(sel, name, attrs.k), pick)),
    backward: () => [null],
  };
}

export const routeRowsOp = routeOp('route_rows', (pair) => pair.token);
export const routeExpertsOp = routeOp('route_experts', (pair) => pair.expert);

export interface IndexAddAttrs {
  size: number;
}

function indexAddShape(src: readonly number[], index: readonly number[], attrs: IndexAddAttrs): number[] {
  if (src.length === 0 || index.length !== 1 || index[0] !== src[0]) {
    throw new Error(`index_add: index ${formatShape(index)} must be one-dimensional with one entry per row of src ${formatShape(src)}`);
  }
  if (!Number.isInteger(attrs.size) || attrs.size < 1) throw new Error(`index_add: size ${attrs.size} must be a positive integer`);
  return [attrs.size, ...src.slice(1)];
}

export function indexAddTarget(index: Tensor, row: number, size: number): number {
  const target = Math.round(index.data[row]);
  if (target < 0 || target >= size) throw new Error(`index_add: index ${target} at [${row}] is out of range for size ${size}`);
  return target;
}

export const indexAddOp: OpDef<IndexAddAttrs> = {
  name: 'index_add',
  args: ['src', 'index', 'size'],
  tensors: ['src', 'index'],
  sizes: ['size'],
  differentiable: [true, false],
  shape: ([src, index], attrs) => indexAddShape(src, index, attrs),
  forward: ([src, index], attrs) => {
    const out = new Tensor(indexAddShape(src.shape, index.shape, attrs));
    const width = src.size / src.shape[0];
    for (let r = 0; r < src.shape[0]; r++) {
      const t = indexAddTarget(index, r, attrs.size);
      for (let c = 0; c < width; c++) out.data[t * width + c] += src.data[r * width + c];
    }
    return out;
  },
  backward: (gradOut, [src, index], _out, attrs) => {
    const grad = new Tensor(src.shape);
    const width = src.size / src.shape[0];
    for (let r = 0; r < src.shape[0]; r++) {
      const t = indexAddTarget(index, r, attrs.size);
      grad.data.set(gradOut.data.subarray(t * width, (t + 1) * width), r * width);
    }
    return [grad, null];
  },
};
