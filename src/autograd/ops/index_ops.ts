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
