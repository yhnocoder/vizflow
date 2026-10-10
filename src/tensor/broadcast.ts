import { type Shape, formatShape, reduceMap } from './shape';
import { Tensor } from './tensor';

export function broadcastShapes(a: Shape, b: Shape): number[] {
  const ndim = Math.max(a.length, b.length);
  const out = new Array<number>(ndim);
  for (let i = 1; i <= ndim; i++) {
    const da = i <= a.length ? a[a.length - i] : 1;
    const db = i <= b.length ? b[b.length - i] : 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`cannot broadcast ${formatShape(a)} with ${formatShape(b)}`);
    }
    out[ndim - i] = da === 1 ? db : da;
  }
  return out;
}

export function broadcastIndex(outIndex: readonly number[], inShape: Shape): number[] {
  const skip = outIndex.length - inShape.length;
  if (skip < 0) {
    throw new Error(`output index ${formatShape(outIndex)} has fewer dimensions than input shape ${formatShape(inShape)}`);
  }
  return inShape.map((size, d) => (size === 1 ? 0 : outIndex[skip + d]));
}

export function unbroadcast(grad: Tensor, shape: Shape): Tensor {
  if (grad.ndim < shape.length) {
    throw new Error(`cannot reduce gradient ${formatShape(grad.shape)} to shape ${formatShape(shape)}`);
  }
  const skip = grad.ndim - shape.length;
  const dims: number[] = [];
  for (let d = 0; d < grad.ndim; d++) {
    if (d < skip) dims.push(d);
    else if (shape[d - skip] === 1 && grad.shape[d] !== 1) dims.push(d);
  }
  if (dims.length === 0) return grad.clone();
  const { map } = reduceMap(grad.shape, dims, true);
  const out = new Tensor(shape);
  for (let i = 0; i < map.length; i++) out.data[map[i]] += grad.data[i];
  return out;
}
