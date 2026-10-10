export { Tensor, inferShape, type Nested } from './tensor';
export {
  type Shape,
  numel,
  stridesOf,
  offsetOf,
  indexOf,
  forEachIndex,
  normDim,
  normDims,
  reducedShape,
  reduceMap,
  formatShape,
} from './shape';
export { broadcastShapes, broadcastIndex, unbroadcast } from './broadcast';
export { formatNumber } from './format';
