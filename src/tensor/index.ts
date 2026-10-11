export { EXACT_LIMIT, Tensor, exactInteger, inferShape, type Nested } from './tensor';
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
export { formatNumber, defaultDigits } from './format';
