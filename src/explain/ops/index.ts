import type { OpExplain } from '../../autograd';
import { registerExplain } from '../registry';
import {
  addExplain,
  divExplain,
  expExplain,
  logExplain,
  mulExplain,
  negExplain,
  powExplain,
  rsqrtExplain,
  sigmoidExplain,
  geluExplain,
  identityExplain,
  reluExplain,
  rotateHalfExplain,
  siluExplain,
  softplusExplain,
  sqrtExplain,
  subExplain,
  transposeExplain,
  whereExplain,
  geExplain,
} from './elementwise';
import { allReduceExplain, combineExplain, dispatchExplain } from './comm';
import {
  catExplain,
  causalConv1dExplain,
  embeddingExplain,
  gatherExplain,
  indexAddExplain,
  narrowExplain,
  repeatInterleaveExplain,
  routeExpertsExplain,
  routeRowsExplain,
  selectExplain,
  topkMaskExplain,
  viewExplain,
  indexCopyExplain,
  topkIndicesExplain,
} from './index_ops';
import { nllLossExplain } from './loss';
import { groupedMmExplain, matmulExplain, scaledMmExplain, selectMatmulExplain } from './matmul';
import { logsumexpExplain, maxExplain, meanExplain, sumExplain } from './reduce';
import { blockScaleExplain, dequantizeExplain, quantizeExplain } from './quant';
import { logSoftmaxExplain, softmaxExplain } from './softmax';

export const builtinExplains: Readonly<Record<string, OpExplain>> = {
  add: addExplain,
  sub: subExplain,
  mul: mulExplain,
  div: divExplain,
  neg: negExplain,
  exp: expExplain,
  log: logExplain,
  pow: powExplain,
  rsqrt: rsqrtExplain,
  sqrt: sqrtExplain,
  sigmoid: sigmoidExplain,
  silu: siluExplain,
  softplus: softplusExplain,
  relu: reluExplain,
  gelu: geluExplain,
  identity: identityExplain,
  rotate_half: rotateHalfExplain,
  transpose: transposeExplain,
  matmul: matmulExplain,
  sum: sumExplain,
  mean: meanExplain,
  max: maxExplain,
  logsumexp: logsumexpExplain,
  softmax: softmaxExplain,
  log_softmax: logSoftmaxExplain,
  gather: gatherExplain,
  embedding: embeddingExplain,
  narrow: narrowExplain,
  cat: catExplain,
  nll_loss: nllLossExplain,
  view: viewExplain,
  select: selectExplain,
  select_matmul: selectMatmulExplain,
  repeat_interleave: repeatInterleaveExplain,
  causal_conv1d: causalConv1dExplain,
  topk_mask: topkMaskExplain,
  where: whereExplain,
  route_rows: routeRowsExplain,
  route_experts: routeExpertsExplain,
  grouped_mm: groupedMmExplain,
  scaled_mm: scaledMmExplain,
  index_add: indexAddExplain,
  ge: geExplain,
  index_copy: indexCopyExplain,
  topk_indices: topkIndicesExplain,
  block_scale: blockScaleExplain,
  quantize: quantizeExplain,
  dequantize: dequantizeExplain,
};

for (const [name, def] of Object.entries(builtinExplains)) registerExplain(name, def);
registerExplain('all_reduce', allReduceExplain);
registerExplain('dispatch', dispatchExplain);
registerExplain('combine', combineExplain);
