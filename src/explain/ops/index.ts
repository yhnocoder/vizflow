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
  sqrtExplain,
  subExplain,
  transposeExplain,
} from './elementwise';
import { catExplain, embeddingExplain, gatherExplain, narrowExplain } from './index_ops';
import { nllLossExplain } from './loss';
import { matmulExplain } from './matmul';
import { logsumexpExplain, maxExplain, meanExplain, sumExplain } from './reduce';
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
};

for (const [name, def] of Object.entries(builtinExplains)) registerExplain(name, def);
