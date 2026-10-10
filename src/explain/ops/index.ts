import type { OpExplain } from '../../autograd';
import { registerExplain } from '../registry';
import { addExplain, divExplain, expExplain, logExplain, mulExplain, negExplain, subExplain, transposeExplain } from './elementwise';
import { gatherExplain } from './index_ops';
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
  transpose: transposeExplain,
  matmul: matmulExplain,
  sum: sumExplain,
  mean: meanExplain,
  max: maxExplain,
  logsumexp: logsumexpExplain,
  softmax: softmaxExplain,
  log_softmax: logSoftmaxExplain,
  gather: gatherExplain,
  nll_loss: nllLossExplain,
};

for (const [name, def] of Object.entries(builtinExplains)) registerExplain(name, def);
