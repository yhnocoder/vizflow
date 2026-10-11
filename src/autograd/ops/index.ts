import type { OpOptions } from '../graph';
import type { Node } from '../node';
import type { AnyOpDef } from '../op';
import { defineOp } from '../registry';
import {
  addOp,
  clampOp,
  divOp,
  erfOp,
  expOp,
  geluOp,
  identityOp,
  logOp,
  mulOp,
  negOp,
  powOp,
  reciprocalOp,
  reluOp,
  rotateHalfOp,
  rsqrtOp,
  sigmoidOp,
  siluOp,
  softplusOp,
  sqrtOp,
  subOp,
  transposeOp,
  whereOp,
  greaterOp,
  maskedFillOp,
  triuOp,
  remainderOp,
  bitwiseXorOp,
  bitwiseAndOp,
  eqOp,
  geOp,
  absOp,
  signOp,
  clampMinOp,
  logicalNotOp,
} from './elementwise';
import {
  catOp,
  causalConv1dOp,
  cummaxOp,
  embeddingOp,
  expandOp,
  gatherOp,
  indexAddOp,
  indexOp,
  narrowOp,
  nonzeroStaticOp,
  oneHotOp,
  padOp,
  permuteOp,
  repeatInterleaveOp,
  repeatOp,
  reshapeOp,
  scatterOp,
  selectOp,
  topkIndicesOp,
  topkMaskOp,
  unsqueezeOp,
  viewOp,
} from './index_ops';
import { type Reduction, nllLossOp } from './loss';
import { groupedLinearOp, matmulOp, selectMatmulOp, solveTriangularOp } from './matmul';
import { type Dim, cumsumOp, logsumexpOp, maxOp, meanOp, sumOp } from './reduce';
import { logSoftmaxOp, softmaxOp } from './softmax';

export type { Dim, Reduction };

export interface ReduceOptions extends OpOptions {
  keepdim?: boolean;
}

export interface SumOptions extends ReduceOptions {
  dim?: Dim | null;
}

export interface NllLossOptions extends OpOptions {
  reduction?: Reduction;
  ignore_index?: number;
}

export interface RsqrtOptions extends OpOptions {
  eps?: number;
}

export interface ClampOptions extends OpOptions {
  min?: number | null;
  max?: number | null;
}

export interface ConvOptions extends OpOptions {
  dilation?: number | string;
}

export interface CatOptions extends OpOptions {
  dim?: number;
  pieces?: number | string | null;
}

export interface NonzeroStaticOptions extends OpOptions {
  size?: number | string;
  fill_value?: number;
}

export interface SolveTriangularOptions extends OpOptions {
  upper?: boolean;
  unitriangular?: boolean;
}

export interface TriuOptions extends OpOptions {
  diagonal?: number;
}

export interface BuiltinOps {
  add(input: Node, other: Node | number, options?: OpOptions): Node;
  sub(input: Node, other: Node | number, options?: OpOptions): Node;
  mul(input: Node, other: Node | number, options?: OpOptions): Node;
  div(input: Node, other: Node | number, options?: OpOptions): Node;
  neg(input: Node, options?: OpOptions): Node;
  exp(input: Node, options?: OpOptions): Node;
  log(input: Node, options?: OpOptions): Node;
  pow(input: Node, exponent: number, options?: OpOptions): Node;
  rsqrt(input: Node, options?: RsqrtOptions): Node;
  sqrt(input: Node, options?: OpOptions): Node;
  reciprocal(input: Node, options?: OpOptions): Node;
  erf(input: Node, options?: OpOptions): Node;
  sigmoid(input: Node, options?: OpOptions): Node;
  silu(input: Node, options?: OpOptions): Node;
  softplus(input: Node, options?: OpOptions): Node;
  relu(input: Node, options?: OpOptions): Node;
  gelu(input: Node, options?: OpOptions): Node;
  identity(input: Node, options?: OpOptions): Node;
  rotate_half(input: Node, options?: OpOptions): Node;
  clamp(input: Node, options?: ClampOptions): Node;
  clamp(input: Node, min: number | null, max?: number | null, options?: ClampOptions): Node;
  transpose(input: Node, dim0: number, dim1: number, options?: OpOptions): Node;
  matmul(input: Node, other: Node, options?: OpOptions): Node;
  sum(input: Node, dim?: Dim | null | SumOptions, options?: SumOptions): Node;
  mean(input: Node, dim?: Dim | null | SumOptions, options?: SumOptions): Node;
  max(input: Node, dim: number, options?: ReduceOptions): Node;
  logsumexp(input: Node, dim: Dim, options?: ReduceOptions): Node;
  softmax(input: Node, dim: number, options?: OpOptions): Node;
  log_softmax(input: Node, dim: number, options?: OpOptions): Node;
  gather(input: Node, dim: number, index: Node, options?: OpOptions): Node;
  embedding(input: Node, weight: Node, options?: OpOptions): Node;
  narrow(input: Node, dim: number, start: number, length: number | string, options?: OpOptions): Node;
  cat(tensors: Node[], dim?: number | CatOptions, options?: CatOptions): Node;
  nll_loss(input: Node, target: Node, options?: NllLossOptions): Node;
  view(input: Node, shape: (number | string)[], options?: OpOptions): Node;
  view(input: Node, ...shape: (number | string)[]): Node;
  select(input: Node, dim: number, index: number, options?: OpOptions): Node;
  select_matmul(input: Node, other: Node, index: number, options?: OpOptions): Node;
  repeat_interleave(input: Node, repeats: number | string, dim: number, options?: OpOptions): Node;
  causal_conv1d(input: Node, weight: Node, options?: ConvOptions): Node;
  remainder(input: Node, other: Node | number, options?: OpOptions): Node;
  bitwise_xor(input: Node, other: Node | number, options?: OpOptions): Node;
  bitwise_and(input: Node, other: Node | number, options?: OpOptions): Node;
  eq(input: Node, other: Node | number, options?: OpOptions): Node;
  ge(input: Node, other: Node | number, options?: OpOptions): Node;
  abs(input: Node, options?: OpOptions): Node;
  sign(input: Node, options?: OpOptions): Node;
  clamp_min(input: Node, min: number, options?: OpOptions): Node;
  repeat(input: Node, sizes: (number | string)[], options?: OpOptions): Node;
  repeat(input: Node, ...sizes: (number | string)[]): Node;
  cummax(input: Node, dim: number, options?: OpOptions): Node;
  topk_mask(input: Node, k: number | string, dim?: number, options?: OpOptions): Node;
  where(condition: Node, input: Node, other: number, options?: OpOptions): Node;
  greater(input: Node, other: Node | number, options?: OpOptions): Node;
  topk_indices(input: Node, k: number | string, dim?: number, options?: OpOptions): Node;
  one_hot(input: Node, num_classes: number | string, options?: OpOptions): Node;
  permute(input: Node, dims: number[], options?: OpOptions): Node;
  permute(input: Node, ...dims: number[]): Node;
  nonzero_static(input: Node, size: number | string, options?: NonzeroStaticOptions): Node;
  nonzero_static(input: Node, options: NonzeroStaticOptions): Node;
  index(input: Node, indices: Node[], options?: OpOptions): Node;
  index_add(input: Node, dim: number, index: Node, source: Node, options?: OpOptions): Node;
  grouped_linear(input: Node, weight: Node, group: Node, options?: OpOptions): Node;
  cumsum(input: Node, dim: number, options?: OpOptions): Node;
  unsqueeze(input: Node, dim: number, options?: OpOptions): Node;
  pad(input: Node, pad: (number | string)[], options?: OpOptions): Node;
  pad(input: Node, ...pad: (number | string)[]): Node;
  triu(input: Node, diagonal?: number | TriuOptions, options?: TriuOptions): Node;
  masked_fill(input: Node, mask: Node, value: number, options?: OpOptions): Node;
  logical_not(input: Node, options?: OpOptions): Node;
  scatter(input: Node, dim: number, index: Node, value: number, options?: OpOptions): Node;
  expand(input: Node, sizes: (number | string)[], options?: OpOptions): Node;
  expand(input: Node, ...sizes: (number | string)[]): Node;
  reshape(input: Node, shape: (number | string)[], options?: OpOptions): Node;
  reshape(input: Node, ...shape: (number | string)[]): Node;
  solve_triangular(A: Node, B: Node, options: SolveTriangularOptions): Node;
}

export const builtinOps: readonly AnyOpDef[] = [
  addOp,
  subOp,
  mulOp,
  divOp,
  negOp,
  expOp,
  logOp,
  powOp,
  rsqrtOp,
  sqrtOp,
  reciprocalOp,
  erfOp,
  sigmoidOp,
  siluOp,
  softplusOp,
  reluOp,
  geluOp,
  identityOp,
  rotateHalfOp,
  clampOp,
  transposeOp,
  matmulOp,
  sumOp,
  meanOp,
  maxOp,
  logsumexpOp,
  softmaxOp,
  logSoftmaxOp,
  gatherOp,
  embeddingOp,
  narrowOp,
  catOp,
  nllLossOp,
  viewOp,
  selectOp,
  selectMatmulOp,
  repeatInterleaveOp,
  causalConv1dOp,
  topkMaskOp,
  whereOp,
  greaterOp,
  topkIndicesOp,
  oneHotOp,
  permuteOp,
  nonzeroStaticOp,
  indexOp,
  indexAddOp,
  groupedLinearOp,
  cumsumOp,
  unsqueezeOp,
  padOp,
  triuOp,
  maskedFillOp,
  solveTriangularOp,
  remainderOp,
  bitwiseXorOp,
  bitwiseAndOp,
  eqOp,
  geOp,
  absOp,
  signOp,
  clampMinOp,
  repeatOp,
  cummaxOp,
  logicalNotOp,
  scatterOp,
  expandOp,
  reshapeOp,
];

for (const def of builtinOps) defineOp(def);
