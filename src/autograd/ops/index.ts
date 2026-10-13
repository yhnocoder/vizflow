import type { OpOptions } from '../graph';
import type { Node } from '../node';
import type { Dtype } from '../dtype';
import type { AnyOpDef, CompositeDef } from '../op';
import { defineComposite, defineOp } from '../registry';
import {
  addOp,
  clampOp,
  divOp,
  erfOp,
  cosOp,
  expOp,
  sinOp,
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
  rsubOp,
  transposeOp,
  whereOp,
  greaterOp,
  maskedFillOp,
  triuOp,
  remainderOp,
  floorDivideOp,
  bitwiseXorOp,
  bitwiseAndOp,
  eqOp,
  geOp,
  absOp,
  signOp,
  clampMinOp,
  ltOp,
  bitwiseOrOp,
  logicalNotOp,
  copysignOp,
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
  repeatKvOp,
  contiguousOp,
  repeatOp,
  reshapeOp,
  scatterOp,
  selectOp,
  topkIndicesOp,
  topkMaskOp,
  unsqueezeOp,
  viewOp,
  sortOp,
  indexPutOp,
} from './index_ops';
import { type Reduction, nllLossOp } from './loss';
import { groupedLinearOp, linearComposite, matmulOp, selectMatmulOp, solveTriangularOp } from './matmul';
import { arangeOp, causalMaskOp } from './creation';
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

export interface PadOptions extends OpOptions {
  value?: number;
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
  cos(input: Node, options?: OpOptions): Node;
  sin(input: Node, options?: OpOptions): Node;
  log(input: Node, options?: OpOptions): Node;
  rsub(input: Node, other: Node | number, options?: OpOptions): Node;
  pow(input: Node | number, exponent: Node | number, options?: OpOptions): Node;
  arange(end: number | string, options?: OpOptions): Node;
  arange(start: number, end: number | string, step?: number, options?: OpOptions): Node;
  causal_mask(size: number | string, dtype: Dtype, options?: OpOptions): Node;
  linear(input: Node, weight: Node, bias?: Node | null, options?: OpOptions): Node;
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
  repeat_kv(input: Node, n_rep: number | string, options?: OpOptions): Node;
  contiguous(input: Node, options?: OpOptions): Node;
  causal_conv1d(input: Node, weight: Node, options?: ConvOptions): Node;
  remainder(input: Node, other: Node | number, options?: OpOptions): Node;
  floor_divide(input: Node, other: Node | number, options?: OpOptions): Node;
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
  where(condition: Node, input: Node, other: Node | number, options?: OpOptions): Node;
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
  pad(input: Node, pad: (number | string)[], options?: PadOptions): Node;
  pad(input: Node, ...pad: (number | string)[]): Node;
  triu(input: Node, diagonal?: number | TriuOptions, options?: TriuOptions): Node;
  masked_fill(input: Node, mask: Node, value: number, options?: OpOptions): Node;
  expand(input: Node, sizes: (number | string)[], options?: OpOptions): Node;
  expand(input: Node, ...sizes: (number | string)[]): Node;
  reshape(input: Node, shape: (number | string)[], options?: OpOptions): Node;
  reshape(input: Node, ...shape: (number | string)[]): Node;
  solve_triangular(A: Node, B: Node, options: SolveTriangularOptions): Node;
  lt(input: Node, other: Node | number, options?: OpOptions): Node;
  bitwise_or(input: Node, other: Node | number, options?: OpOptions): Node;
  logical_not(input: Node, options?: OpOptions): Node;
  copysign(input: Node, other: Node | number, options?: OpOptions): Node;
  sort(input: Node, dim?: number, options?: OpOptions): Node;
  scatter(input: Node, dim: number, index: Node, src: Node | number, options?: OpOptions): Node;
  index_put(input: Node, mask: Node, values: Node, options?: OpOptions): Node;
}

export const builtinOps: readonly AnyOpDef[] = [
  addOp,
  subOp,
  rsubOp,
  mulOp,
  divOp,
  negOp,
  cosOp,
  expOp,
  sinOp,
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
  floorDivideOp,
  bitwiseXorOp,
  bitwiseAndOp,
  eqOp,
  geOp,
  absOp,
  signOp,
  clampMinOp,
  repeatOp,
  cummaxOp,
  ltOp,
  bitwiseOrOp,
  logicalNotOp,
  copysignOp,
  sortOp,
  scatterOp,
  indexPutOp,
  expandOp,
  reshapeOp,
  repeatKvOp,
  contiguousOp,
  arangeOp,
  causalMaskOp,
];

export const builtinComposites: readonly CompositeDef[] = [linearComposite];

for (const def of builtinOps) defineOp(def);
for (const def of builtinComposites) defineComposite(def);
