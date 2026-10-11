import type { OpOptions } from '../graph';
import type { Node } from '../node';
import type { AnyOpDef } from '../op';
import { defineOp } from '../registry';
import {
  addOp,
  divOp,
  expOp,
  logOp,
  mulOp,
  negOp,
  powOp,
  rsqrtOp,
  sigmoidOp,
  siluOp,
  sqrtOp,
  subOp,
  transposeOp,
} from './elementwise';
import { catOp, embeddingOp, gatherOp, narrowOp } from './index_ops';
import { type Reduction, nllLossOp } from './loss';
import { matmulOp } from './matmul';
import { type Dim, logsumexpOp, maxOp, meanOp, sumOp } from './reduce';
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

export interface CatOptions extends OpOptions {
  dim?: number;
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
  rsqrt(input: Node, options?: OpOptions): Node;
  sqrt(input: Node, options?: OpOptions): Node;
  sigmoid(input: Node, options?: OpOptions): Node;
  silu(input: Node, options?: OpOptions): Node;
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
  sigmoidOp,
  siluOp,
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
];

for (const def of builtinOps) defineOp(def);
