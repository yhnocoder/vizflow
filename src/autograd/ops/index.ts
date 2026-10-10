import type { OpOptions } from '../graph';
import type { Node } from '../node';
import type { AnyOpDef } from '../op';
import { defineOp } from '../registry';
import { addOp, divOp, expOp, logOp, mulOp, negOp, subOp, transposeOp } from './elementwise';

export interface BuiltinOps {
  add(input: Node, other: Node, options?: OpOptions): Node;
  sub(input: Node, other: Node, options?: OpOptions): Node;
  mul(input: Node, other: Node, options?: OpOptions): Node;
  div(input: Node, other: Node, options?: OpOptions): Node;
  neg(input: Node, options?: OpOptions): Node;
  exp(input: Node, options?: OpOptions): Node;
  log(input: Node, options?: OpOptions): Node;
  transpose(input: Node, dim0: number, dim1: number, options?: OpOptions): Node;
}

export const builtinOps: readonly AnyOpDef[] = [addOp, subOp, mulOp, divOp, negOp, expOp, logOp, transposeOp];

for (const def of builtinOps) defineOp(def);
