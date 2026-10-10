import type { Node, OpExplain } from '../autograd';
import { explainOf } from './registry';

export function explainDefOf(op: Node): OpExplain {
  const def = op.op === undefined ? undefined : explainOf(op.op);
  if (def === undefined) throw new Error(`explain: op '${op.op}' of node '${op.name}' has no explain definition`);
  return def;
}
