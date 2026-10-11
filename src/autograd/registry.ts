import type { AnyOpDef, OpDef } from './op';

export const RESERVED_GRAPH_NAMES: ReadonlySet<string> = new Set([
  'dims',
  'real',
  'nodes',
  'version',
  'self',
  'states',
  'realDims',
  'check',
  'subscribers',
  'opCounters',
  'lossNode',
  'backwardValid',
  'batchDepth',
  'pending',
  'touched',
  'constructor',
  'input',
  'param',
  'label',
  'buffer',
  'split',
  'loss',
  'findLoss',
  'node',
  'subscribe',
  'batch',
  'reset',
  'jacobian',
  'apply',
  'leaf',
  'createOp',
  'addNode',
  'autoName',
  'ownNode',
  'tensorFor',
  'gradOf',
  'setLeaf',
  'replaceLeaf',
  'rollbackLeaves',
  'transposeOf',
  'recompute',
  'ensureBackward',
  'jacobianBlock',
  'computeBackward',
  'propagate',
]);

const registry = new Map<string, AnyOpDef>();

export function defineOp<A extends object>(def: OpDef<A>): void {
  if (registry.has(def.name)) throw new Error(`op '${def.name}' is already defined`);
  if (RESERVED_GRAPH_NAMES.has(def.name)) throw new Error(`op '${def.name}' is a reserved graph member name`);
  for (const t of def.tensors) {
    if (!def.args.includes(t)) throw new Error(`op '${def.name}': tensor '${t}' is not in args`);
  }
  if (def.differentiable && def.differentiable.length !== def.tensors.length) {
    throw new Error(`op '${def.name}': differentiable must have one entry per tensor`);
  }
  if (def.list !== undefined && (def.tensors.length !== 1 || def.tensors[0] !== def.list)) {
    throw new Error(`op '${def.name}': list '${def.list}' must be the only tensor argument`);
  }
  if (def.list !== undefined && def.differentiable !== undefined) {
    throw new Error(`op '${def.name}': differentiable cannot be combined with list`);
  }
  for (const s of def.scalars ?? []) {
    if (!def.tensors.includes(s)) throw new Error(`op '${def.name}': scalar '${s}' is not a tensor argument`);
  }
  for (const s of def.sizes ?? []) {
    if ((!def.args.includes(s) && !(def.defaults !== undefined && s in def.defaults)) || def.tensors.includes(s)) throw new Error(`op '${def.name}': size '${s}' is not a non-tensor argument`);
  }
  registry.set(def.name, def);
}

export function opDef(name: string): AnyOpDef | undefined {
  return registry.get(name);
}
