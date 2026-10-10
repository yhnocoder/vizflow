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
  'loss',
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
  'resolveLoss',
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
  registry.set(def.name, def);
}

export function opDef(name: string): AnyOpDef | undefined {
  return registry.get(name);
}
