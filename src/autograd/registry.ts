import type { AnyOpDef, OpDef } from './op';

const registry = new Map<string, AnyOpDef>();

export function defineOp<A extends object>(def: OpDef<A>): void {
  if (registry.has(def.name)) throw new Error(`op '${def.name}' is already defined`);
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

export function opNames(): string[] {
  return [...registry.keys()];
}
