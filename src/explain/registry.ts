import { type OpExplain, opDef } from '../autograd';

const registry = new Map<string, OpExplain>();

export function registerExplain(opName: string, def: OpExplain): void {
  if (registry.has(opName)) throw new Error(`explain for op '${opName}' is already registered`);
  registry.set(opName, def);
}

export function explainOf(opName: string): OpExplain | undefined {
  return registry.get(opName) ?? opDef(opName)?.explain;
}
