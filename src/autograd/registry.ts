import { type Dtype, isIntegerDtype, promoteDtypes } from './dtype';
import type { AnyOpDef, CompositeDef, OpDef } from './op';

export const RESERVED_GRAPH_NAMES: ReadonlySet<string> = new Set([
  'dims',
  'real',
  'dtype',
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
  'detachedKey',
  'detached',
  'realDerivations',
  'derivedSizes',
  'realLayout',
  'copies',
  'realSizes',
  'realShapes',
  'realCopies',
  'isDetached',
  'detachedOps',
  'constructor',
  'input',
  'param',
  'label',
  'buffer',
  'split',
  'loss',
  'findLoss',
  'addDim',
  'node',
  'subscribe',
  'batch',
  'reset',
  'jacobian',
  'apply',
  'compose',
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
  'phase',
  'functions',
  'owner',
  'anchors',
  'chip',
  'copiesNow',
  'backwardPhase',
  'defineFunction',
  'functionOf',
  'markOnChip',
  'onChip',
  'sealed',
  'gradEdges',
  'runFunction',
  'valueOf',
  'byName',
]);

const registry = new Map<string, AnyOpDef>();
const composites = new Map<string, CompositeDef>();

export function defineOp<A extends object>(def: OpDef<A>): void {
  if (registry.has(def.name) || composites.has(def.name)) throw new Error(`op '${def.name}' is already defined`);
  if (RESERVED_GRAPH_NAMES.has(def.name)) throw new Error(`op '${def.name}' is a reserved graph member name`);
  for (const t of def.tensors) {
    if (!def.args.includes(t)) throw new Error(`op '${def.name}': tensor '${t}' is not in args`);
  }
  if (def.differentiable && def.differentiable.length !== def.tensors.length) {
    throw new Error(`op '${def.name}': differentiable must have one entry per tensor`);
  }
  if (def.list !== undefined && def.tensors[def.tensors.length - 1] !== def.list) {
    throw new Error(`op '${def.name}': list '${def.list}' must be the last tensor argument`);
  }
  for (const s of def.scalars ?? []) {
    if (!def.tensors.includes(s)) throw new Error(`op '${def.name}': scalar '${s}' is not a tensor argument`);
  }
  for (const s of def.sizes ?? []) {
    if ((!def.args.includes(s) && !(def.defaults !== undefined && s in def.defaults)) || def.tensors.includes(s)) throw new Error(`op '${def.name}': size '${s}' is not a non-tensor argument`);
  }
  for (const s of def.sameDtype ?? []) {
    if (!def.tensors.includes(s) || s === def.list) throw new Error(`op '${def.name}': sameDtype '${s}' is not a single tensor argument`);
  }
  registry.set(def.name, def);
}

export function defineComposite(def: CompositeDef): void {
  if (registry.has(def.name) || composites.has(def.name)) throw new Error(`op '${def.name}' is already defined`);
  if (RESERVED_GRAPH_NAMES.has(def.name)) throw new Error(`op '${def.name}' is a reserved graph member name`);
  for (const t of def.optional) {
    if (!def.tensors.includes(t)) throw new Error(`op '${def.name}': optional '${t}' is not a tensor argument`);
  }
  composites.set(def.name, def);
}

export function compositeDef(name: string): CompositeDef | undefined {
  return composites.get(name);
}

export interface BoundOp<T> {
  inputs: T[];
  attrs: Record<string, unknown>;
  rest: Record<string, unknown>;
}

export function bindOp<T>(def: AnyOpDef, args: readonly unknown[], options: Readonly<Record<string, unknown>>, isTensor: (value: unknown) => value is T): BoundOp<T> {
  const positional = [...args];
  const restAt = def.rest === undefined ? -1 : def.args.indexOf(def.rest);
  if (restAt >= 0 && positional.length > restAt && !(positional.length === restAt + 1 && Array.isArray(positional[restAt]))) {
    positional.splice(restAt, positional.length - restAt, positional.slice(restAt));
  }
  if (positional.length > def.args.length) {
    throw new Error(`${def.name}: takes at most ${def.args.length} positional arguments, got ${positional.length}`);
  }
  const attrs: Record<string, unknown> = { ...(def.defaults ?? {}) };
  const rest: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(options)) {
    if (def.args.includes(key) && !def.tensors.includes(key)) {
      if (positional[def.args.indexOf(key)] !== undefined) throw new Error(`${def.name}: argument '${key}' is given both by position and by keyword`);
      attrs[key] = value;
    } else if (def.tensors.includes(key)) {
      throw new Error(`${def.name}: tensor argument '${key}' must be given by position`);
    } else if (def.defaults !== undefined && key in def.defaults) attrs[key] = value;
    else rest[key] = value;
  }
  const inputs: T[] = [];
  def.args.forEach((argName, i) => {
    const given = positional[i];
    if (argName === def.list) {
      if (!Array.isArray(given) || given.length === 0 || !given.every(isTensor)) {
        throw new Error(`${def.name}: argument '${argName}' must be a non-empty list of tensors`);
      }
      inputs.push(...given);
    } else if (def.tensors.includes(argName)) {
      const scalar = def.scalars?.includes(argName) ?? false;
      if (scalar && typeof given === 'number') {
        attrs[argName] = given;
        return;
      }
      if (!isTensor(given)) throw new Error(`${def.name}: argument '${argName}' must be a tensor${scalar ? ' or a number' : ''}`);
      inputs.push(given);
    } else if (given !== undefined) {
      attrs[argName] = given;
    } else if (!(argName in attrs)) {
      throw new Error(`${def.name}: missing argument '${argName}'`);
    }
  });
  return { inputs, attrs, rest };
}

export function bindComposite<T>(def: CompositeDef, args: readonly unknown[], options: Readonly<Record<string, unknown>>, isTensor: (value: unknown) => value is T): { inputs: (T | null)[]; rest: Record<string, unknown> } {
  if (args.length > def.args.length) throw new Error(`${def.name}: takes at most ${def.args.length} positional arguments, got ${args.length}`);
  const given = new Map<string, unknown>(args.map((value, i) => [def.args[i], value]));
  const rest: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(options)) {
    if (!def.args.includes(key)) rest[key] = value;
    else if (given.has(key)) throw new Error(`${def.name}: argument '${key}' is given both by position and by keyword`);
    else given.set(key, value);
  }
  const inputs = def.tensors.map((name) => {
    const value = given.get(name);
    if (isTensor(value)) return value;
    if ((value === undefined || value === null) && def.optional.includes(name)) return null;
    throw new Error(`${def.name}: argument '${name}' must be a tensor${def.optional.includes(name) ? ' or None' : ''}`);
  });
  return { inputs, rest };
}

function tensorInput<T>(def: AnyOpDef, inputs: readonly T[], attrs: Record<string, unknown>, name: string): T | undefined {
  const present = def.tensors.filter((t) => !(t in attrs));
  const listLength = def.list === undefined ? 0 : inputs.length - (present.length - 1);
  let at = 0;
  for (const t of present) {
    if (t === name) return inputs[at];
    at += t === def.list ? listLength : 1;
  }
  return undefined;
}

export function outputDtype(def: AnyOpDef, inputs: readonly { name: string; dtype: Dtype }[], attrs: Record<string, unknown>, fallback: Dtype, where: string, given?: Dtype): Dtype {
  const same = (def.sameDtype ?? []).map((name) => tensorInput(def, inputs, attrs, name)).filter((n) => n !== undefined);
  if (new Set(same.map((n) => n.dtype)).size > 1) {
    throw new Error(`${where}: expected inputs of the same dtype, got ${same.map((n) => `${n.name} (${n.dtype})`).join(' and ')}`);
  }
  if (given !== undefined) return given;
  if (typeof def.dtype === 'function') return def.dtype(attrs);
  const promoted = def.dtype ?? promoteDtypes(inputs.map((n) => n.dtype), fallback, where);
  if (def.integral !== undefined && isIntegerDtype(promoted)) return def.integral;
  return def.floating === true && isIntegerDtype(promoted) ? fallback : promoted;
}

export function inputDifferentiable(def: AnyOpDef, input: number): boolean {
  if (def.differentiable === undefined) return true;
  return def.differentiable[Math.min(input, def.tensors.length - 1)];
}

export function inputName(def: AnyOpDef, input: number): string {
  const last = def.tensors.length - 1;
  if (def.list === undefined || input < last) return def.tensors[input];
  return `${def.list}[${input - last}]`;
}

export function opDef(name: string): AnyOpDef | undefined {
  return registry.get(name);
}
