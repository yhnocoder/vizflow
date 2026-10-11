import { type Graph, type Node, graph } from '../../autograd';
import type { LayoutEntry } from '../../scene3d';
import type { BackwardOverride } from '../../steps';
import type { Nested } from '../../tensor';
import type { ModuleDef } from '../load';
import { type Where, child, fail } from '../source';
import { type Env, type FormulaRec, type ModuleRec, type Num, Scope } from './types';
import { isRecord, record } from './util';

const DEFAULT_SEED = 1;
const INIT_DIGITS = 2;

function rng(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class CompilerState {
  readonly g: Graph;
  readonly nodes = new Map<string, Node>();
  readonly modules: ModuleRec[] = [];
  readonly moduleStack: ModuleRec[] = [];
  readonly backward: { node: Node; text: BackwardOverride; where: Where }[] = [];
  readonly pendingPatches: (() => void)[][] = [];
  readonly formulas: FormulaRec[] = [];
  readonly switchedOffSteps = new Set<string>();
  readonly root: Scope;
  protected readonly rangeCounts = new WeakMap<Node[], Num>();
  private readonly values: Record<string, unknown>;
  private readonly usedValues = new Set<string>();
  private readonly random: () => number;
  private readonly real: boolean;
  private readonly namedDims = new Map<string, string>();
  private readonly derivedDims = new Set<string>();
  private readonly buffers = new Map<string, Node>();
  private readonly owners = new Map<Node, LayoutEntry[]>();

  constructor(
    readonly page: Record<string, unknown>,
    readonly pageWhere: Where,
    rootDef: ModuleDef,
    rootDims: Map<string, Num>,
  ) {
    const dims: Record<string, number> = {};
    const real: Record<string, number | string> = { ...(isRecord(page.real) ? (page.real as Record<string, number | string>) : {}) };
    for (const [name, value] of rootDims) {
      dims[name] = value.toy;
      if (!(name in real)) real[name] = value.real;
      this.namedDims.set(`${value.toy}:${value.real}`, name);
    }
    this.real = page.real !== undefined;
    const dtype = page.dtype;
    if (dtype !== undefined && typeof dtype !== 'string') fail(child(pageWhere, page, 'dtype'), 'dtype must be a string');
    try {
      this.g = graph({ dims, ...(this.real ? { real } : {}), ...(dtype === undefined ? {} : { dtype }) });
    } catch (error) {
      fail(child(pageWhere, page, 'dtype'), (error as Error).message);
    }
    this.values = record(page.values, child(pageWhere, page, 'values'), 'values');
    const init = record(page.init, child(pageWhere, page, 'init'), 'init');
    this.random = rng(typeof init.seed === 'number' ? init.seed : DEFAULT_SEED);
    this.root = new Scope(rootDef, '', []);
    for (const [name, value] of rootDims) this.root.dims.set(name, { ...value, dim: name, derive: (sizes) => sizes[name] });
  }

  dimSpec(value: Num): number | string {
    if (!this.real) return value.toy;
    if (value.dim !== undefined) return value.dim;
    if (value.toy === value.real && value.derive === undefined) return value.toy;
    const key = `${value.toy}:${value.real}`;
    const name = this.namedDims.get(key);
    if (name !== undefined && (value.derive === undefined || this.derivedDims.has(name))) return name;
    const derived = value.derive === undefined ? key : this.derivedName(key);
    this.g.addDim(derived, value.toy, value.real, value.derive);
    if (value.derive !== undefined) this.derivedDims.add(derived);
    if (name === undefined) this.namedDims.set(key, derived);
    return derived;
  }

  private derivedName(key: string): string {
    if (this.g.dims[key] === undefined) return key;
    let n = 2;
    while (this.g.dims[`${key}#${n}`] !== undefined) n++;
    return `${key}#${n}`;
  }

  checkValuesUsed(): void {
    for (const key of Object.keys(this.values)) {
      if (!this.usedValues.has(key)) fail(child(this.pageWhere, this.values, key), `values: '${key}' is not a weight, buffer or input of this page`);
    }
  }

  takeValue(path: string): Nested | undefined {
    if (!(path in this.values)) return undefined;
    this.usedValues.add(path);
    return exactStrings(this.values[path]);
  }

  initValues(kind: string, shape: number[]): Nested {
    const size = shape.reduce((a, b) => a * b, 1);
    const round = (x: number) => Math.round(x * 10 ** INIT_DIGITS) / 10 ** INIT_DIGITS;
    const flat = Array.from({ length: size }, () => {
      if (kind === 'ones') return 1;
      if (kind === 'zeros') return 0;
      if (kind === 'normal') {
        const u = 1 - this.random();
        const v = this.random();
        return round(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
      }
      return round(this.random() * 2 - 1);
    });
    const nest = (offset: number, dims: number[]): Nested =>
      dims.length === 0 ? flat[offset] : Array.from({ length: dims[0] }, (_, i) => nest(offset + i * dims.slice(1).reduce((a, b) => a * b, 1), dims.slice(1)));
    return nest(0, shape);
  }

  bufferNode(key: string, make: () => Node): Node {
    const existing = this.buffers.get(key);
    if (existing !== undefined) return existing;
    const created = make();
    this.buffers.set(key, created);
    return created;
  }

  setOwner(node: Node, entries: LayoutEntry[]): void {
    this.owners.set(node, entries);
  }

  ownerOf(node: Node): LayoutEntry[] | undefined {
    return this.owners.get(node);
  }

  scopeLabel(scope: Scope): string {
    return scope.path === '' ? 'the page' : `'${scope.path}' (${scope.def.name})`;
  }

  fail(env: Env, message: string, offset?: number): never {
    return fail(env.where, message, offset);
  }
}

function exactStrings(value: unknown): Nested {
  if (Array.isArray(value)) return value.map(exactStrings);
  if (typeof value === 'string' && /^-?\d+$/.test(value)) return BigInt(value);
  return value as Nested;
}
