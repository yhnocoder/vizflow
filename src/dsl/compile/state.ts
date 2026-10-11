import { type Graph, type Mesh, type MeshTopology, type Node, type Placement, type Placements, graph, placements } from '../../autograd';
import type { LayoutEntry } from '../../scene3d';
import type { BackwardOverride } from '../../steps';
import type { Nested } from '../../tensor';
import type { ModuleDef } from '../load';
import { type Where, child, fail } from '../source';
import { type Env, type FormulaRec, type ModuleRec, type Num, Scope } from './types';
import { MESH_AXIS_KEYS } from './keys';
import { checkKeys, isRecord, record } from './util';
import { mulberry32 } from '../../tensor';

const DEFAULT_SEED = 1;
const INIT_DIGITS = 2;


function meshAxis(value: unknown, where: Where): [string, MeshTopology] {
  const entries = isRecord(value) ? Object.entries(value) : [];
  if (entries.length !== 1) fail(where, 'mesh must name exactly one axis and its number of devices, for example { tp: 2 } or { ep: { size: 4, nodes: 2 } }');
  const [axis, spec] = entries[0];
  const at = child(where, value as object, axis);
  const record = isRecord(spec) ? spec : { size: spec };
  if (isRecord(spec)) checkKeys(spec, MESH_AXIS_KEYS, at);
  const { size, nodes = 1 } = record;
  if (typeof size !== 'number' || !Number.isInteger(size) || size < 2) fail(at, `mesh axis '${axis}' needs an integer number of devices of at least 2`);
  if (typeof nodes !== 'number' || !Number.isInteger(nodes) || nodes < 1 || size % nodes !== 0) fail(at, `mesh axis '${axis}' has ${size} devices, so nodes must be a positive integer that divides ${size}`);
  return [axis, { size, nodes }];
}

function meshOf(page: Record<string, unknown>, where: Where): Mesh | undefined {
  if (page.mesh === undefined) {
    if (isRecord(page.real) && page.real.mesh !== undefined) fail(child(child(where, page, 'real'), page.real, 'mesh'), 'real.mesh needs a mesh on the page');
    return undefined;
  }
  const [axis, drawn] = meshAxis(page.mesh, child(where, page, 'mesh'));
  if (!isRecord(page.real) || page.real.mesh === undefined) return { axis, ...drawn, real: drawn };
  const realWhere = child(child(where, page, 'real'), page.real, 'mesh');
  const [realAxis, real] = meshAxis(page.real.mesh, realWhere);
  if (realAxis !== axis) fail(realWhere, `real.mesh names axis '${realAxis}', but mesh names '${axis}'`);
  return { axis, ...drawn, real };
}

export class CompilerState {
  readonly g: Graph;
  readonly nodes = new Map<string, Node>();
  readonly modules: ModuleRec[] = [];
  readonly moduleStack: ModuleRec[] = [];
  readonly backward: { node: Node; text: BackwardOverride; where: Where }[] = [];
  readonly pendingPatches: (() => void)[][] = [];
  readonly formulas: FormulaRec[] = [];
  readonly caches: { nodes: Node[]; members: Node[] }[] = [];
  readonly root: Scope;
  protected readonly rangeCounts = new WeakMap<Node[], Num>();
  private readonly values: Record<string, unknown>;
  private readonly usedValues = new Set<string>();
  private readonly random: () => number;
  private readonly real: boolean;
  private readonly namedDims = new Map<string, string>();
  private readonly buffers = new Map<string, Node>();
  private readonly owners = new Map<Node, LayoutEntry[]>();
  readonly mesh: Mesh | undefined;
  readonly leafPlacements = new Map<Node, Placement>();
  readonly placements: Placements | undefined;

  constructor(
    readonly page: Record<string, unknown>,
    readonly pageWhere: Where,
    rootDef: ModuleDef,
    rootDims: Map<string, Num>,
  ) {
    const dims: Record<string, number> = {};
    const real: Record<string, number | string> = { ...(isRecord(page.real) ? (page.real as Record<string, number | string>) : {}) };
    delete real.mesh;
    for (const [name, value] of rootDims) {
      dims[name] = value.toy;
      if (!(name in real)) real[name] = value.real;
      this.namedDims.set(`${value.toy}:${value.real}`, name);
    }
    this.real = page.real !== undefined;
    const dtype = page.dtype;
    if (dtype !== undefined && typeof dtype !== 'string') fail(child(pageWhere, page, 'dtype'), 'dtype must be a string');
    if (page.rounding !== undefined && typeof page.rounding !== 'boolean') fail(child(pageWhere, page, 'rounding'), 'rounding must be true or false');
    try {
      this.g = graph({ dims, ...(this.real ? { real } : {}), ...(dtype === undefined ? {} : { dtype }), rounding: page.rounding === true });
    } catch (error) {
      fail(child(pageWhere, page, 'dtype'), (error as Error).message);
    }
    this.values = record(page.values, child(pageWhere, page, 'values'), 'values');
    const init = record(page.init, child(pageWhere, page, 'init'), 'init');
    this.random = mulberry32(typeof init.seed === 'number' ? init.seed : DEFAULT_SEED);
    this.root = new Scope(rootDef, '', []);
    this.mesh = meshOf(page, pageWhere);
    this.placements = this.mesh === undefined ? undefined : placements(this.g, this.mesh, this.leafPlacements);
    for (const [name, value] of rootDims) this.root.dims.set(name, value);
  }

  dimSpec(value: Num): number | string {
    if (!this.real || value.toy === value.real) return value.toy;
    const key = `${value.toy}:${value.real}`;
    const name = this.namedDims.get(key);
    if (name !== undefined) return name;
    this.g.addDim(key, value.toy, value.real);
    this.namedDims.set(key, key);
    return key;
  }

  checkValuesUsed(): void {
    for (const key of Object.keys(this.values)) {
      if (!this.usedValues.has(key)) fail(child(this.pageWhere, this.values, key), `values: '${key}' is not a weight, buffer or input of this page`);
    }
  }

  takeValue(path: string): Nested | undefined {
    if (!(path in this.values)) return undefined;
    this.usedValues.add(path);
    return this.values[path] as Nested;
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
