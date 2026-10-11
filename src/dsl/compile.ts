import { type Graph, Node, graph, opDef } from '../autograd';
import { explainOf, trailingSubscript } from '../explain';
import type { CameraOptions, LayoutEntry, LayoutTree } from '../scene3d';
import { type BackwardOverride, type ForwardStep, type PageModule, type StepModule, type StepsSpec, isLeafLike } from '../steps';
import { type Nested, inferShape } from '../tensor';
import type { PanelItem, SectionBlock, SectionItem, SectionRow } from '../ui';
import { checkFigure, figureNames, figureTag } from '../ui/figure-spec';
import { type Ast, ParseError, parseExpr, parseRange } from './expr';
import { type LoadOptions, Loader, type ModuleDef } from './load';
import { sectionRows } from './section-md';
import { DslError, type Where, child, fail } from './source';
import { OP_TEX, addMark, defaultTex, elemOf, markSteps, numberTex, opTex } from './tex';

export interface CompiledPage {
  graph: Graph;
  steps: StepsSpec;
  tree: LayoutTree;
  modules: PageModule[];
  panels: PanelItem[];
  sections: SectionItem[];
  nodes: Map<string, Node>;
  sequence?: PageSequence;
  layout?: unknown;
  page: { title: string; lede?: string; hint?: string; camera?: CameraOptions };
}

export interface PageSequence {
  tokens: string[];
  length: number;
  get(): number[];
  set(ids: number[]): void;
}

interface ModuleRec {
  id: string;
  parent: string | null;
  title: string;
  subtitle: string;
  formula?: string;
  fixed: boolean;
  folded: boolean;
  split: boolean;
  nodes: Node[];
  fold: { forward: BackwardOverride; backward: BackwardOverride };
  decl: Record<string, unknown>;
  env: Env;
  defaults?: { decl: Record<string, unknown>; env: Env };
}

interface Num {
  toy: number;
  real: number;
}

interface OpValue {
  op: string;
}

interface BufferValue {
  buffer: true;
  shape: Num[];
  values: Nested;
}

type ParamType = 'int' | 'float' | 'dim' | 'op' | 'tex' | 'text';
type Value = Node | Num | Node[] | string | boolean | OpValue | BufferValue;
type Resolved = Value | Scope;

interface Make {
  path: string;
  tex: string;
  anon: Node[];
  counter: number;
}

interface Env {
  scope: Scope;
  loops: Map<string, number>;
  where: Where;
  buffers?: boolean;
}

interface StepRec {
  refs: { owner: string; id: string }[];
  step: ForwardStep;
}

type Slot =
  | { kind: 'node'; name: string; node: Node; created: Node[] }
  | { kind: 'child'; def: ModuleDef; steps: StepRec[]; created: Node[]; merge?: Slot }
  | { kind: 'loop'; iterations: Slot[][] }
  | { kind: 'group'; decl: Record<string, unknown>; where: Where; loops: Map<string, number>; slots: Slot[] };

interface Lanes {
  start: number;
  members: { name: string | undefined; created: Node[]; slot: Slot | undefined }[];
  parallel?: { kind: 'parallel'; branches: LayoutTree[] };
}

interface ResolvedStep {
  rec: StepRec;
  show: Node[];
}

const ITEM_KEYS = new Set(['op', 'tex']);
const MODULE_ITEM_KEYS = ['title', 'subtitle', 'fixed', 'folded', 'split', 'fold'];
const INSTANCE_KEYS = new Set(['use', 'params', 'inputs', 'outputs', 'weights', 'suffix', 'tex', 'steps', 'beside', 'inline', 'brace', 'formula', ...MODULE_ITEM_KEYS]);
const GROUP_KEYS = new Set(['body', 'steps', 'beside', 'formula', ...MODULE_ITEM_KEYS]);
const FOR_KEYS = new Set(['for', 'in', 'name', 'brace', 'beside', 'body']);
const MODULE_KEYS = new Set(['module', 'import', 'subtitle', 'formula', 'params', 'inputs', 'outputs', 'weights', 'buffers', 'body', 'steps', 'backward', 'fold']);
const PAGE_KEYS = new Set([
  'title', 'lede', 'hint', 'camera', 'overview', 'leaves', 'import', 'modules', 'subtitle', 'params', 'dims', 'real', 'vocab', 'init', 'inputs', 'sequence',
  'weights', 'buffers', 'values', 'body', 'steps', 'backward', 'panels', 'sections', 'layout',
]);
const STEP_KEYS = new Set(['id', 'show', 'active', 'label', 'title', 'text', 'segment', 'brace', 'backward']);
const BRACE_KEYS = new Set(['title', 'subtitle']);
const PATCH_KEYS = new Set(['title', 'label', 'text', 'append', 'backward']);
const TEXT_KEYS = new Set(['title', 'text', 'label']);
const FOLD_KEYS = new Set(['title', 'text', 'label', 'backward']);
const OVERVIEW_KEYS = new Set(['title', 'text', 'label', 'backward']);
const SEQUENCE_KEYS = new Set(['tokens', 'input', 'label']);
const PARAM_TYPES: ReadonlySet<string> = new Set(['int', 'float', 'dim', 'op', 'tex', 'text']);
const INITS = new Set(['ones', 'zeros', 'normal', 'uniform']);
const DEFAULT_INIT = 'uniform';
const DEFAULT_SEED = 1;
const INIT_DIGITS = 2;
const ROPE_POSITIONS = 24;

class NeedDim {
  constructor(
    readonly name: string,
    readonly value: Num,
  ) {}
}

class Scope {
  readonly entries = new Map<string, Node>();
  readonly children = new Map<string, Scope>();
  readonly params = new Map<string, { type: ParamType; value: Num | string | OpValue }>();
  readonly dims = new Map<string, Num>();
  readonly inputTex = new Map<string, string>();
  readonly anon = new Map<Node, Node[]>();
  readonly outputTex = new Map<string, string>();
  readonly texOverride = new Map<string, string>();
  readonly groups = new Set<string>();

  constructor(
    readonly def: ModuleDef,
    readonly path: string,
    readonly marks: string[],
  ) {}

  local(key: string): Node | Scope | undefined {
    return this.entries.get(key) ?? this.children.get(key);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNum(value: unknown): value is Num {
  return isRecord(value) && typeof value.toy === 'number' && typeof value.real === 'number';
}

function isOp(value: unknown): value is OpValue {
  return isRecord(value) && typeof value.op === 'string' && !(value instanceof Node);
}

function num(n: number): Num {
  return { toy: n, real: n };
}

function join(path: string, name: string): string {
  return path === '' ? name : `${path}.${name}`;
}

function list<T>(value: T | T[] | undefined): T[] {
  return value === undefined ? [] : Array.isArray(value) ? value : [value];
}

function checkKeys(obj: Record<string, unknown>, allowed: ReadonlySet<string>, where: Where): void {
  for (const key of Object.keys(obj)) if (!allowed.has(key)) fail(child(where, obj, key), `unknown key '${key}'`);
}

function record(value: unknown, where: Where, what: string): Record<string, unknown> {
  if (value === undefined) return {};
  if (!isRecord(value)) fail(where, `${what} must be a mapping`);
  return value;
}

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

const BUFFER_FUNCTIONS: Record<string, (args: Num[]) => { shape: Num[]; values: Nested }> = {
  rope_cos: ([T, dh, theta]) => ({ shape: [T, dh], values: ropeTable(T.toy, dh.toy, theta.toy, Math.cos) }),
  rope_sin: ([T, dh, theta]) => ({ shape: [T, dh], values: ropeTable(T.toy, dh.toy, theta.toy, Math.sin) }),
  causal_mask: ([T]) => ({
    shape: [T, T],
    values: [...Array(T.toy).keys()].map((t) => [...Array(T.toy).keys()].map((s) => (s > t ? -Infinity : 0))),
  }),
};

const BUFFER_ARITY: Record<string, number> = { rope_cos: 3, rope_sin: 3, causal_mask: 1 };

function ropeTable(T: number, dh: number, theta: number, fn: (x: number) => number): number[][] {
  const inv = [...Array(dh / 2).keys()].map((i) => 1 / theta ** ((2 * i) / dh));
  return [...Array(T).keys()].map((t) => [...Array(dh).keys()].map((c) => fn(t * inv[c % (dh / 2)])));
}

function combineLabels(a: string | undefined, b: string | undefined): string | undefined {
  if (a === undefined || b === undefined || a === b) return a ?? b;
  const math = /^\$([^$]*)\$$/;
  const ma = math.exec(a);
  const mb = math.exec(b);
  return ma !== null && mb !== null ? `$${ma[1]},\\ ${mb[1]}$` : `${a}, ${b}`;
}

function reshapeNested(values: Nested, shape: number[]): Nested {
  const flat: number[] = [];
  const collect = (v: Nested): void => {
    if (Array.isArray(v)) v.forEach(collect);
    else flat.push(v as number);
  };
  collect(values);
  const nest = (offset: number, dims: number[]): Nested => {
    if (dims.length === 0) return flat[offset];
    const stride = dims.slice(1).reduce((a, b) => a * b, 1);
    return Array.from({ length: dims[0] }, (_, i) => nest(offset + i * stride, dims.slice(1)));
  };
  return nest(0, shape);
}

function matrixTemplate(name: string, shape: readonly number[], format: string): string {
  if (shape.length === 0) return `{{${name}${format}}}`;
  const pmatrix = (lines: string[][]) => `\\begin{pmatrix}${lines.map((line) => line.map((c) => ` ${c} `).join('&')).join('\\\\')}\\end{pmatrix}`;
  const grid = (rows: number, cols: number, cell: (r: number, c: number) => string) =>
    [...Array(rows).keys()].map((r) => [...Array(cols).keys()].map((c) => `{{${name}[${cell(r, c)}]${format}}}`));
  if (shape.length === 1) return pmatrix(grid(1, shape[0], (_r, c) => `${c}`));
  if (shape.length === 2) {
    const [rows, cols] = shape;
    return pmatrix(grid(rows, cols, (r, c) => (cols === 1 ? `${r}` : rows === 1 ? `${c}` : `${r}, ${c}`)));
  }
  const middle = shape.slice(1, -1);
  const heads = middle.reduce((a, b) => a * b, 1);
  const unravel = (h: number) => middle.map((_, d) => Math.floor(h / middle.slice(d + 1).reduce((a, b) => a * b, 1)) % middle[d]);
  const blocks = [...Array(heads).keys()].map((h) => {
    const index = unravel(h);
    const block = pmatrix(grid(shape[0], shape[shape.length - 1], (r, c) => [r, ...index, c].join(', ')));
    return heads === 1 ? block : `${block}_{h=${index.map((i) => i + 1).join(',')}}`;
  });
  return heads === 1 ? blocks[0] : `\\left[\\,${blocks.join('\\;\\;')}\\,\\right]`;
}

class Compiler {
  readonly g: Graph;
  readonly nodes = new Map<string, Node>();
  readonly modules: ModuleRec[] = [];
  readonly moduleStack: ModuleRec[] = [];
  readonly backward: { node: Node; text: BackwardOverride; where: Where }[] = [];
  private readonly values: Record<string, unknown>;
  private readonly usedValues = new Set<string>();
  private readonly random: () => number;
  private readonly real: boolean;
  private readonly namedDims = new Map<string, string>();
  private readonly buffers = new Map<string, Node>();
  private readonly owners = new Map<Node, LayoutEntry[]>();
  private readonly rangeCounts = new WeakMap<Node[], Num>();

  constructor(
    readonly page: Record<string, unknown>,
    readonly pageWhere: Where,
    readonly rootDims: Map<string, Num>,
    extraDims: Map<string, Num>,
  ) {
    const dims: Record<string, number> = {};
    const real: Record<string, number | string> = { ...(isRecord(page.real) ? (page.real as Record<string, number | string>) : {}) };
    for (const [name, value] of [...rootDims, ...extraDims]) {
      dims[name] = value.toy;
      if (!(name in real)) real[name] = value.real;
      this.namedDims.set(`${value.toy}:${value.real}`, name);
    }
    for (const [name, value] of extraDims) real[name] = value.real;
    this.real = page.real !== undefined;
    this.g = graph(this.real ? { dims, real } : { dims });
    this.values = record(page.values, child(pageWhere, page, 'values'), 'values');
    const init = record(page.init, child(pageWhere, page, 'init'), 'init');
    this.random = rng(typeof init.seed === 'number' ? init.seed : DEFAULT_SEED);
  }

  dimSpec(value: Num): number | string {
    if (!this.real || value.toy === value.real) return value.toy;
    const key = `${value.toy}:${value.real}`;
    const name = this.namedDims.get(key);
    if (name === undefined) throw new NeedDim(key, value);
    return name;
  }

  checkValuesUsed(): void {
    for (const key of Object.keys(this.values)) {
      if (!this.usedValues.has(key)) fail(child(this.pageWhere, this.values, key), `values: '${key}' is not a weight or input of this page`);
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

  lookupFirst(key: string, env: Env): Resolved | undefined {
    const loop = env.loops.get(key);
    if (loop !== undefined) return num(loop);
    const scope = env.scope;
    const local = scope.local(key);
    if (local !== undefined) return local;
    const param = scope.params.get(key);
    if (param !== undefined) return param.value;
    return scope.dims.get(key);
  }

  keyOf(ast: Ast, env: Env, offset: number): { scope: Scope | undefined; key: string } | undefined {
    if (ast.k === 'name') return { scope: undefined, key: ast.name };
    if (ast.k === 'member') {
      const base = this.resolve(ast.obj, env, undefined, offset);
      if (!(base instanceof Scope)) return undefined;
      return { scope: base, key: ast.name };
    }
    if (ast.k === 'index' && ast.end === undefined) {
      const inner = this.keyOf(ast.obj, env, offset);
      if (inner === undefined) return undefined;
      return { scope: inner.scope, key: `${inner.key}[${this.int(ast.index, env, offset)}]` };
    }
    return undefined;
  }

  lookupKey(target: { scope: Scope | undefined; key: string }, env: Env): Resolved | undefined {
    return target.scope === undefined ? this.lookupFirst(target.key, env) : target.scope.local(target.key);
  }

  resolve(ast: Ast, env: Env, make: Make | undefined, offset: number, top = false): Resolved {
    if (ast.k === 'name') {
      const found = this.lookupFirst(ast.name, env);
      if (found === undefined) this.fail(env, `name '${ast.name}' is not defined in ${this.scopeLabel(env.scope)}`, offset + ast.pos);
      return found;
    }
    if (ast.k === 'member') {
      if (ast.name === 'T' && top && make !== undefined) return this.value(ast, env, make, offset, true);
      const base = this.resolve(ast.obj, env, make, offset);
      if (base instanceof Scope) {
        const found = base.local(ast.name);
        if (found === undefined) this.fail(env, `'${ast.name}' is not defined in ${this.scopeLabel(base)}`, offset + ast.pos);
        return found;
      }
      if (base instanceof Node && ast.name === 'T') return this.transposed(base, env, offset + ast.pos);
      this.fail(env, `cannot take '.${ast.name}' here`, offset + ast.pos);
    }
    if (ast.k === 'index') {
      if (ast.end !== undefined) {
        const from = this.int(ast.index, env, offset);
        const to = this.int(ast.end, env, offset);
        const fromNum = this.evaluate(ast.index, env, undefined, offset) as Num;
        const toNum = this.evaluate(ast.end, env, undefined, offset) as Num;
        const nodes: Node[] = [];
        for (let i = from; i <= to; i++) {
          const target = this.keyOf({ k: 'index', obj: ast.obj, index: { k: 'num', v: i, pos: ast.pos }, pos: ast.pos }, env, offset);
          const found = target === undefined ? undefined : this.lookupKey(target, env);
          if (!(found instanceof Node)) this.fail(env, `'${target?.key ?? '?'}' is not a node`, offset + ast.pos);
          nodes.push(found);
        }
        this.rangeCounts.set(nodes, { toy: to - from + 1, real: toNum.real - fromNum.real + 1 });
        return nodes;
      }
      const target = this.keyOf(ast, env, offset);
      const found = target === undefined ? undefined : this.lookupKey(target, env);
      if (found === undefined) this.fail(env, `'${target?.key ?? 'expression'}' is not defined in ${this.scopeLabel(target?.scope ?? env.scope)}`, offset + ast.pos);
      return found;
    }
    return this.value(ast, env, make, offset, top);
  }

  scopeLabel(scope: Scope): string {
    return scope.path === '' ? 'the page' : `'${scope.path}' (${scope.def.name})`;
  }

  transposed(node: Node, env: Env, offset: number): Node {
    try {
      return node.T;
    } catch (error) {
      this.fail(env, (error as Error).message, offset);
    }
  }

  int(ast: Ast, env: Env, offset: number): number {
    const value = this.evaluate(ast, env, undefined, offset);
    if (!isNum(value) || !Number.isInteger(value.toy)) this.fail(env, 'expected an integer', offset + ast.pos);
    return value.toy;
  }

  evaluate(ast: Ast, env: Env, make: Make | undefined, offset: number, top = false): Value {
    const result = this.resolve(ast, env, make, offset, top);
    if (result instanceof Scope) this.fail(env, `'${result.path}' is a module instance, not a value`, offset + ast.pos);
    return result;
  }

  value(ast: Ast, env: Env, make: Make | undefined, offset: number, top: boolean): Value {
    switch (ast.k) {
      case 'num':
        return num(ast.v);
      case 'str':
        return ast.v;
      case 'bool':
        return ast.v;
      case 'neg': {
        const arg = this.evaluate(ast.arg, env, make, offset);
        if (isNum(arg)) return { toy: -arg.toy, real: -arg.real };
        if (arg instanceof Node) return this.apply('mul', [arg, num(-1)], {}, env, make, top, offset + ast.pos);
        return this.fail(env, 'cannot negate this value', offset + ast.pos);
      }
      case 'bin':
        return this.binary(ast, env, make, offset, top);
      case 'call':
        return this.call(ast, env, make, offset, top);
      case 'member': {
        const base = this.evaluate(ast.obj, env, make, offset);
        if (!(base instanceof Node)) this.fail(env, "'.T' needs a node", offset + ast.pos);
        if (make === undefined) this.fail(env, 'this expression creates a node; only references are allowed here', offset + ast.pos);
        return this.create('transpose', [base, num(0), num(1)], {}, env, make, true, offset + ast.pos);
      }
      default:
        return this.fail(env, 'unsupported expression', offset + ast.pos);
    }
  }

  binary(ast: Extract<Ast, { k: 'bin' }>, env: Env, make: Make | undefined, offset: number, top: boolean): Value {
    const a = this.evaluate(ast.a, env, make, offset);
    const b = this.evaluate(ast.b, env, make, offset);
    const at = offset + ast.pos;
    if (isNum(a) && isNum(b)) {
      const both = (fn: (x: number, y: number) => number): Num => ({ toy: fn(a.toy, b.toy), real: fn(a.real, b.real) });
      switch (ast.op) {
        case '+': return both((x, y) => x + y);
        case '-': return both((x, y) => x - y);
        case '*': return both((x, y) => x * y);
        case '/': return both((x, y) => x / y);
        case '//': return both((x, y) => Math.floor(x / y));
        case '**': return both((x, y) => x ** y);
        default: return this.fail(env, "'@' needs two nodes", at);
      }
    }
    if (ast.op === '@') {
      if (!(a instanceof Node) || !(b instanceof Node)) this.fail(env, "'@' needs two nodes", at);
      return this.apply('matmul', [a, b], {}, env, make, top, at);
    }
    if (ast.op === '**') {
      if (!(a instanceof Node) || !isNum(b)) this.fail(env, "'**' needs a node and a number", at);
      return this.apply('pow', [a, b], {}, env, make, top, at);
    }
    if (ast.op === '//') this.fail(env, "'//' only applies to numbers", at);
    const name = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div' }[ast.op];
    if (a instanceof Node && (b instanceof Node || isNum(b))) return this.apply(name, [a, b], {}, env, make, top, at);
    if (isNum(a) && b instanceof Node && (ast.op === '+' || ast.op === '*')) return this.apply(name, [b, a], {}, env, make, top, at);
    return this.fail(env, `'${ast.op}' cannot combine these operands; put the node on the left`, at);
  }

  call(ast: Extract<Ast, { k: 'call' }>, env: Env, make: Make | undefined, offset: number, top: boolean): Value {
    const at = offset + ast.pos;
    if (ast.fn.k !== 'name') this.fail(env, 'only named functions can be called', at);
    const fnName = ast.fn.name;
    const param = env.scope.params.get(fnName);
    const args = ast.args.map((arg) => this.evaluate(arg, env, make, offset));
    const kwargs: Record<string, Value> = {};
    for (const [key, arg] of ast.kwargs) kwargs[key] = this.evaluate(arg, env, make, offset);
    if (env.buffers && fnName in BUFFER_FUNCTIONS) {
      if (args.length !== BUFFER_ARITY[fnName] || ast.kwargs.length > 0 || !args.every(isNum)) {
        this.fail(env, `${fnName} takes ${BUFFER_ARITY[fnName]} numbers`, at);
      }
      return { buffer: true, ...BUFFER_FUNCTIONS[fnName](args as Num[]) };
    }
    if (param !== undefined) {
      if (param.type !== 'op') this.fail(env, `parameter '${fnName}' is not an op`, at);
      return this.apply((param.value as OpValue).op, args, kwargs, env, make, top, at);
    }
    if (opDef(fnName) === undefined) this.fail(env, `unknown function '${fnName}'`, at);
    return this.apply(fnName, args, kwargs, env, make, top, at);
  }

  apply(op: string, args: Value[], kwargs: Record<string, Value>, env: Env, make: Make | undefined, top: boolean, at: number): Node {
    if (make === undefined) this.fail(env, 'this expression creates a node; only references are allowed here', at);
    return this.create(op, args, kwargs, env, make, top, at);
  }

  create(op: string, args: Value[], kwargs: Record<string, Value>, env: Env, make: Make, top: boolean, at: number): Node {
    const def = opDef(op);
    if (def === undefined) this.fail(env, `unknown op '${op}'`, at);
    const convert = (argName: string, value: Value): unknown => {
      if (isNum(value)) return def.sizes?.includes(argName) ? this.dimSpec(value) : value.toy;
      if (isOp(value) || (isRecord(value) && 'buffer' in value)) this.fail(env, `argument '${argName}' of ${op} has the wrong type`, at);
      return value;
    };
    const restAt = def.rest === undefined ? -1 : def.args.indexOf(def.rest);
    const positional = args.map((value, i) => convert(restAt >= 0 && i >= restAt ? def.args[restAt] : (def.args[i] ?? `#${i}`), value));
    const options: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(kwargs)) options[key] = convert(key, value);
    const pieces = def.list === undefined ? undefined : this.rangeCounts.get(args[def.args.indexOf(def.list)] as Node[]);
    if (pieces !== undefined && pieces.real !== pieces.toy && def.defaults !== undefined && 'pieces' in def.defaults && !('pieces' in options)) {
      options.pieces = this.dimSpec(pieces);
    }
    const attrs: Record<string, unknown> = { ...(def.defaults ?? {}), ...options };
    def.args.forEach((argName, i) => {
      if (positional[i] !== undefined && (!def.tensors.includes(argName) || typeof positional[i] === 'number')) attrs[argName] = positional[i];
    });
    if (restAt >= 0 && positional.length > restAt) attrs[def.args[restAt]] = positional.slice(restAt);
    const name = top ? make.path : `${make.path}/${++make.counter}`;
    const operandTex = def.args.flatMap((argName, i) => {
      const value = positional[i];
      if (value instanceof Node) return [value.tex];
      if (Array.isArray(value)) return (value as Node[]).map((n) => n.tex);
      return def.tensors.includes(argName) && typeof value === 'number' ? [numberTex(value)] : [];
    });
    const tex = top ? make.tex : (explainOf(op)?.tex.forward({ out: '', in: operandTex, attrs }) ?? `\\mathrm{${op}}`);
    const elem = top ? elemOf(tex) : `(${elemOf(tex)})`;
    let node: Node;
    try {
      node = (this.g as unknown as Record<string, (...a: unknown[]) => Node>)[op](...positional, { ...options, name, tex, elem });
    } catch (error) {
      if (error instanceof NeedDim || error instanceof DslError) throw error;
      this.fail(env, (error as Error).message, at);
    }
    if (!top) make.anon.push(node);
    return node;
  }

  fail(env: Env, message: string, offset?: number): never {
    return fail(env.where, message, offset);
  }

  expression(text: unknown, env: Env, make: Make | undefined): Value {
    if (typeof text === 'number') return num(text);
    if (typeof text === 'boolean') return text;
    if (typeof text !== 'string') this.fail(env, 'expected an expression');
    let ast: Ast;
    try {
      ast = parseExpr(text);
    } catch (error) {
      if (error instanceof ParseError) this.fail(env, `${error.message} in '${text}'`, error.offset);
      throw error;
    }
    return this.evaluate(ast, env, make, 0, true);
  }

  reference(text: unknown, env: Env): Resolved {
    if (typeof text !== 'string') this.fail(env, 'expected a name');
    let ast: Ast;
    try {
      ast = parseExpr(text);
    } catch (error) {
      if (error instanceof ParseError) this.fail(env, `${error.message} in '${text}'`, error.offset);
      throw error;
    }
    return this.resolve(ast, env, undefined, 0);
  }

  node(text: unknown, env: Env): Node {
    const found = this.reference(text, env);
    if (!(found instanceof Node)) this.fail(env, `'${String(text)}' is not a node`);
    return found;
  }

  nodeList(text: unknown, env: Env): Node[] {
    const found = this.reference(text, env);
    if (found instanceof Node) return [found];
    if (Array.isArray(found)) return found;
    return this.fail(env, `'${String(text)}' is not a node`);
  }

  concreteName(key: string, env: Env): { name: string; base: string; indices: number[] } {
    let ast: Ast;
    try {
      ast = parseExpr(key);
    } catch (error) {
      if (error instanceof ParseError) this.fail(env, `bad name '${key}': ${error.message}`);
      throw error;
    }
    const indices: number[] = [];
    let current = ast;
    while (current.k === 'index' && current.end === undefined) {
      indices.unshift(this.int(current.index, env, 0));
      current = current.obj;
    }
    if (current.k !== 'name') this.fail(env, `'${key}' is not a valid name`);
    return { name: `${current.name}${indices.map((i) => `[${i}]`).join('')}`, base: current.name, indices };
  }

  loopTex(base: string, indices: number[]): string {
    const tex = defaultTex(base);
    return indices.length === 0 ? tex : `${tex}^{(${indices.join(',')})}`;
  }

  ofText(text: string, env: Env): string {
    const accented = text.replace(/\\(hat|bar|tilde|widehat|widetilde|overline|vec|dot)\{\\of\{([^{}]+)\}\}/g, (_match, accent: string, ref: string) => {
      const value = this.ofValue(ref.trim(), env);
      const split = trailingSubscript(value);
      return split === undefined ? `\\${accent}{${value}}` : `\\${accent}{${split.base}}_{${split.inner}}`;
    });
    return accented.replace(/\\of\{([^{}]+)\}/g, (_match, ref: string, offset: number, whole: string) => {
      const value = this.ofValue(ref.trim(), env);
      return /\\[A-Za-z]+$/.test(whole.slice(0, offset)) && /^[A-Za-z]/.test(value) ? ` ${value}` : value;
    });
  }

  ofValue(ref: string, env: Env): string {
    if (ref.endsWith('.def')) {
      const base = ref.slice(0, -4);
      const param = env.scope.params.get(base);
      if (param?.type !== 'op') this.fail(env, `\\of{${ref}}: '${base}' is not an op parameter`);
      const op = (param.value as OpValue).op;
      const def = OP_TEX[op];
      if (def === undefined) this.fail(env, `\\of{${ref}}: op '${op}' has no registered definition`);
      return def.def;
    }
    if (!env.loops.has(ref)) {
      const param = env.scope.params.get(ref);
      if (param !== undefined) {
        if (param.type === 'op') return opTex((param.value as OpValue).op);
        if (param.type === 'float') return numberTex((param.value as Num).toy);
        if (param.type === 'tex') return param.value as string;
        if (param.type === 'text') return this.ofText(param.value as string, env);
        return String((param.value as Num).toy);
      }
      const inputTex = env.scope.inputTex.get(ref);
      if (inputTex !== undefined) return inputTex;
    }
    const found = this.reference(ref, env);
    if (found instanceof Node) return found.tex;
    if (isNum(found)) return String(found.toy);
    if (typeof found === 'string') return found;
    return this.fail(env, `\\of{${ref}} does not name a node, parameter or loop variable`);
  }

  text(text: unknown, env: Env): string {
    if (text === undefined) return '';
    if (typeof text !== 'string') this.fail(env, 'expected text');
    const substituted = this.ofText(text, env);
    return substituted.replace(/\{\{([^{}]*?)\}\}/g, (_m, inner: string) => this.template(inner, env));
  }

  template(source: string, env: Env): string {
    const pipe = source.lastIndexOf('|');
    if (pipe >= 0) {
      const filter = source.slice(pipe + 1).trim();
      const match = /^matrix(?::(\d+))?$/.exec(filter);
      let target = source.slice(0, pipe).trim();
      if (match === null) {
        const [name, ...args] = filter.split(':').map((part) => part.trim());
        if (!figureNames.includes(name)) this.fail(env, `unknown template filter '${filter}' in '{{${source}}}'`);
        const node = this.node(target, env);
        const problem = checkFigure(name, node.shape, args);
        if (problem !== undefined) this.fail(env, `${problem} in '{{${source}}}'`);
        return figureTag(name, node.name, args);
      }
      let grad = '';
      if (target.endsWith('.grad')) {
        grad = '.grad';
        target = target.slice(0, -5);
      }
      const node = this.node(target, env);
      return matrixTemplate(`${node.name}${grad}`, node.shape, match[1] === undefined ? '' : `:${match[1]}`);
    }
    return `{{${this.rewriteNames(source, env)}}}`;
  }

  rewriteNames(source: string, env: Env): string {
    let out = '';
    let i = 0;
    while (i < source.length) {
      const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(i));
      const prev = source[i - 1];
      if (id === null || (prev !== undefined && /[A-Za-z0-9_.:]/.test(prev))) {
        out += source[i];
        i++;
        continue;
      }
      const ends: number[] = [];
      let j = i + id[0].length;
      ends.push(j);
      for (;;) {
        if (source[j] === '.') {
          const member = /^\.[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(j));
          if (member === null) break;
          j += member[0].length;
        } else if (source[j] === '[') {
          let depth = 0;
          let k = j;
          for (; k < source.length; k++) {
            if (source[k] === '[') depth++;
            else if (source[k] === ']' && --depth === 0) break;
          }
          if (k >= source.length) break;
          j = k + 1;
        } else {
          break;
        }
        ends.push(j);
      }
      let replaced = false;
      for (const end of ends.reverse()) {
        const candidate = source.slice(i, end);
        if (candidate === 'b') break;
        const found = this.tryNode(candidate, env);
        if (found !== undefined) {
          out += found.name;
          i = end;
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        out += id[0];
        i += id[0].length;
      }
    }
    return out;
  }

  tryNode(text: string, env: Env): Node | undefined {
    try {
      const ast = parseExpr(text);
      const found = this.resolve(ast, env, undefined, 0);
      return found instanceof Node ? found : undefined;
    } catch (error) {
      if (error instanceof NeedDim) throw error;
      return undefined;
    }
  }

  param(decl: Record<string, unknown>, given: unknown, name: string, env: Env, parentEnv: Env | undefined, where: Where, instance?: string): { type: ParamType; value: Num | string | OpValue } {
    const type = decl.type;
    if (typeof type !== 'string' || !PARAM_TYPES.has(type)) fail(where, `parameter '${name}' needs a type: int, float, dim, op, tex or text`);
    if (decl.default_key !== undefined && (decl.default_key !== true || type !== 'text')) fail(where, `parameter '${name}': default_key must be true and needs type text`);
    const keyDefault = decl.default_key === true ? instance : undefined;
    const raw = given !== undefined ? given : (keyDefault ?? decl.default);
    if (raw === undefined) fail(where, `parameter '${name}' has no value and no default`);
    const sourceEnv = given !== undefined && parentEnv !== undefined ? parentEnv : env;
    const t = type as ParamType;
    if (t === 'op') {
      if (typeof raw !== 'string') fail(where, `parameter '${name}' must name an op`);
      const parentParam = given !== undefined ? parentEnv?.scope.params.get(raw) : undefined;
      const op = parentParam?.type === 'op' ? (parentParam.value as OpValue).op : raw;
      const allowed = Array.isArray(decl.values) ? (decl.values as string[]) : undefined;
      if (allowed !== undefined && !allowed.includes(op)) fail(where, `parameter '${name}' must be one of ${allowed.join(', ')}, got '${op}'`);
      if (opDef(op) === undefined) fail(where, `parameter '${name}': op '${op}' is not defined`);
      return { type: t, value: { op } };
    }
    if (t === 'tex' || t === 'text') {
      if (typeof raw !== 'string') fail(where, `parameter '${name}' must be a string`);
      return { type: t, value: raw };
    }
    const value = this.expression(raw, { ...sourceEnv, where }, undefined);
    if (!isNum(value)) fail(where, `parameter '${name}' must be a number`);
    if ((t === 'int' || t === 'dim') && !Number.isInteger(value.toy)) fail(where, `parameter '${name}' must be an integer, got ${value.toy}`);
    if (t === 'dim' && value.toy <= 0) fail(where, `parameter '${name}' must be a positive dimension, got ${value.toy}`);
    return { type: t, value: t === 'dim' ? value : num(value.toy) };
  }

  shapeOf(spec: unknown, env: Env, where: Where): Num[] {
    if (!Array.isArray(spec)) fail(where, 'shape must be a list');
    return spec.map((entry, i) => {
      const at = child(where, spec, i);
      const value = this.expression(entry, { ...env, where: at }, undefined);
      if (!isNum(value) || !Number.isInteger(value.toy) || value.toy <= 0) fail(at, `shape entry ${JSON.stringify(entry)} is not a positive integer`);
      return value;
    });
  }

  leafSpec(shape: Num[]): (number | string)[] {
    return shape.map((d) => this.dimSpec(d));
  }

}

class PageCompiler {
  readonly c: Compiler;
  private readonly root: Scope;
  private readonly pendingPatches: (() => void)[][] = [];
  private readonly formulas: { owner: string; tex: string; env: Env; target: { formula?: string } }[] = [];

  constructor(
    private readonly page: Record<string, unknown>,
    private readonly where: Where,
    rootDef: ModuleDef,
    rootDims: Map<string, Num>,
    extraDims: Map<string, Num>,
    private readonly sectionSources: Map<object, { file: string; rows: unknown[] }>,
  ) {
    this.c = new Compiler(page, where, rootDims, extraDims);
    this.root = new Scope(rootDef, '', []);
    for (const [name, value] of rootDims) this.root.dims.set(name, value);
  }

  run(): CompiledPage {
    const c = this.c;
    const page = this.page;
    const env: Env = { scope: this.root, loops: new Map(), where: this.where };
    const tree: LayoutTree = { kind: 'module', children: [] };
    this.params(this.root, page, env, undefined, {}, this.where);
    const inputsWhere = child(this.where, page, 'inputs');
    const inputs = record(page.inputs, inputsWhere, 'inputs');
    const vocab = page.vocab;
    const sequence = this.sequenceInputs(env, tree);
    for (const [name, spec] of Object.entries(inputs)) {
      const at = child(inputsWhere, inputs, name);
      if (!isRecord(spec)) fail(at, 'a page input needs tokens or value');
      const given = c.takeValue(name);
      let node: Node;
      const tex = defaultTex(name);
      try {
        if (Array.isArray(spec.tokens)) {
          node = c.g.label(name, given ?? this.tokenIds(spec.tokens as unknown[], child(at, spec, 'tokens')), { tokens: vocab as string[], tex, elem: elemOf(tex) });
        } else if (spec.value !== undefined) {
          const values = (given ?? spec.value) as Nested;
          node = c.g.input(name, inferShape(values), values, { tex, elem: elemOf(tex) });
        } else {
          fail(at, 'a page input needs tokens or value');
        }
      } catch (error) {
        if (error instanceof DslError) throw error;
        fail(at, (error as Error).message);
      }
      this.define(this.root, name, node, { ...env, where: at });
      tree.children.push({ kind: 'node', node });
      c.setOwner(node, tree.children);
    }
    this.weights(this.root, page, env, undefined, tree.children);
    this.buffers(this.root, page, env, tree.children);
    const { slots } = this.body(this.root, page, env, tree.children);
    const steps = this.moduleSteps(this.root, page, env, slots);
    this.moduleBackward(page, env);
    c.checkValuesUsed();
    this.placeLeaves(tree);
    const forward = steps.map((s) => s.step);
    for (const entry of c.backward) {
      const target = forward.find((step) => list(step.show).filter((n) => !isLeafLike(n)).includes(entry.node));
      if (target === undefined) fail(entry.where, `node '${entry.node.name}' is not computed in any forward step`);
      target.backward = { ...target.backward, ...entry.text };
    }
    this.markFormulas(steps);
    const modules = this.pageModules();
    this.placeSteps(forward, modules);
    const leavesStep = this.leavesStep(env);
    const overviewWhere = child(this.where, page, 'overview');
    const overview = page.overview === undefined ? undefined : record(page.overview, overviewWhere, 'overview');
    const stepsSpec: StepsSpec = { forward: [...leavesStep, ...forward], backward: 'reverse', modules };
    if (overview !== undefined) {
      const texts = this.texts(overview, { ...env, where: overviewWhere }, OVERVIEW_KEYS);
      stepsSpec.overview = { title: texts.title ?? String(page.title), text: texts.text ?? '' };
      if (texts.label !== undefined) stepsSpec.overview.label = texts.label;
      if (overview.backward !== undefined) stepsSpec.overview.backward = this.texts(overview.backward, { ...env, where: child(overviewWhere, overview, 'backward') }, TEXT_KEYS);
    }
    this.warnDuplicateTex();
    const compiled: CompiledPage = {
      graph: c.g,
      steps: stepsSpec,
      tree,
      modules,
      panels: this.panels(env),
      sections: this.sections(env),
      nodes: c.nodes,
      page: {
        title: String(page.title),
        lede: page.lede === undefined ? undefined : c.text(page.lede, { ...env, where: child(this.where, page, 'lede') }),
        hint: typeof page.hint === 'string' ? page.hint : undefined,
        camera: isRecord(page.camera) ? (page.camera as CameraOptions) : undefined,
      },
    };
    if (sequence !== undefined) compiled.sequence = sequence;
    if (page.layout !== undefined) compiled.layout = page.layout;
    return compiled;
  }

  tokenIds(tokens: unknown[], where: Where): number[] {
    const vocab = this.page.vocab;
    if (!Array.isArray(vocab)) fail(where, 'tokens need a vocab list on the page');
    return tokens.map((token, i) => {
      const id = (vocab as unknown[]).indexOf(token);
      if (id < 0) fail(child(where, tokens, i), `token '${String(token)}' is not in vocab`);
      return id;
    });
  }

  sequenceInputs(env: Env, tree: LayoutTree): PageSequence | undefined {
    const c = this.c;
    const page = this.page;
    if (page.sequence === undefined) return undefined;
    const where = child(this.where, page, 'sequence');
    const spec = record(page.sequence, where, 'sequence');
    checkKeys(spec, SEQUENCE_KEYS, where);
    if (!Array.isArray(spec.tokens) || spec.tokens.length < 2) fail(child(where, spec, 'tokens'), 'sequence needs a list of at least two tokens');
    for (const key of ['input', 'label']) if (typeof spec[key] !== 'string') fail(child(where, spec, key), `sequence needs '${key}', the name of the node it creates`);
    const ids = this.tokenIds(spec.tokens as unknown[], child(where, spec, 'tokens'));
    const T = ids.length - 1;
    const rows = this.root.dims.get('T');
    if (rows !== undefined && rows.toy !== T) fail(child(where, spec, 'tokens'), `sequence has ${ids.length} tokens, so T must be ${T}, but dims gives T = ${rows.toy}`);
    const tokens = [...(page.vocab as string[])];
    const make = (key: 'input' | 'label', values: number[]) => {
      const name = spec[key] as string;
      const tex = defaultTex(name);
      const node = c.g.label(name, values, { tokens, tex, elem: elemOf(tex) });
      this.define(this.root, name, node, { ...env, where: child(where, spec, key) });
      tree.children.push({ kind: 'node', node });
      c.setOwner(node, tree.children);
      return node;
    };
    const input = make('input', ids.slice(0, T));
    const label = make('label', ids.slice(1));
    const g = c.g;
    return {
      tokens,
      length: ids.length,
      get: () => [...Array.from(input.value.data, Math.round), Math.round(label.value.data[T - 1])],
      set: (next) => {
        if (next.length !== ids.length || !next.every((id) => Number.isInteger(id) && id >= 0 && id < tokens.length)) {
          throw new Error(`sequence.set: expected ${ids.length} token ids between 0 and ${tokens.length - 1}, got [${next.join(', ')}]`);
        }
        g.batch(() => {
          input.set(next.slice(0, T));
          label.set(next.slice(1));
        });
      },
    };
  }

  texts(value: unknown, env: Env, allowed: ReadonlySet<string>): BackwardOverride {
    const spec = record(value, env.where, 'a step text');
    checkKeys(spec, allowed, env.where);
    const out: BackwardOverride = {};
    for (const key of ['title', 'text', 'label'] as const) {
      if (spec[key] !== undefined) out[key] = this.c.text(spec[key], { ...env, where: child(env.where, spec, key) });
    }
    return out;
  }

  leavesStep(env: Env): ForwardStep[] {
    const page = this.page;
    if (page.leaves === undefined) return [];
    const where = child(this.where, page, 'leaves');
    const texts = this.texts(page.leaves, { ...env, where }, TEXT_KEYS);
    if (texts.title === undefined) fail(where, 'leaves needs a title');
    const show = this.c.g.nodes.filter((n) => isLeafLike(n));
    return [{ kind: 'inputs', show, title: texts.title, label: texts.label ?? texts.title, text: texts.text ?? '' }];
  }

  openModule(id: string, title: string, subtitle: string | undefined, decl: Record<string, unknown>, env: Env): ModuleRec {
    const flag = (key: string) => {
      const value = decl[key];
      if (value !== undefined && typeof value !== 'boolean') fail(child(env.where, decl, key), `${key} must be true or false`);
      return value === true;
    };
    const parent = this.c.moduleStack[this.c.moduleStack.length - 1];
    const fixed = flag('fixed');
    const rec: ModuleRec = {
      id,
      parent: parent?.id ?? null,
      title,
      subtitle: subtitle ?? '',
      fixed,
      folded: fixed || flag('folded'),
      split: flag('split'),
      nodes: [],
      fold: { forward: {}, backward: {} },
      decl,
      env,
    };
    this.c.modules.push(rec);
    this.c.moduleStack.push(rec);
    return rec;
  }

  closeModule(rec: ModuleRec, created: Node[]): void {
    this.c.moduleStack.pop();
    rec.nodes = created;
    for (const { decl, env } of [rec.defaults, rec].filter((x) => x !== undefined)) {
      if (decl.fold === undefined) continue;
      const where = child(env.where, decl, 'fold');
      const fold = record(decl.fold, where, 'fold');
      rec.fold.forward = { ...rec.fold.forward, ...this.texts(fold, { ...env, where }, FOLD_KEYS) };
      if (fold.backward !== undefined) rec.fold.backward = { ...rec.fold.backward, ...this.texts(fold.backward, { ...env, where: child(where, fold, 'backward') }, TEXT_KEYS) };
    }
  }

  pageModules(): StepModule[] {
    const recs = this.c.modules.filter((m) => m.nodes.some((n) => !isLeafLike(n)));
    const byId = new Map(recs.map((m) => [m.id, m]));
    const chain = (m: ModuleRec): ModuleRec[] => (m.parent === null || !byId.has(m.parent) ? [m] : [...chain(byId.get(m.parent)!), m]);
    const consumers = (n: Node) => this.c.g.nodes.filter((u) => u.inputs.includes(n));
    return recs.map((m): StepModule => {
      const nodes = m.nodes.filter((n) => !isLeafLike(n));
      const inside = new Set(nodes);
      const used = nodes.flatMap((n) => n.inputs);
      const inputs = [...new Set(used.filter((n) => !inside.has(n) && !isLeafLike(n)))];
      const params = [...new Set(used.flatMap((n) => (isLeafLike(n) && n.kind === 'op' ? [n, n.inputs[0]] : [n])).filter((n) => isLeafLike(n) && n.kind !== 'label'))].sort((a, b) => a.id - b.id);
      const outputs = nodes.filter((n) => consumers(n).some((u) => !inside.has(u)));
      const { segment, sub } = this.segmentOf(chain(m));
      const module: StepModule = {
        id: m.id,
        parent: m.parent !== null && byId.has(m.parent) ? m.parent : null,
        title: m.title,
        subtitle: m.subtitle,
        inputs,
        outputs: outputs.length > 0 ? outputs : nodes.slice(-1),
        nodes,
        params,
        fixed: m.fixed,
        folded: m.folded,
        segment: segment ?? m.title,
        forward: m.fold.forward,
        backward: m.fold.backward,
      };
      if (sub !== undefined) module.sub = sub;
      if (m.formula !== undefined) module.formula = m.formula;
      return module;
    });
  }

  segmentOf(chain: ModuleRec[]): { segment: string | undefined; sub: string | undefined } {
    const head = chain.find((m) => !m.split);
    const last = chain[chain.length - 1];
    const sub = head !== undefined && last !== head && !last.split ? last.title : undefined;
    return { segment: head?.title, sub };
  }

  placeSteps(forward: ForwardStep[], modules: StepModule[]): void {
    const recs = new Map(this.c.modules.map((m) => [m.id, m]));
    const depth = new Map<string, number>();
    for (const m of modules) depth.set(m.id, m.parent === null ? 0 : (depth.get(m.parent) ?? 0) + 1);
    const innermost = (node: Node) => modules.filter((m) => m.nodes.includes(node)).sort((a, b) => depth.get(b.id)! - depth.get(a.id)!)[0];
    for (const step of forward) {
      const first = list(step.show).find((n) => !isLeafLike(n));
      const module = first === undefined ? undefined : innermost(first);
      const chain: ModuleRec[] = [];
      for (let m = module === undefined ? undefined : recs.get(module.id); m !== undefined; m = m.parent === null ? undefined : recs.get(m.parent)) chain.unshift(m);
      const { segment, sub } = this.segmentOf(chain);
      if (module !== undefined) step.module = module.id;
      if (step.segment === undefined) step.segment = segment ?? step.label ?? step.title;
      if (sub !== undefined) step.sub = sub;
    }
  }

  warnDuplicateTex(): void {
    const byTex = new Map<string, string[]>();
    for (const node of this.c.g.nodes) byTex.set(node.tex, [...(byTex.get(node.tex) ?? []), node.name]);
    for (const [tex, names] of byTex) {
      if (names.length > 1) console.warn(`vizflow dsl: nodes ${names.map((n) => `'${n}'`).join(', ')} share the tex '${tex}'`);
    }
  }

  define(scope: Scope, name: string, node: Node, env: Env): void {
    if (scope.local(name) !== undefined || scope.groups.has(name)) this.c.fail(env, `name '${name}' is already defined in ${this.c.scopeLabel(scope)}`);
    scope.entries.set(name, node);
    this.c.nodes.set(join(scope.path, name), node);
  }

  params(scope: Scope, decl: Record<string, unknown>, env: Env, parentEnv: Env | undefined, given: Record<string, unknown>, where: Where, instance?: string): void {
    const paramsWhere = child(where, decl, 'params');
    const params = record(decl.params, paramsWhere, 'params');
    for (const key of Object.keys(given)) {
      if (!(key in params)) fail(child(parentEnv?.where ?? where, given, key), `module ${scope.def.name} has no parameter '${key}'`);
    }
    for (const [name, paramDecl] of Object.entries(params)) {
      const at = given[name] !== undefined && parentEnv !== undefined ? child(parentEnv.where, given, name) : child(paramsWhere, params, name);
      if (!isRecord(paramDecl)) fail(at, `parameter '${name}' must be a mapping`);
      const value = this.c.param(paramDecl, given[name], name, env, parentEnv, at, instance);
      scope.params.set(name, value);
      if (value.type === 'dim') scope.dims.set(name, value.value as Num);
    }
  }

  weights(scope: Scope, decl: Record<string, unknown>, env: Env, bound: { map: Record<string, unknown>; env: Env } | undefined, entries: LayoutEntry[]): void {
    const c = this.c;
    const weightsWhere = child(env.where, decl, 'weights');
    const weights = record(decl.weights, weightsWhere, 'weights');
    for (const key of Object.keys(bound?.map ?? {})) {
      if (!(key in weights)) fail(child(bound!.env.where, bound!.map, key), `module ${scope.def.name} has no weight '${key}'`);
    }
    for (const [name, weightDecl] of Object.entries(weights)) {
      const at = child(weightsWhere, weights, name);
      if (!isRecord(weightDecl)) fail(at, `weight '${name}' must be a mapping`);
      checkKeys(weightDecl, new Set(['shape', 'init', 'tex']), at);
      const shape = c.shapeOf(weightDecl.shape, { ...env, where: child(at, weightDecl, 'shape') }, child(at, weightDecl, 'shape'));
      const binding = bound?.map[name];
      if (binding !== undefined) {
        const bindWhere = child(bound!.env.where, bound!.map, name);
        const node = c.node(binding, { ...bound!.env, where: bindWhere });
        if (node.shape.join(',') !== shape.map((d) => d.toy).join(',')) {
          fail(bindWhere, `weight '${name}' of ${scope.def.name} has shape (${shape.map((d) => d.toy).join(', ')}), but '${String(binding)}' has shape (${node.shape.join(', ')})`);
        }
        this.define(scope, name, node, { ...env, where: at });
        continue;
      }
      const init = weightDecl.init ?? DEFAULT_INIT;
      if (typeof init !== 'string' || !INITS.has(init)) fail(child(at, weightDecl, 'init'), `init must be one of ${[...INITS].join(', ')}`);
      const path = join(scope.path, name);
      const declared = typeof weightDecl.tex === 'string' ? c.ofText(weightDecl.tex, { ...env, where: child(at, weightDecl, 'tex') }) : undefined;
      const tex = scope.texOverride.get(name) ?? this.marked(declared ?? defaultTex(name), scope, typeof weightDecl.tex === 'string' && weightDecl.tex.includes('\\of{'));
      const values = c.takeValue(path) ?? c.initValues(init, shape.map((d) => d.toy));
      let node: Node;
      try {
        node = c.g.param(path, c.leafSpec(shape), values, { tex, elem: elemOf(tex) });
      } catch (error) {
        if (error instanceof DslError || error instanceof NeedDim) throw error;
        fail(at, (error as Error).message);
      }
      this.define(scope, name, node, { ...env, where: at });
      c.setOwner(node, entries);
    }
  }

  buffers(scope: Scope, decl: Record<string, unknown>, env: Env, entries: LayoutEntry[]): void {
    const c = this.c;
    const buffersWhere = child(env.where, decl, 'buffers');
    const buffers = record(decl.buffers, buffersWhere, 'buffers');
    for (const [name, bufferDecl] of Object.entries(buffers)) {
      const at = child(buffersWhere, buffers, name);
      if (!isRecord(bufferDecl)) fail(at, `buffer '${name}' must be a mapping`);
      checkKeys(bufferDecl, new Set(['value', 'shape', 'tex']), at);
      const valueWhere = child(at, bufferDecl, 'value');
      const value = c.expression(bufferDecl.value, { ...env, where: valueWhere, buffers: true }, undefined);
      if (!isRecord(value) || !('buffer' in value)) fail(valueWhere, 'a buffer value must call rope_cos, rope_sin or causal_mask');
      let buffer = value as BufferValue;
      if (bufferDecl.shape !== undefined) {
        const shapeWhere = child(at, bufferDecl, 'shape');
        const shape = c.shapeOf(bufferDecl.shape, { ...env, where: shapeWhere }, shapeWhere);
        const size = (dims: Num[]) => dims.reduce((a, d) => a * d.toy, 1);
        if (size(shape) !== size(buffer.shape)) {
          fail(shapeWhere, `buffer '${name}' has ${size(buffer.shape)} values, which do not fit the shape (${shape.map((d) => d.toy).join(', ')})`);
        }
        buffer = { buffer: true, shape, values: reshapeNested(buffer.values, shape.map((d) => d.toy)) };
      }
      const key = `${scope.def.file}#${scope.def.name}.${name}=${String(bufferDecl.value)}(${buffer.shape.map((d) => `${d.toy}:${d.real}`).join(',')};${JSON.stringify(buffer.values)})`;
      const tex = typeof bufferDecl.tex === 'string' ? c.ofText(bufferDecl.tex, { ...env, where: child(at, bufferDecl, 'tex') }) : defaultTex(name);
      const node = c.bufferNode(key, () => {
        try {
          const created = c.g.buffer(join(scope.path, name), c.leafSpec(buffer.shape), buffer.values, { tex, elem: elemOf(tex) });
          c.setOwner(created, entries);
          return created;
        } catch (error) {
          if (error instanceof NeedDim) throw error;
          return fail(at, (error as Error).message);
        }
      });
      this.define(scope, name, node, { ...env, where: at });
    }
  }

  marked(tex: string, scope: Scope, hasOf: boolean): string {
    if (hasOf) return tex;
    return scope.marks.reduce((acc, mark) => addMark(acc, mark), tex);
  }

  countInstances(items: unknown, env: Env, counts: Map<ModuleDef, number>): void {
    if (!Array.isArray(items)) return;
    for (const item of items) {
      if (!isRecord(item)) continue;
      if ('for' in item) {
        const iterations = this.range(item.in, env);
        for (const v of iterations) this.countInstances(item.body, { ...env, loops: new Map(env.loops).set(String(item.for), v) }, counts);
        continue;
      }
      for (const value of Object.values(item)) {
        if (!isRecord(value)) continue;
        if (typeof value.use === 'string') {
          const def = env.scope.def.namespace.get(value.use);
          if (def !== undefined) counts.set(def, (counts.get(def) ?? 0) + 1);
        } else if (Array.isArray(value.body)) {
          this.countInstances(value.body, env, counts);
        }
      }
    }
  }

  range(text: unknown, env: Env): number[] {
    if (typeof text !== 'string') this.c.fail(env, "'in' must be a range like '1..H'");
    let from: Ast;
    let to: Ast;
    try {
      [from, to] = parseRange(text);
    } catch (error) {
      if (error instanceof ParseError) this.c.fail(env, `${error.message} in '${text}'`, error.offset);
      throw error;
    }
    const a = this.c.int(from, env, 0);
    const b = this.c.int(to, env, 0);
    return Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);
  }

  body(scope: Scope, decl: Record<string, unknown>, env: Env, entries: LayoutEntry[]): { slots: Slot[] } {
    const bodyWhere = child(env.where, decl, 'body');
    if (!Array.isArray(decl.body)) fail(bodyWhere, 'body must be a list');
    const counts = new Map<ModuleDef, number>();
    this.countInstances(decl.body, env, counts);
    const ordinals = new Map<ModuleDef, number>();
    return { slots: this.items(scope, decl.body, { ...env, where: bodyWhere }, entries, counts, ordinals) };
  }

  items(scope: Scope, items: unknown[], env: Env, entries: LayoutEntry[], counts: Map<ModuleDef, number>, ordinals: Map<ModuleDef, number>): Slot[] {
    const c = this.c;
    const slots: Slot[] = [];
    const named = new Set<string>();
    let lanes: Lanes | undefined;
    const patches: (() => void)[] = [];
    this.pendingPatches.push(patches);
    items.forEach((item, index) => {
      const at = child(env.where, items, index);
      if (!isRecord(item)) fail(at, 'a body item must be a mapping');
      const start = entries.length;
      const before = c.g.nodes.length;
      const slotCount = slots.length;
      const { name, beside } = this.item(scope, item, { ...env, where: at }, entries, counts, ordinals, slots);
      const created = c.g.nodes.slice(before);
      const slot = slots.length > slotCount ? slots[slots.length - 1] : undefined;
      if (beside === undefined) {
        lanes = name === undefined ? undefined : { start, members: [{ name, created, slot }] };
      } else {
        const { target, group } = this.besideGroup(beside, lanes, named);
        this.checkLaneUse(group, created, name, target, beside.where);
        const lane = (es: LayoutEntry[]): LayoutTree => (es.length === 1 && es[0].kind === 'module' ? es[0] : { kind: 'module', children: es });
        const mine = lane(entries.splice(start));
        if (group.parallel === undefined) {
          group.parallel = { kind: 'parallel', branches: [lane(entries.splice(group.start))] };
          entries.push(group.parallel);
        }
        group.parallel.branches.push(mine);
        const first = group.members[0].slot;
        if (slot?.kind === 'child' && first?.kind === 'child' && first.def === slot.def) slot.merge = first;
        group.members.push({ name, created, slot });
      }
      if (name !== undefined) named.add(name);
    });
    this.pendingPatches.pop();
    for (const patch of patches) patch();
    return slots;
  }

  besideGroup(beside: { value: unknown; env: Env; where: Where }, lanes: Lanes | undefined, named: ReadonlySet<string>): { target: string; group: Lanes } {
    if (typeof beside.value !== 'string') fail(beside.where, 'beside must name an instance, a group or a for item');
    const target = this.c.concreteName(beside.value, { ...beside.env, where: beside.where }).name;
    if (lanes !== undefined && lanes.members.some((m) => m.name === target)) return { target, group: lanes };
    if (named.has(target)) {
      fail(beside.where, `beside '${target}': '${target}' is not the item right before this one; beside names the item right before it or an item beside that one`);
    }
    fail(beside.where, `beside '${target}': no instance, group or named for item '${target}' comes before this one in the same body`);
  }

  checkLaneUse(lanes: Lanes, created: Node[], name: string | undefined, target: string, where: Where): void {
    const constant = (n: Node): boolean => n.kind === 'param' || n.kind === 'buffer' || (n.kind === 'op' && n.inputs.every(constant));
    for (const n of created) {
      for (const input of n.inputs) {
        const owner = lanes.members.find((m) => m.created.includes(input));
        if (owner === undefined || constant(input)) continue;
        const self = name === undefined ? 'this for item' : `'${name}'`;
        const source = owner.name === undefined ? 'a for item' : `'${owner.name}'`;
        fail(
          where,
          `${self} is beside '${target}' but uses '${input.name}', which ${source} in the same beside group computes; an item beside another starts at the same level, so it cannot use that item's results. Put it after the beside group`,
        );
      }
    }
  }

  item(
    scope: Scope,
    item: Record<string, unknown>,
    env: Env,
    entries: LayoutEntry[],
    counts: Map<ModuleDef, number>,
    ordinals: Map<ModuleDef, number>,
    slots: Slot[],
  ): { name: string | undefined; beside: { value: unknown; env: Env; where: Where } | undefined } {
    const c = this.c;
    const at = env.where;
    const before = c.g.nodes.length;
    const besideOf = (owner: Record<string, unknown>, ownerWhere: Where) =>
      'beside' in owner ? { value: owner.beside, env, where: child(ownerWhere, owner, 'beside') } : undefined;
    if ('for' in item) {
      checkKeys(item, FOR_KEYS, at);
      if (typeof item.for !== 'string') fail(child(at, item, 'for'), 'for needs a variable name');
      if (!Array.isArray(item.body)) fail(child(at, item, 'body'), 'for needs a body list');
      let forName: string | undefined;
      if (item.name !== undefined) {
        const nameWhere = child(at, item, 'name');
        if (typeof item.name !== 'string') fail(nameWhere, 'name must be a string');
        forName = c.concreteName(item.name, { ...env, where: nameWhere }).name;
        if (scope.local(forName) !== undefined || scope.groups.has(forName)) fail(nameWhere, `name '${forName}' is already defined in ${c.scopeLabel(scope)}`);
        scope.groups.add(forName);
      }
      const branches: LayoutTree[] = [];
      const iterations: Slot[][] = [];
      for (const v of this.range(item.in, { ...env, where: child(at, item, 'in') })) {
        const loopEnv: Env = { ...env, loops: new Map(env.loops).set(item.for, v), where: child(at, item, 'body') };
        const branch: LayoutTree = { kind: 'module', children: [] };
        const start = c.g.nodes.length;
        if (typeof item.brace === 'string') branch.title = c.ofText(item.brace, { ...loopEnv, where: child(at, item, 'brace') });
        const rec = branch.title === undefined ? undefined : this.openModule(join(scope.path, `${forName ?? item.for}[${v}]`), branch.title, undefined, {}, loopEnv);
        iterations.push(this.items(scope, item.body, loopEnv, branch.children, counts, ordinals));
        branches.push(branch);
        if (rec !== undefined) this.closeModule(rec, c.g.nodes.slice(start));
      }
      entries.push({ kind: 'parallel', branches });
      slots.push({ kind: 'loop', iterations });
      return { name: forName, beside: besideOf(item, at) };
    }
    const keys = Object.keys(item);
    if (keys.length !== 1) fail(at, `a body item must have exactly one name, got ${keys.length}`);
    const key = keys[0];
    const value = item[key];
    const valueWhere = child(at, item, key);
    const valueEnv = { ...env, where: valueWhere };
    const { name, base, indices } = c.concreteName(key, valueEnv);
    const braceTitle = env.loops.size > 0 ? base : name;
    if (isRecord(value) && 'use' in value) {
      slots.push(this.instance(scope, name, braceTitle, indices, value, valueEnv, entries, counts, ordinals, before));
      return { name, beside: besideOf(value, valueWhere) };
    }
    if (isRecord(value) && 'body' in value) {
      checkKeys(value, GROUP_KEYS, valueWhere);
      if (!Array.isArray(value.body)) fail(child(valueWhere, value, 'body'), 'body must be a list');
      if (scope.local(name) !== undefined || scope.groups.has(name)) fail(valueWhere, `name '${name}' is already defined`);
      scope.groups.add(name);
      const group: LayoutTree = { kind: 'module', title: this.titleOf(value, braceTitle, valueEnv), children: [] };
      if (typeof value.subtitle === 'string') group.subtitle = value.subtitle;
      entries.push(group);
      const rec = this.openModule(join(scope.path, name), group.title as string, group.subtitle, value, valueEnv);
      const groupSlots = this.items(scope, value.body, { ...valueEnv, where: child(valueWhere, value, 'body') }, group.children, counts, ordinals);
      this.formula(scope.path, [{ decl: value, env: valueEnv }], rec);
      this.closeModule(rec, c.g.nodes.slice(before));
      slots.push({ kind: 'group', decl: value, where: valueWhere, loops: env.loops, slots: groupSlots });
      return { name, beside: besideOf(value, valueWhere) };
    }
    if (isRecord(value) && 'beside' in value) fail(child(valueWhere, value, 'beside'), 'beside applies to an instance, a group or a for item, not to a single node');
    slots.push(this.nodeItem(scope, name, base, indices, value, valueEnv, entries, before));
    return { name: undefined, beside: undefined };
  }

  nodeItem(scope: Scope, name: string, base: string, indices: number[], value: unknown, env: Env, entries: LayoutEntry[], before: number): Slot {
    const c = this.c;
    let expr: unknown = value;
    let declared: string | undefined;
    let exprWhere = env.where;
    if (isRecord(value)) {
      checkKeys(value, ITEM_KEYS, env.where);
      if (!('op' in value)) fail(env.where, `'${name}' needs op, use or body`);
      expr = value.op;
      exprWhere = child(env.where, value, 'op');
      if (value.tex !== undefined) {
        if (typeof value.tex !== 'string') fail(child(env.where, value, 'tex'), 'tex must be a string');
        declared = value.tex;
      }
    }
    if (scope.local(name) !== undefined || scope.groups.has(name)) fail(env.where, `name '${name}' is already defined in ${c.scopeLabel(scope)}`);
    const override = scope.texOverride.get(name) ?? scope.texOverride.get(base);
    let tex: string;
    if (override !== undefined) tex = override;
    else if (declared !== undefined) tex = this.marked(c.ofText(declared, env), scope, declared.includes('\\of{'));
    else if (scope.outputTex.has(name)) tex = scope.outputTex.get(name) as string;
    else tex = this.marked(c.loopTex(base, indices), scope, false);
    const make: Make = { path: join(scope.path, name), tex, anon: [], counter: 0 };
    const result = c.expression(expr, { ...env, where: exprWhere }, make);
    if (!(result instanceof Node) || result.name !== make.path) fail(exprWhere, `'${name}' must compute a new node; its expression only refers to an existing value`);
    scope.anon.set(result, make.anon);
    this.define(scope, name, result, env);
    for (const node of make.anon) {
      entries.push({ kind: 'node', node });
      c.nodes.set(node.name, node);
    }
    entries.push({ kind: 'node', node: result });
    return { kind: 'node', name, node: result, created: c.g.nodes.slice(before) };
  }

  instance(
    parent: Scope,
    name: string,
    braceTitle: string,
    indices: number[],
    use: Record<string, unknown>,
    env: Env,
    entries: LayoutEntry[],
    counts: Map<ModuleDef, number>,
    ordinals: Map<ModuleDef, number>,
    before: number,
  ): Slot {
    const c = this.c;
    checkKeys(use, INSTANCE_KEYS, env.where);
    const useWhere = child(env.where, use, 'use');
    if (typeof use.use !== 'string') fail(useWhere, 'use must name a module');
    const def = parent.def.namespace.get(use.use);
    if (def === undefined) fail(useWhere, `module '${use.use}' is not imported`);
    if (parent.local(name) !== undefined || parent.groups.has(name)) fail(env.where, `name '${name}' is already defined in ${c.scopeLabel(parent)}`);
    const ordinal = (ordinals.get(def) ?? 0) + 1;
    ordinals.set(def, ordinal);
    let mark: string | undefined;
    if (typeof use.suffix === 'string') mark = c.ofText(use.suffix, { ...env, where: child(env.where, use, 'suffix') }) || undefined;
    else if ((counts.get(def) ?? 0) > 1) mark = indices.length > 0 ? indices.join(',') : String(ordinal);
    const scope = new Scope(def, join(parent.path, name), mark === undefined ? [...parent.marks] : [mark, ...parent.marks]);
    parent.children.set(name, scope);
    const decl = def.decl;
    const where = def.where;
    checkKeys(decl, MODULE_KEYS, where);
    const childEnv: Env = { scope, loops: new Map(), where };
    const given = record(use.params, child(env.where, use, 'params'), 'params');
    this.params(scope, decl, childEnv, { ...env, where: child(env.where, use, 'params') }, given, where, name);
    const texWhere = child(env.where, use, 'tex');
    for (const [key, tex] of Object.entries(record(use.tex, texWhere, 'tex'))) {
      if (typeof tex !== 'string') fail(child(texWhere, use.tex as object, key), 'tex must be a string');
      scope.texOverride.set(key, c.ofText(tex, { ...env, where: child(texWhere, use.tex as object, key) }));
    }
    const declInputsWhere = child(where, decl, 'inputs');
    const declInputs = record(decl.inputs, declInputsWhere, 'inputs');
    const inputsWhere = child(env.where, use, 'inputs');
    const givenInputs = record(use.inputs, inputsWhere, 'inputs');
    for (const key of Object.keys(givenInputs)) if (!(key in declInputs)) fail(child(inputsWhere, givenInputs, key), `module ${def.name} has no input '${key}'`);
    for (const [inputName, inputDecl] of Object.entries(declInputs)) {
      if (!(inputName in givenInputs)) fail(inputsWhere, `input '${inputName}' of ${def.name} is not given`);
      const at = child(inputsWhere, givenInputs, inputName);
      const node = c.node(givenInputs[inputName], { ...env, where: at });
      const declWhere = child(declInputsWhere, declInputs, inputName);
      const inputSpec = record(inputDecl, declWhere, `input '${inputName}'`);
      checkKeys(inputSpec, new Set(['shape', 'tex']), declWhere);
      if (typeof inputSpec.tex === 'string') scope.inputTex.set(inputName, inputSpec.tex);
      if (inputSpec.shape !== undefined) this.match(scope, inputSpec.shape, node, childEnv, child(declWhere, inputSpec, 'shape'), at);
      scope.entries.set(inputName, node);
      c.nodes.set(join(scope.path, inputName), node);
    }
    const outputsWhere = child(where, decl, 'outputs');
    if (!Array.isArray(decl.outputs)) fail(outputsWhere, 'outputs must be a list of names');
    const outputs = decl.outputs as unknown[];
    const boundWhere = child(env.where, use, 'outputs');
    const bound = record(use.outputs, boundWhere, 'outputs');
    const bindings: [string, string, Where][] = [];
    for (const [output, target] of Object.entries(bound)) {
      const at = child(boundWhere, bound, output);
      if (!outputs.includes(output)) fail(at, `module ${def.name} has no output '${output}'`);
      if (typeof target !== 'string') fail(at, 'an output binding must be a name');
      const concrete = c.concreteName(target, { ...env, where: at });
      bindings.push([output, concrete.name, at]);
      if (!scope.texOverride.has(output)) scope.outputTex.set(output, this.marked(c.loopTex(concrete.base, concrete.indices), parent, false));
    }
    if (use.inline !== undefined && typeof use.inline !== 'boolean') fail(child(env.where, use, 'inline'), 'inline must be true or false');
    const inline = use.inline === true;
    for (const key of MODULE_ITEM_KEYS) {
      if (inline && use[key] !== undefined) fail(child(env.where, use, key), `an inline instance has no box of its own, so it cannot have '${key}'`);
    }
    const tree: LayoutTree = inline ? { kind: 'module', children: [] } : { kind: 'module', title: this.titleOf(use, braceTitle, env), children: [] };
    if (!inline && typeof use.subtitle === 'string') tree.subtitle = use.subtitle;
    else if (!inline && typeof decl.subtitle === 'string') tree.subtitle = decl.subtitle;
    entries.push(tree);
    const rec = inline ? undefined : this.openModule(scope.path, tree.title as string, tree.subtitle, use, env);
    if (rec !== undefined) rec.defaults = { decl, env: childEnv };
    this.weights(scope, decl, childEnv, use.weights === undefined ? undefined : { map: record(use.weights, child(env.where, use, 'weights'), 'weights'), env: { ...env, where: child(env.where, use, 'weights') } }, tree.children);
    this.buffers(scope, decl, childEnv, tree.children);
    const { slots } = this.body(scope, decl, childEnv, tree.children);
    for (const output of outputs) {
      if (typeof output !== 'string' || !(scope.entries.get(output) instanceof Node)) fail(outputsWhere, `output '${String(output)}' of ${def.name} is not defined in its body`);
    }
    const steps = this.moduleSteps(scope, decl, childEnv, slots);
    const formulaSources = [{ decl, env: childEnv }, { decl: use, env }];
    if (use.brace !== undefined) {
      if (!inline) fail(child(env.where, use, 'brace'), 'only an inline instance takes brace; a module with a box has its own brace');
      const brace = this.braceOf(use.brace, { ...env, where: child(env.where, use, 'brace') });
      this.formula(scope.path, formulaSources, brace);
      for (const rec of steps) rec.step.brace ??= brace;
    } else if (inline && use.formula !== undefined) {
      fail(child(env.where, use, 'formula'), 'an inline instance shows a formula only under its brace, so formula needs brace');
    }
    if (rec !== undefined) this.formula(scope.path, formulaSources, rec);
    this.moduleBackward(decl, childEnv);
    for (const [output, target, at] of bindings) this.define(parent, target, scope.entries.get(output) as Node, { ...env, where: at });
    this.pendingPatches[this.pendingPatches.length - 1].push(() => this.patch(scope, steps, use, env));
    const created = c.g.nodes.slice(before);
    if (rec !== undefined) this.closeModule(rec, created);
    return { kind: 'child', def, steps, created };
  }

  match(scope: Scope, pattern: unknown, node: Node, env: Env, where: Where, givenAt: Where): void {
    if (!Array.isArray(pattern)) fail(where, 'shape must be a list');
    const rest = pattern.indexOf('...');
    const extra = node.shape.length - (pattern.length - 1);
    if (rest >= 0 ? extra < 0 || pattern.lastIndexOf('...') !== rest : pattern.length !== node.shape.length) {
      fail(givenAt, `input has shape (${node.shape.join(', ')}), but ${scope.def.name} expects (${pattern.join(', ')})`);
    }
    pattern.forEach((entry, at) => {
      if (at === rest) return;
      const i = rest >= 0 && at > rest ? at - 1 + extra : at;
      const actual: Num = { toy: node.shape[i], real: node.realShape[i] };
      if (typeof entry === 'string' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(entry) && !scope.dims.has(entry) && !scope.params.has(entry)) {
        scope.dims.set(entry, actual);
        return;
      }
      const expected = this.c.expression(entry, { ...env, where: child(where, pattern, at) }, undefined);
      if (!isNum(expected) || expected.toy !== actual.toy) {
        fail(givenAt, `input has shape (${node.shape.join(', ')}), but dimension ${i} of ${scope.def.name}'s pattern (${pattern.join(', ')}) is ${isNum(expected) ? expected.toy : '?'}`);
      }
    });
  }

  formula(owner: string, sources: { decl: Record<string, unknown>; env: Env }[], target: { formula?: string }): void {
    const source = sources.filter((s) => s.decl.formula !== undefined).pop();
    if (source === undefined) return;
    const where = child(source.env.where, source.decl, 'formula');
    if (typeof source.decl.formula !== 'string') fail(where, 'formula must be a string');
    this.formulas.push({ owner, tex: source.decl.formula, env: { ...source.env, where }, target });
  }

  markFormulas(steps: StepRec[]): void {
    const marks = new Map<StepRec, string>();
    const markOf = (rec: StepRec): string => {
      let name = marks.get(rec);
      if (name === undefined) {
        name = `s${marks.size}`;
        marks.set(rec, name);
        rec.step.marks = [...(rec.step.marks ?? []), name];
      }
      return name;
    };
    for (const { owner, tex, env, target } of this.formulas) {
      const where = env.where;
      const recsOf = (path: string): StepRec[] => {
        if (path === '*' || path.endsWith('.*')) {
          const base = path === '*' ? owner : join(owner, path.slice(0, -2));
          const inside = (ref: { owner: string }) => ref.owner === base || ref.owner.startsWith(`${base}.`);
          const found = steps.filter((s) => s.refs.some(inside));
          if (found.length === 0) fail(where, `\\step{${path}}: '${base || 'page'}' has no steps`);
          return found;
        }
        const dot = path.lastIndexOf('.');
        const stepOwner = dot < 0 ? owner : join(owner, path.slice(0, dot));
        const id = dot < 0 ? path : path.slice(dot + 1);
        const rec = steps.find((s) => s.refs.some((ref) => ref.owner === stepOwner && ref.id === id));
        if (rec === undefined) {
          const known = steps.flatMap((s) => s.refs).filter((ref) => ref.owner === owner).map((ref) => ref.id);
          fail(where, `\\step{${path}}: '${owner || 'page'}' has no step '${path}'; its steps are ${known.join(', ') || '(none)'}`);
        }
        return [rec];
      };
      const mark = (spec: string): string => [...new Set(spec.split(',').flatMap((path) => recsOf(path.trim())).map(markOf))].join('.');
      try {
        target.formula = markSteps(this.c.ofText(tex, env), mark);
      } catch (error) {
        if (error instanceof DslError) throw error;
        fail(where, (error as Error).message);
      }
    }
  }

  braceOf(value: unknown, env: Env): { title: string; subtitle?: string; formula?: string } {
    const spec = record(value, env.where, 'brace');
    checkKeys(spec, BRACE_KEYS, env.where);
    if (typeof spec.title !== 'string') fail(env.where, 'brace needs a title');
    const brace: { title: string; subtitle?: string } = { title: this.c.ofText(spec.title, { ...env, where: child(env.where, spec, 'title') }) };
    if (spec.subtitle !== undefined) {
      if (typeof spec.subtitle !== 'string') fail(child(env.where, spec, 'subtitle'), 'subtitle must be a string');
      brace.subtitle = this.c.ofText(spec.subtitle, { ...env, where: child(env.where, spec, 'subtitle') });
    }
    return brace;
  }

  titleOf(decl: Record<string, unknown>, fallback: string, env: Env): string {
    if (decl.title === undefined) return fallback;
    if (typeof decl.title !== 'string') fail(child(env.where, decl, 'title'), 'title must be a string');
    return this.c.ofText(decl.title, { ...env, where: child(env.where, decl, 'title') });
  }

  resolveSteps(scope: Scope, decl: Record<string, unknown>, env: Env, field: string): ResolvedStep[] {
    const c = this.c;
    if (decl.steps === undefined) return [];
    const stepsWhere = child(env.where, decl, field);
    if (!Array.isArray(decl.steps)) fail(stepsWhere, 'steps must be a list');
    const seen = new Set<string>();
    return decl.steps.map((step, i) => {
      const at = child(stepsWhere, decl.steps as unknown[], i);
      if (!isRecord(step)) fail(at, 'a step must be a mapping');
      checkKeys(step, STEP_KEYS, at);
      if (typeof step.id !== 'string') fail(at, 'a step needs an id');
      if (seen.has(step.id)) fail(child(at, step, 'id'), `step id '${step.id}' is used twice`);
      seen.add(step.id);
      const fieldEnv = (key: string): Env => ({ ...env, where: child(at, step, key) });
      if (step.show === undefined) fail(at, 'a step needs show');
      const shown = list(step.show as unknown).flatMap((s, k) => c.nodeList(s, { ...env, where: Array.isArray(step.show) ? child(child(at, step, 'show'), step.show, k) : child(at, step, 'show') }));
      const withAnon = (nodes: Node[]) => [...new Set(nodes.flatMap((n) => [...(scope.anon.get(n) ?? []), n]))];
      const show = withAnon(shown);
      const forward: ForwardStep = {
        show,
        title: step.title === undefined ? step.id : c.text(step.title, fieldEnv('title')),
      };
      if (step.active !== undefined) forward.active = withAnon(list(step.active as unknown).flatMap((s) => c.nodeList(s, fieldEnv('active'))));
      if (step.label !== undefined) forward.label = c.text(step.label, fieldEnv('label'));
      if (step.text !== undefined) forward.text = c.text(step.text, fieldEnv('text'));
      if (step.segment !== undefined) {
        if (typeof step.segment !== 'string') fail(child(at, step, 'segment'), 'segment must be a string');
        forward.segment = step.segment;
      }
      if (step.brace !== undefined) forward.brace = this.braceOf(step.brace, fieldEnv('brace'));
      if (step.backward !== undefined) forward.backward = this.texts(step.backward, fieldEnv('backward'), TEXT_KEYS);
      return { rec: { refs: [{ owner: scope.path, id: step.id }], step: forward }, show };
    });
  }

  moduleSteps(scope: Scope, decl: Record<string, unknown>, env: Env, slots: Slot[]): StepRec[] {
    const containers: { key: string; steps: ResolvedStep[] }[] = [{ key: '', steps: this.resolveSteps(scope, decl, env, 'steps') }];
    const events: { slot: Extract<Slot, { kind: 'node' | 'child' }>; key: string }[] = [];
    const keys = new Map<Slot, string>();
    const flatten = (level: Slot[], prefix: string) => {
      level.forEach((slot, i) => {
        const key = slot.kind === 'child' && slot.merge !== undefined ? (keys.get(slot.merge) ?? `${prefix}/${i}`) : `${prefix}/${i}`;
        keys.set(slot, key);
        if (slot.kind === 'group') {
          containers.push({ key, steps: this.resolveSteps(scope, slot.decl, { ...env, where: slot.where, loops: slot.loops }, 'steps') });
          flatten(slot.slots, key);
        } else if (slot.kind === 'loop') {
          for (const iteration of slot.iterations) flatten(iteration, `${key}*`);
        } else {
          events.push({ slot, key });
        }
      });
    };
    flatten(slots, '');
    const covered = new Set<Node>(containers.flatMap((container) => container.steps.flatMap((s) => s.show)));
    const anchored = new Map<number, { key: string; rec: StepRec }[]>();
    for (const container of containers) {
      let floor = -1;
      for (const step of container.steps) {
        const first = events.findIndex((event) => step.show.some((n) => event.slot.created.includes(n)));
        floor = Math.max(floor, first);
        anchored.set(floor, [...(anchored.get(floor) ?? []), { key: `own${container.key}:${step.rec.refs[0].id}`, rec: step.rec }]);
      }
    }
    const out: { key: string; rec: StepRec; combine?: boolean }[] = [...(anchored.get(-1) ?? [])];
    events.forEach(({ slot, key }, i) => {
      if (slot.kind === 'child') slot.steps.forEach((rec, j) => out.push({ key: `${key}#${j}`, rec, combine: slot.merge !== undefined }));
      else if (!covered.has(slot.node)) out.push({ key, rec: this.autoStep(scope, slot.name, slot.node, covered) });
      out.push(...(anchored.get(i) ?? []));
    });
    const firsts = new Map<string, StepRec>();
    const merged: StepRec[] = [];
    for (const { key, rec, combine } of out) {
      const first = firsts.get(key);
      if (first === undefined) {
        firsts.set(key, rec);
        merged.push(rec);
        continue;
      }
      first.refs.push(...rec.refs);
      if (combine) {
        first.step.label = combineLabels(first.step.label, rec.step.label);
        const backward = combineLabels(first.step.backward?.label, rec.step.backward?.label);
        if (backward !== undefined) first.step.backward = { ...first.step.backward, label: backward };
      }
      const union = (a: Node | Node[], b: Node | Node[]) => [...new Set([...list(a), ...list(b)])];
      if (first.step.active !== undefined || rec.step.active !== undefined) {
        first.step.active = union(first.step.active ?? first.step.show, rec.step.active ?? rec.step.show);
      }
      first.step.show = union(first.step.show, rec.step.show);
    }
    return merged;
  }

  autoStep(scope: Scope, name: string, node: Node, covered: Set<Node>): StepRec {
    const anon = scope.anon.get(node) ?? [];
    const leaves = [...anon, node]
      .flatMap((n) => n.inputs)
      .flatMap((input) => (input.op === 'transpose' && input.inputs[0].kind === 'param' ? [input.inputs[0], input] : [input]))
      .filter((input) => (input.kind === 'param' || input.kind === 'buffer' || (input.op === 'transpose' && input.inputs[0].kind === 'param')) && !covered.has(input));
    const show = [...new Set([...leaves, ...anon, node])];
    for (const n of leaves) covered.add(n);
    return { refs: [{ owner: scope.path, id: name }], step: { show, title: name, label: name } };
  }

  moduleBackward(decl: Record<string, unknown>, env: Env): void {
    const c = this.c;
    const where = child(env.where, decl, 'backward');
    const backward = record(decl.backward, where, 'backward');
    for (const [name, text] of Object.entries(backward)) {
      const at = child(where, backward, name);
      const node = c.node(name, { ...env, where: at });
      const spec = record(text, at, 'a backward text');
      checkKeys(spec, new Set(['title', 'text', 'label']), at);
      const override: BackwardOverride = {};
      if (spec.title !== undefined) override.title = c.text(spec.title, { ...env, where: child(at, spec, 'title') });
      if (spec.text !== undefined) override.text = c.text(spec.text, { ...env, where: child(at, spec, 'text') });
      if (spec.label !== undefined) override.label = c.text(spec.label, { ...env, where: child(at, spec, 'label') });
      c.backward.push({ node, text: override, where: at });
    }
  }

  patch(scope: Scope, steps: StepRec[], use: Record<string, unknown>, env: Env): void {
    const c = this.c;
    const where = child(env.where, use, 'steps');
    const patches = record(use.steps, where, 'steps');
    for (const [path, patchSpec] of Object.entries(patches)) {
      const at = child(where, patches, path);
      const spec = record(patchSpec, at, 'a step patch');
      checkKeys(spec, PATCH_KEYS, at);
      const dot = path.lastIndexOf('.');
      const owner = dot < 0 ? scope.path : join(scope.path, path.slice(0, dot));
      const id = dot < 0 ? path : path.slice(dot + 1);
      const target = steps.find((s) => s.refs.some((ref) => ref.owner === owner && ref.id === id));
      if (target === undefined) fail(at, `step '${path}' does not exist in '${scope.path}'`);
      const fieldEnv = (key: string): Env => ({ ...env, where: child(at, spec, key) });
      if (spec.title !== undefined) target.step.title = c.text(spec.title, fieldEnv('title'));
      if (spec.label !== undefined) target.step.label = c.text(spec.label, fieldEnv('label'));
      if (spec.text !== undefined) target.step.text = c.text(spec.text, fieldEnv('text'));
      if (spec.append !== undefined) {
        const extra = c.text(spec.append, fieldEnv('append'));
        target.step.text = target.step.text ? `${target.step.text}\n${extra}` : extra;
      }
      if (spec.backward !== undefined) target.step.backward = { ...target.step.backward, ...this.texts(spec.backward, fieldEnv('backward'), TEXT_KEYS) };
    }
  }

  placeLeaves(tree: LayoutTree): void {
    const c = this.c;
    const placed = new Set<Node>();
    const walk = (entries: LayoutEntry[]) => {
      for (const entry of entries) {
        if (entry.kind === 'node') placed.add(entry.node);
        else if (entry.kind === 'parallel') entry.branches.forEach((b) => walk(b.children));
        else walk(entry.children);
      }
    };
    walk(tree.children);
    const leaves = c.g.nodes.filter((n) => (n.kind === 'param' || n.kind === 'buffer') && !placed.has(n));
    const uses = (node: Node, leaf: Node) => node.inputs.some((input) => input === leaf || (input.op === 'transpose' && input.inputs[0] === leaf));
    const insert = (entries: LayoutEntry[], leaf: Node): boolean => {
      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i];
        if (entry.kind === 'node') {
          if (uses(entry.node, leaf)) {
            entries.splice(i, 0, { kind: 'node', node: leaf });
            return true;
          }
        } else if (entry.kind === 'parallel') {
          if (entry.branches.some((b) => insert(b.children, leaf))) return true;
        } else if (insert(entry.children, leaf)) {
          return true;
        }
      }
      return false;
    };
    for (const leaf of leaves) {
      if (!insert(tree.children, leaf)) (c.ownerOf(leaf) ?? tree.children).push({ kind: 'node', node: leaf });
    }
  }

  panels(env: Env): PanelItem[] {
    const c = this.c;
    const page = this.page;
    if (page.panels === undefined) return [];
    const where = child(this.where, page, 'panels');
    if (!Array.isArray(page.panels)) fail(where, 'panels must be a list');
    const nodeAt = (value: unknown, at: Where) => c.node(value, { ...env, where: at });
    const textAt = (value: unknown, at: Where) => c.text(value, { ...env, where: at });
    const nodesAt = (value: unknown, at: Where) =>
      Array.isArray(value) ? value.flatMap((v, k) => c.nodeList(v, { ...env, where: child(at, value, k) })) : c.nodeList(value, { ...env, where: at });
    return page.panels.map((panel, i): PanelItem => {
      const at = child(where, page.panels as unknown[], i);
      if (panel === 'loss') return 'loss';
      if (!isRecord(panel)) fail(at, "a panel must be 'loss' or a mapping");
      const out: Record<string, unknown> = { ...panel };
      const field = (key: string) => child(at, panel, key);
      for (const key of ['title', 'note', 'caption']) if (typeof panel[key] === 'string') out[key] = textAt(panel[key], field(key));
      for (const key of ['tokens', 'matrix', 'sliders', 'ids']) {
        if (panel[key] === undefined) continue;
        out[key] = nodeAt(panel[key], field(key));
      }
      if (panel.attention !== undefined) out.attention = nodesAt(panel.attention, field('attention'));
      if (panel.highlight !== undefined) {
        const highlight = panel.highlight;
        out.highlight = isRecord(highlight) ? { ...highlight, node: nodeAt(highlight.node, child(field('highlight'), highlight, 'node')) } : nodeAt(highlight, field('highlight'));
      }
      if (panel.matrices !== undefined) {
        const matrices = panel.matrices;
        if (!Array.isArray(matrices)) fail(field('matrices'), 'matrices must be a list');
        out.matrices = matrices.map((entry, k) => {
          const entryAt = child(field('matrices'), matrices, k);
          if (!isRecord(entry)) fail(entryAt, 'a matrices entry must be a mapping');
          const converted: Record<string, unknown> = { ...entry, matrix: nodeAt(entry.matrix, child(entryAt, entry, 'matrix')) };
          if (typeof entry.caption === 'string') converted.caption = textAt(entry.caption, child(entryAt, entry, 'caption'));
          return converted;
        });
      }
      if (Array.isArray(panel.columns)) {
        const columns = panel.columns;
        out.columns = columns.map((column, k) => {
          const columnAt = child(field('columns'), columns, k);
          if (isRecord(column)) {
            const converted: Record<string, unknown> = { ...column, node: nodeAt(column.node, child(columnAt, column, 'node')) };
            if (typeof column.title === 'string') converted.title = textAt(column.title, child(columnAt, column, 'title'));
            return converted;
          }
          return nodeAt(column, columnAt);
        });
      }
      if (panel.sequence !== undefined) {
        if (panel.sequence !== true) fail(field('sequence'), 'sequence must be true');
        Object.assign(out, this.trainable(panel, ['sgd'], env, at));
      }
      if (isRecord(panel.loss)) out.loss = this.lossPanel(panel.loss, { ...env, where: field('loss') });
      return out as PanelItem;
    });
  }

  lossPanel(loss: Record<string, unknown>, env: Env): Record<string, unknown> {
    const c = this.c;
    const out: Record<string, unknown> = { ...loss, ...this.trainable(loss, ['sgd', 'random'], env, env.where) };
    const at = (key: string, obj: object = loss) => ({ ...env, where: child(env.where, obj, key) });
    if (typeof loss.text === 'string') out.text = c.text(loss.text, at('text'));
    if (loss.parts !== undefined) out.parts = c.node(loss.parts, at('parts'));
    return out;
  }

  trainable(spec: Record<string, unknown>, keys: string[], env: Env, where: Where): Record<string, unknown> {
    const c = this.c;
    const out: Record<string, unknown> = {};
    for (const key of keys) {
      const value = spec[key];
      if (value === undefined) continue;
      const keyWhere = child(where, spec, key);
      if (!isRecord(value)) fail(keyWhere, `${key} must be a mapping`);
      const nodesKey = key === 'sgd' ? 'params' : 'nodes';
      const selected = value[nodesKey];
      const nodesWhere = child(keyWhere, value, nodesKey);
      const nodes = selected === 'all'
        ? c.g.nodes.filter((n) => n.kind === 'param')
        : list(selected as unknown).flatMap((s) => c.nodeList(s, { ...env, where: nodesWhere }));
      out[key] = { ...value, [nodesKey]: nodes };
    }
    return out;
  }

  sections(env: Env): SectionItem[] {
    const c = this.c;
    const page = this.page;
    if (page.sections === undefined) return [];
    const where = child(this.where, page, 'sections');
    if (!Array.isArray(page.sections)) fail(where, 'sections must be a list');
    const sections = page.sections;
    const nodesAt = (value: unknown, at: Where) =>
      Array.isArray(value) ? value.flatMap((v, k) => c.nodeList(v, { ...env, where: child(at, value, k) })) : c.nodeList(value, { ...env, where: at });
    const textAt = (value: unknown, at: Where) => {
      if (typeof value !== 'string') fail(at, 'expected text');
      return c.text(value, { ...env, where: at });
    };
    const numberAt = (value: unknown, at: Where) => {
      if (typeof value === 'number') return value;
      const found = typeof value === 'string' ? c.expression(value, { ...env, where: at }, undefined) : undefined;
      if (typeof found !== 'object' || found === null || !('toy' in found)) return fail(at, 'expected a number or a dimension expression');
      return found.toy;
    };
    const block = (value: unknown, at: Where): SectionBlock => {
      if (value === 'rope_dial' || value === 'rope_curve') return value;
      if (!isRecord(value)) fail(at, "a block must be 'rope_dial', 'rope_curve' or a mapping");
      if (Object.keys(value).length !== 1) fail(at, 'a block must have exactly one key');
      const key = Object.keys(value)[0];
      const keyAt = child(at, value, key);
      if (key === 'text') return { text: textAt(value.text, keyAt) };
      if (key === 'lead') return { lead: textAt(value.lead, keyAt) };
      if (key === 'buttons') {
        const buttons = value.buttons;
        if (!Array.isArray(buttons)) fail(keyAt, 'buttons must be a list');
        return {
          buttons: buttons.map((b, k) => {
            const buttonAt = child(keyAt, buttons, k);
            if (!isRecord(b)) fail(buttonAt, 'a button must be a mapping with label and go');
            return { label: textAt(b.label, child(buttonAt, b, 'label')), go: c.node(b.go, { ...env, where: child(buttonAt, b, 'go') }) };
          }),
        };
      }
      if (key === 'rope_compare') {
        const spec = value.rope_compare;
        if (!isRecord(spec)) fail(keyAt, 'rope_compare must be a mapping');
        const specAt = (k: string) => child(keyAt, spec, k);
        return {
          rope_compare: {
            scores: nodesAt(spec.scores, specAt('scores')),
            weights: nodesAt(spec.weights, specAt('weights')),
            ids: c.node(spec.ids, { ...env, where: specAt('ids') }),
          },
        };
      }
      return fail(keyAt, `unknown block '${key}'`);
    };
    return sections.map((section, i): SectionItem => {
      const at = child(where, sections, i);
      if (!isRecord(section)) fail(at, 'a section must be a mapping');
      const field = (key: string) => child(at, section, key);
      const out: SectionItem = { rows: [] };
      const source = this.sectionSources.get(section);
      if (source !== undefined && section.rows !== undefined) fail(field('rows'), 'a section takes either src or rows');
      if (section.title !== undefined) out.title = textAt(section.title, field('title'));
      if (section.rope !== undefined) {
        const rope = section.rope;
        if (!isRecord(rope)) fail(field('rope'), 'rope must be a mapping');
        const ropeAt = (key: string) => child(field('rope'), rope, key);
        const rows = rope.rows ?? [-1, 1];
        if (!Array.isArray(rows) || rows.length !== 2) fail(ropeAt('rows'), 'rows must be [q row, k row]');
        out.rope = {
          q: nodesAt(rope.q, ropeAt('q')),
          k: nodesAt(rope.k, ropeAt('k')),
          rows: [numberAt(rows[0], ropeAt('rows')), numberAt(rows[1], ropeAt('rows'))],
          head: rope.head === undefined ? 1 : numberAt(rope.head, ropeAt('head')),
          theta: numberAt(rope.theta, ropeAt('theta')),
          dh: numberAt(rope.dh, ropeAt('dh')),
          positions: rope.positions === undefined ? ROPE_POSITIONS : numberAt(rope.positions, ropeAt('positions')),
        };
        if (!Number.isInteger(out.rope.dh) || out.rope.dh < 2 || out.rope.dh % 2 !== 0) fail(ropeAt('dh'), 'dh must be an even integer');
        if (!Number.isInteger(out.rope.positions) || out.rope.positions < 2) fail(ropeAt('positions'), 'positions must be an integer of at least 2');
      }
      const rows = source === undefined ? section.rows : source.rows;
      const rowsField = source === undefined ? field('rows') : { file: source.file, path: 'rows', obj: undefined };
      if (!Array.isArray(rows)) fail(rowsField, 'a section needs src or rows');
      out.rows = rows.map((row, r): SectionRow => {
        const rowAt = child(rowsField, rows, r);
        if (Array.isArray(row)) return { cells: [row.map((b, k) => block(b, child(rowAt, row, k)))] };
        if (!isRecord(row) || !Array.isArray(row.cells)) fail(rowAt, 'a row must be a list of blocks or a mapping with cells');
        const cells = row.cells;
        const columns = row.columns;
        if (columns !== undefined && (!Array.isArray(columns) || columns.length !== cells.length)) fail(child(rowAt, row, 'columns'), 'columns must list one width per cell');
        return {
          columns: columns as (number | string)[] | undefined,
          cells: cells.map((cell, k) => {
            const cellAt = child(child(rowAt, row, 'cells'), cells, k);
            if (!Array.isArray(cell)) fail(cellAt, 'a cell must be a list of blocks');
            return cell.map((b, n) => block(b, child(cellAt, cell, n)));
          }),
        };
      });
      return out;
    });
  }
}

function rootDimsOf(page: Record<string, unknown>, where: Where): Map<string, Num> {
  const dimsWhere = child(where, page, 'dims');
  const dims = record(page.dims, dimsWhere, 'dims');
  const real = isRecord(page.real) ? page.real : {};
  const out = new Map<string, Num>();
  for (const [name, value] of Object.entries(dims)) {
    if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) fail(child(dimsWhere, dims, name), `dimension '${name}' must be a positive integer`);
    const realValue = real[name];
    out.set(name, { toy: value, real: typeof realValue === 'number' ? realValue : value });
  }
  return out;
}

export async function compileSpec(spec: unknown, file: string, options: LoadOptions): Promise<CompiledPage> {
  const where: Where = { file, path: '', obj: isRecord(spec) ? spec : undefined };
  if (!isRecord(spec)) fail(where, 'a page must be a mapping');
  checkKeys(spec, PAGE_KEYS, where);
  if (typeof spec.title !== 'string') fail(where, "a page needs 'title'");
  const loader = new Loader(options);
  const pageFile = options.base;
  const namespace = await loader.namespace(spec, pageFile, pageFile === undefined ? [] : [pageFile], where);
  const modulesWhere = child(where, spec, 'modules');
  const locals = record(spec.modules, modulesWhere, 'modules');
  for (const [name, decl] of Object.entries(locals)) {
    const at = child(modulesWhere, locals, name);
    if (!isRecord(decl)) fail(at, `module '${name}' must be a mapping`);
    if (namespace.has(name)) fail(at, `module '${name}' is already imported`);
    namespace.set(name, { name, file, decl, where: { ...at, obj: decl, key: undefined }, namespace });
  }
  for (const [name, decl] of Object.entries(locals)) {
    await loader.directUses(decl as Record<string, unknown>, pageFile, pageFile === undefined ? [] : [pageFile], child(modulesWhere, locals, name), namespace);
  }
  const sectionSources = new Map<object, { file: string; rows: unknown[] }>();
  if (Array.isArray(spec.sections)) {
    const sectionsWhere = child(where, spec, 'sections');
    for (const [i, section] of spec.sections.entries()) {
      if (!isRecord(section) || section.src === undefined) continue;
      const at = child(child(sectionsWhere, spec.sections, i), section, 'src');
      if (typeof section.src !== 'string') fail(at, 'src must be a path to a Markdown file');
      const url = loader.resolve(section.src, pageFile, at);
      sectionSources.set(section, { file: url, rows: sectionRows(await loader.text(url), url) });
    }
  }
  const rootDef: ModuleDef = { name: String(spec.title), file, decl: spec, where, namespace };
  const rootDims = rootDimsOf(spec, where);
  const extra = new Map<string, Num>();
  for (;;) {
    try {
      return new PageCompiler(spec, where, rootDef, rootDims, extra, sectionSources).run();
    } catch (error) {
      if (!(error instanceof NeedDim)) throw error;
      if (extra.has(error.name)) throw new Error(`dsl: dimension ${error.name} could not be registered`);
      extra.set(error.name, error.value);
    }
  }
}
