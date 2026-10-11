import { type Graph, Node, graph, opDef } from '../autograd';
import { explainOf } from '../explain';
import type { LayoutEntry, LayoutTree } from '../scene3d/auto';
import type { SceneModule } from '../scene3d/modules';
import type { CameraOptions } from '../scene3d/scene';
import { type BackwardOverride, type ForwardStep, type StepsSpec } from '../steps';
import { backwardGroups } from '../steps/steps';
import { type Nested, inferShape } from '../tensor';
import type { PanelItem } from '../ui';
import { type Ast, ParseError, parseExpr, parseRange } from './expr';
import { type LoadOptions, Loader, type ModuleDef } from './load';
import { DslError, type Where, child, fail } from './source';
import { OP_TEX, addMark, defaultTex, elemOf, numberTex, opTex } from './tex';

export interface CompiledPage {
  graph: Graph;
  steps: StepsSpec;
  tree: LayoutTree;
  modules: SceneModule[];
  panels: PanelItem[];
  nodes: Map<string, Node>;
  page: { title: string; lede?: string; hint?: string; camera?: CameraOptions };
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
  owner: string;
  id: string;
  step: ForwardStep;
}

type Slot =
  | { kind: 'node'; name: string; node: Node; created: Node[] }
  | { kind: 'child'; steps: StepRec[]; created: Node[] }
  | { kind: 'loop'; iterations: Slot[][] }
  | { kind: 'group'; decl: Record<string, unknown>; where: Where; slots: Slot[] };

interface ResolvedStep {
  rec: StepRec;
  show: Node[];
}

const ITEM_KEYS = new Set(['op', 'tex']);
const INSTANCE_KEYS = new Set(['use', 'params', 'inputs', 'outputs', 'weights', 'suffix', 'tex', 'steps']);
const GROUP_KEYS = new Set(['body', 'steps', 'subtitle']);
const FOR_KEYS = new Set(['for', 'in', 'brace', 'body']);
const MODULE_KEYS = new Set(['module', 'import', 'subtitle', 'params', 'inputs', 'outputs', 'weights', 'buffers', 'body', 'steps', 'backward']);
const PAGE_KEYS = new Set([
  'title', 'lede', 'hint', 'camera', 'overview', 'import', 'modules', 'subtitle', 'params', 'dims', 'real', 'vocab', 'init', 'inputs', 'weights',
  'buffers', 'values', 'body', 'steps', 'backward', 'panels',
]);
const STEP_KEYS = new Set(['id', 'show', 'active', 'label', 'title', 'text']);
const PATCH_KEYS = new Set(['title', 'label', 'text', 'append']);
const PARAM_TYPES: ReadonlySet<string> = new Set(['int', 'float', 'dim', 'op', 'tex', 'text']);
const INITS = new Set(['ones', 'zeros', 'normal', 'uniform']);
const DEFAULT_INIT = 'uniform';
const DEFAULT_SEED = 1;
const INIT_DIGITS = 2;

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

function matrixTemplate(name: string, shape: readonly number[], format: string): string {
  if (shape.length === 0) return `{{${name}${format}}}`;
  const rows = shape.length === 1 ? 1 : shape[0];
  const cols = shape.length === 1 ? shape[0] : shape[1];
  const cell = (r: number, c: number) => {
    if (shape.length === 1) return `{{${name}[${c}]${format}}}`;
    if (cols === 1) return `{{${name}[${r}]${format}}}`;
    if (rows === 1) return `{{${name}[${c}]${format}}}`;
    return `{{${name}[${r}, ${c}]${format}}}`;
  };
  const lines = [...Array(rows).keys()].map((r) => [...Array(cols).keys()].map((c) => ` ${cell(r, c)} `).join('&'));
  return `\\begin{pmatrix}${lines.join('\\\\')}\\end{pmatrix}`;
}

class Compiler {
  readonly g: Graph;
  readonly nodes = new Map<string, Node>();
  readonly modules: SceneModule[] = [];
  readonly backward: { node: Node; text: BackwardOverride; where: Where }[] = [];
  private readonly values: Record<string, unknown>;
  private readonly usedValues = new Set<string>();
  private readonly random: () => number;
  private readonly real: boolean;
  private readonly namedDims = new Map<string, string>();
  private readonly buffers = new Map<string, Node>();
  private readonly owners = new Map<Node, LayoutEntry[]>();

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
        const nodes: Node[] = [];
        for (let i = from; i <= to; i++) {
          const target = this.keyOf({ k: 'index', obj: ast.obj, index: { k: 'num', v: i, pos: ast.pos }, pos: ast.pos }, env, offset);
          const found = target === undefined ? undefined : this.lookupKey(target, env);
          if (!(found instanceof Node)) this.fail(env, `'${target?.key ?? '?'}' is not a node`, offset + ast.pos);
          nodes.push(found);
        }
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
    const positional = args.map((value, i) => convert(def.args[i] ?? `#${i}`, value));
    const options: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(kwargs)) options[key] = convert(key, value);
    const attrs: Record<string, unknown> = { ...(def.defaults ?? {}), ...options };
    def.args.forEach((argName, i) => {
      if (positional[i] !== undefined && (!def.tensors.includes(argName) || typeof positional[i] === 'number')) attrs[argName] = positional[i];
    });
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
    return text.replace(/\\of\{([^{}]+)\}/g, (_match, ref: string) => this.ofValue(ref.trim(), env));
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
        if (param.type === 'tex' || param.type === 'text') return param.value as string;
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
      if (match === null) this.fail(env, `unknown template filter '${filter}' in '{{${source}}}'`);
      let target = source.slice(0, pipe).trim();
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

  param(decl: Record<string, unknown>, given: unknown, name: string, env: Env, parentEnv: Env | undefined, where: Where): { type: ParamType; value: Num | string | OpValue } {
    const type = decl.type;
    if (typeof type !== 'string' || !PARAM_TYPES.has(type)) fail(where, `parameter '${name}' needs a type: int, float, dim, op, tex or text`);
    const raw = given !== undefined ? given : decl.default;
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

  constructor(
    private readonly page: Record<string, unknown>,
    private readonly where: Where,
    rootDef: ModuleDef,
    rootDims: Map<string, Num>,
    extraDims: Map<string, Num>,
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
    for (const [name, spec] of Object.entries(inputs)) {
      const at = child(inputsWhere, inputs, name);
      if (!isRecord(spec)) fail(at, 'a page input needs tokens or value');
      const given = c.takeValue(name);
      let node: Node;
      const tex = defaultTex(name);
      try {
        if (Array.isArray(spec.tokens)) {
          if (!Array.isArray(vocab)) fail(at, 'tokens need a vocab list on the page');
          const ids = (spec.tokens as unknown[]).map((token, i) => {
            const id = (vocab as unknown[]).indexOf(token);
            if (id < 0) fail(child(child(at, spec, 'tokens'), spec.tokens as unknown[], i), `token '${String(token)}' is not in vocab`);
            return id;
          });
          node = c.g.label(name, given ?? ids, { tokens: vocab as string[], tex, elem: elemOf(tex) });
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
    const visible = [...new Set(forward.flatMap((s) => list(s.show)))].sort((a, b) => a.id - b.id);
    const merge = forward.map((s) => list(s.show).filter((n) => n.grad !== null)).filter((group) => group.length > 1);
    const override: Record<number, BackwardOverride> = {};
    if (c.backward.length > 0) {
      const groups = backwardGroups(c.g, visible, { auto: true, merge });
      for (const entry of c.backward) {
        const index = groups.findIndex((group) => group.includes(entry.node));
        if (index < 0) fail(entry.where, `node '${entry.node.name}' has no backward step`);
        override[index + 1] = entry.text;
      }
    }
    const overviewWhere = child(this.where, page, 'overview');
    const overview = page.overview === undefined ? undefined : record(page.overview, overviewWhere, 'overview');
    const stepsSpec: StepsSpec = {
      forward,
      backward: { auto: true, merge, override },
    };
    if (overview !== undefined) {
      stepsSpec.overview = {
        title: overview.title === undefined ? String(page.title) : c.text(overview.title, { ...env, where: child(overviewWhere, overview, 'title') }),
        text: c.text(overview.text, { ...env, where: child(overviewWhere, overview, 'text') }),
      };
    }
    this.warnDuplicateTex();
    return {
      graph: c.g,
      steps: stepsSpec,
      tree,
      modules: c.modules,
      panels: this.panels(env),
      nodes: c.nodes,
      page: {
        title: String(page.title),
        lede: page.lede === undefined ? undefined : c.text(page.lede, { ...env, where: child(this.where, page, 'lede') }),
        hint: typeof page.hint === 'string' ? page.hint : undefined,
        camera: isRecord(page.camera) ? (page.camera as CameraOptions) : undefined,
      },
    };
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

  params(scope: Scope, decl: Record<string, unknown>, env: Env, parentEnv: Env | undefined, given: Record<string, unknown>, where: Where): void {
    const paramsWhere = child(where, decl, 'params');
    const params = record(decl.params, paramsWhere, 'params');
    for (const key of Object.keys(given)) {
      if (!(key in params)) fail(child(parentEnv?.where ?? where, given, key), `module ${scope.def.name} has no parameter '${key}'`);
    }
    for (const [name, paramDecl] of Object.entries(params)) {
      const at = given[name] !== undefined && parentEnv !== undefined ? child(parentEnv.where, given, name) : child(paramsWhere, params, name);
      if (!isRecord(paramDecl)) fail(at, `parameter '${name}' must be a mapping`);
      const value = this.c.param(paramDecl, given[name], name, env, parentEnv, at);
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
      checkKeys(bufferDecl, new Set(['value', 'tex']), at);
      const valueWhere = child(at, bufferDecl, 'value');
      const value = c.expression(bufferDecl.value, { ...env, where: valueWhere, buffers: true }, undefined);
      if (!isRecord(value) || !('buffer' in value)) fail(valueWhere, 'a buffer value must call rope_cos, rope_sin or causal_mask');
      const buffer = value as BufferValue;
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
    items.forEach((item, index) => {
      const at = child(env.where, items, index);
      const itemEnv = { ...env, where: at };
      const before = c.g.nodes.length;
      if (!isRecord(item)) fail(at, 'a body item must be a mapping');
      if ('for' in item) {
        checkKeys(item, FOR_KEYS, at);
        if (typeof item.for !== 'string') fail(child(at, item, 'for'), 'for needs a variable name');
        if (!Array.isArray(item.body)) fail(child(at, item, 'body'), 'for needs a body list');
        const branches: LayoutTree[] = [];
        const iterations: Slot[][] = [];
        for (const v of this.range(item.in, { ...itemEnv, where: child(at, item, 'in') })) {
          const loopEnv: Env = { ...itemEnv, loops: new Map(env.loops).set(item.for, v), where: child(at, item, 'body') };
          const branch: LayoutTree = { kind: 'module', children: [] };
          const start = c.g.nodes.length;
          if (typeof item.brace === 'string') branch.title = c.ofText(item.brace, { ...loopEnv, where: child(at, item, 'brace') });
          iterations.push(this.items(scope, item.body, loopEnv, branch.children, counts, ordinals));
          branches.push(branch);
          if (branch.title !== undefined) this.addModule(branch.title, undefined, c.g.nodes.slice(start));
        }
        entries.push({ kind: 'parallel', branches });
        slots.push({ kind: 'loop', iterations });
        return;
      }
      const keys = Object.keys(item);
      if (keys.length !== 1) fail(at, `a body item must have exactly one name, got ${keys.length}`);
      const key = keys[0];
      const value = item[key];
      const valueWhere = child(at, item, key);
      const valueEnv = { ...itemEnv, where: valueWhere };
      const { name, base, indices } = c.concreteName(key, valueEnv);
      if (isRecord(value) && 'use' in value) {
        slots.push(this.instance(scope, name, indices, value, valueEnv, entries, counts, ordinals, before));
        return;
      }
      if (isRecord(value) && 'body' in value) {
        checkKeys(value, GROUP_KEYS, valueWhere);
        if (!Array.isArray(value.body)) fail(child(valueWhere, value, 'body'), 'body must be a list');
        if (scope.local(name) !== undefined || scope.groups.has(name)) fail(valueWhere, `name '${name}' is already defined`);
        scope.groups.add(name);
        const group: LayoutTree = { kind: 'module', title: name, children: [] };
        if (typeof value.subtitle === 'string') group.subtitle = value.subtitle;
        entries.push(group);
        const groupSlots = this.items(scope, value.body, { ...valueEnv, where: child(valueWhere, value, 'body') }, group.children, counts, ordinals);
        const created = c.g.nodes.slice(before);
        this.addModule(name, group.subtitle, created);
        slots.push({ kind: 'group', decl: value, where: valueWhere, slots: groupSlots });
        return;
      }
      slots.push(this.nodeItem(scope, name, base, indices, value, valueEnv, entries, before));
    });
    return slots;
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
    if (typeof use.suffix === 'string') mark = c.ofText(use.suffix, { ...env, where: child(env.where, use, 'suffix') });
    else if ((counts.get(def) ?? 0) > 1) mark = indices.length > 0 ? indices.join(',') : String(ordinal);
    const scope = new Scope(def, join(parent.path, name), mark === undefined ? [...parent.marks] : [mark, ...parent.marks]);
    parent.children.set(name, scope);
    const decl = def.decl;
    const where = def.where;
    checkKeys(decl, MODULE_KEYS, where);
    const childEnv: Env = { scope, loops: new Map(), where };
    const given = record(use.params, child(env.where, use, 'params'), 'params');
    this.params(scope, decl, childEnv, { ...env, where: child(env.where, use, 'params') }, given, where);
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
    const tree: LayoutTree = { kind: 'module', title: name, children: [] };
    if (typeof decl.subtitle === 'string') tree.subtitle = decl.subtitle;
    entries.push(tree);
    this.weights(scope, decl, childEnv, use.weights === undefined ? undefined : { map: record(use.weights, child(env.where, use, 'weights'), 'weights'), env: { ...env, where: child(env.where, use, 'weights') } }, tree.children);
    this.buffers(scope, decl, childEnv, tree.children);
    const { slots } = this.body(scope, decl, childEnv, tree.children);
    for (const output of outputs) {
      if (typeof output !== 'string' || !(scope.entries.get(output) instanceof Node)) fail(outputsWhere, `output '${String(output)}' of ${def.name} is not defined in its body`);
    }
    const steps = this.moduleSteps(scope, decl, childEnv, slots);
    this.moduleBackward(decl, childEnv);
    this.patch(scope, steps, use, env);
    for (const [output, target, at] of bindings) this.define(parent, target, scope.entries.get(output) as Node, { ...env, where: at });
    const created = c.g.nodes.slice(before);
    this.addModule(name, tree.subtitle, created);
    return { kind: 'child', steps, created };
  }

  match(scope: Scope, pattern: unknown, node: Node, env: Env, where: Where, givenAt: Where): void {
    if (!Array.isArray(pattern)) fail(where, 'shape must be a list');
    if (pattern.length !== node.shape.length) {
      fail(givenAt, `input has shape (${node.shape.join(', ')}), but ${scope.def.name} expects (${pattern.join(', ')})`);
    }
    pattern.forEach((entry, i) => {
      const actual: Num = { toy: node.shape[i], real: node.realShape[i] };
      if (typeof entry === 'string' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(entry) && !scope.dims.has(entry) && !scope.params.has(entry)) {
        scope.dims.set(entry, actual);
        return;
      }
      const expected = this.c.expression(entry, { ...env, where: child(where, pattern, i) }, undefined);
      if (!isNum(expected) || expected.toy !== actual.toy) {
        fail(givenAt, `input has shape (${node.shape.join(', ')}), but dimension ${i} of ${scope.def.name}'s pattern (${pattern.join(', ')}) is ${isNum(expected) ? expected.toy : '?'}`);
      }
    });
  }

  addModule(title: string, subtitle: string | undefined, nodes: Node[]): void {
    if (nodes.length === 0) return;
    const module: SceneModule = { title, nodes };
    if (subtitle !== undefined) module.subtitle = subtitle;
    this.c.modules.push(module);
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
      return { rec: { owner: scope.path, id: step.id, step: forward }, show };
    });
  }

  moduleSteps(scope: Scope, decl: Record<string, unknown>, env: Env, slots: Slot[]): StepRec[] {
    const containers: ResolvedStep[][] = [this.resolveSteps(scope, decl, env, 'steps')];
    const events: Extract<Slot, { kind: 'node' | 'child' }>[] = [];
    const flatten = (level: Slot[]) => {
      for (const slot of level) {
        if (slot.kind === 'group') {
          containers.push(this.resolveSteps(scope, slot.decl, { ...env, where: slot.where }, 'steps'));
          flatten(slot.slots);
        } else if (slot.kind === 'loop') {
          slot.iterations.forEach(flatten);
        } else {
          events.push(slot);
        }
      }
    };
    flatten(slots);
    const own = containers.flat();
    const covered = new Set<Node>(own.flatMap((s) => s.show));
    const anchors = new Map<ResolvedStep, number>();
    for (const container of containers) {
      let floor = -1;
      for (const step of container) {
        const first = events.findIndex((event) => step.show.some((n) => event.created.includes(n)));
        floor = Math.max(floor, first);
        anchors.set(step, floor);
      }
    }
    const placed = (i: number) => own.filter((step) => anchors.get(step) === i).map((step) => step.rec);
    const out = placed(-1);
    events.forEach((event, i) => {
      if (event.kind === 'child') out.push(...event.steps);
      else if (!covered.has(event.node)) out.push(this.autoStep(scope, event.name, event.node, covered));
      out.push(...placed(i));
    });
    return out;
  }

  autoStep(scope: Scope, name: string, node: Node, covered: Set<Node>): StepRec {
    const anon = scope.anon.get(node) ?? [];
    const leaves = [...anon, node]
      .flatMap((n) => n.inputs)
      .flatMap((input) => (input.op === 'transpose' && input.inputs[0].kind === 'param' ? [input.inputs[0], input] : [input]))
      .filter((input) => (input.kind === 'param' || input.kind === 'buffer' || (input.op === 'transpose' && input.inputs[0].kind === 'param')) && !covered.has(input));
    const show = [...new Set([...leaves, ...anon, node])];
    for (const n of leaves) covered.add(n);
    return { owner: scope.path, id: name, step: { show, title: name, label: name } };
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
      const target = steps.find((s) => s.owner === owner && s.id === id);
      if (target === undefined) fail(at, `step '${path}' does not exist in '${scope.path}'`);
      const fieldEnv = (key: string): Env => ({ ...env, where: child(at, spec, key) });
      if (spec.title !== undefined) target.step.title = c.text(spec.title, fieldEnv('title'));
      if (spec.label !== undefined) target.step.label = c.text(spec.label, fieldEnv('label'));
      if (spec.text !== undefined) target.step.text = c.text(spec.text, fieldEnv('text'));
      if (spec.append !== undefined) {
        const extra = c.text(spec.append, fieldEnv('append'));
        target.step.text = target.step.text ? `${target.step.text}\n${extra}` : extra;
      }
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
    return page.panels.map((panel, i): PanelItem => {
      const at = child(where, page.panels as unknown[], i);
      if (panel === 'loss') return 'loss';
      if (!isRecord(panel)) fail(at, "a panel must be 'loss' or a mapping");
      const out: Record<string, unknown> = { ...panel };
      const field = (key: string) => child(at, panel, key);
      for (const key of ['title', 'note']) if (typeof panel[key] === 'string') out[key] = textAt(panel[key], field(key));
      for (const key of ['tokens', 'matrix', 'sliders']) {
        if (panel[key] === undefined) continue;
        out[key] = nodeAt(panel[key], field(key));
      }
      if (panel.highlight !== undefined) {
        const highlight = panel.highlight;
        out.highlight = isRecord(highlight) ? { ...highlight, node: nodeAt(highlight.node, child(field('highlight'), highlight, 'node')) } : nodeAt(highlight, field('highlight'));
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
      if (isRecord(panel.loss)) out.loss = this.lossPanel(panel.loss, { ...env, where: field('loss') });
      return out as PanelItem;
    });
  }

  lossPanel(loss: Record<string, unknown>, env: Env): Record<string, unknown> {
    const c = this.c;
    const out: Record<string, unknown> = { ...loss };
    const at = (key: string, obj: object = loss) => ({ ...env, where: child(env.where, obj, key) });
    if (typeof loss.text === 'string') out.text = c.text(loss.text, at('text'));
    for (const key of ['sgd', 'random']) {
      const spec = loss[key];
      if (spec === undefined) continue;
      if (!isRecord(spec)) fail(at(key).where, `${key} must be a mapping`);
      const nodesKey = key === 'sgd' ? 'params' : 'nodes';
      const selected = spec[nodesKey];
      const specEnv = { ...env, where: child(env.where, loss, key) };
      const nodes = selected === 'all'
        ? c.g.nodes.filter((n) => n.kind === 'param')
        : list(selected as unknown).flatMap((s) => c.nodeList(s, { ...specEnv, where: child(specEnv.where, spec, nodesKey) }));
      out[key] = { ...spec, [nodesKey]: nodes };
    }
    return out;
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
  const rootDef: ModuleDef = { name: String(spec.title), file, decl: spec, where, namespace };
  const rootDims = rootDimsOf(spec, where);
  const extra = new Map<string, Num>();
  for (;;) {
    try {
      return new PageCompiler(spec, where, rootDef, rootDims, extra).run();
    } catch (error) {
      if (!(error instanceof NeedDim)) throw error;
      if (extra.has(error.name)) throw new Error(`dsl: dimension ${error.name} could not be registered`);
      extra.set(error.name, error.value);
    }
  }
}
