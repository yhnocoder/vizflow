import { type AnyOpDef, type CompositeDef, DEFAULT_DTYPE, DTYPES, type Dtype, Node, ProgramBuilder, bindComposite, bindOp, checkDtype, compositeDef, opDef, outputDtype } from '../../autograd';
import { explainOf } from '../../explain';
import { type Ast, type IndexItem, ParseError, parseExpr } from '../expr';
import { DslError } from '../source';
import { elemOf } from '../tex';
import { CompilerState } from './state';
import { type DtypeValue, Eager, type Env, Instance, InstanceList, type Make, type Mode, type ModuleRef, type Num, PageFunction, type RealSizeEntry, type RealSizes, Result, type TensorView, type Value, type ViewIndex } from './types';
import { combine, isNum, isRecord, num } from './util';

const METHOD_CASTS: Record<string, Dtype> = { float: 'fp32', double: 'fp64', half: 'fp16', bfloat16: 'bf16' };
const REF: Mode = { k: 'ref' };
const OPS: Record<string, string> = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '//': 'floor_divide' };

type IndexStep = { op: 'select'; dim: number; index: number } | { op: 'narrow'; dim: number; start: Num; length: Num } | { op: 'unsqueeze'; dim: number };

export function isDtype(value: unknown): value is DtypeValue {
  return isRecord(value) && typeof value.dtype === 'string' && Object.keys(value).length === 1;
}

export function isModuleRef(value: unknown): value is ModuleRef {
  return isRecord(value) && 'module' in value && Object.keys(value).length === 1;
}

export function isView(value: unknown): value is TensorView {
  return isRecord(value) && value.node instanceof Node && Array.isArray(value.index);
}

export function isRealSizes(value: unknown): value is RealSizes {
  return isRecord(value) && 'realSizes' in value && Object.keys(value).length === 1;
}

export function isRealSizeEntry(value: unknown): value is RealSizeEntry {
  return isRecord(value) && typeof value.realSize === 'string' && 'values' in value;
}

function isTensor(value: unknown): value is Node | Eager {
  return value instanceof Node || value instanceof Eager;
}

function isNode(value: unknown): value is Node {
  return value instanceof Node;
}

function isEager(value: unknown): value is Eager {
  return value instanceof Eager;
}

function isNone(item: IndexItem | undefined): boolean {
  return item !== undefined && item.k === 'at' && item.value.k === 'none';
}

export function describe(value: Value): string {
  if (value === null) return 'None';
  if (value instanceof Node) return `tensor '${value.name}'`;
  if (value instanceof Eager) return `a computed tensor of shape (${value.value.shape.join(', ')})`;
  if (value instanceof PageFunction) return `function '${value.name}'`;
  if (value instanceof Instance) return `module instance '${value.path}'`;
  if (value instanceof InstanceList) return `module list '${value.path}'`;
  if (value instanceof Result) return 'a result with named outputs';
  if (isNum(value)) return `the number ${value.toy}`;
  if (isDtype(value)) return `dtype ${value.dtype}`;
  if (isModuleRef(value)) return `module ${value.module.name}`;
  if (isView(value)) return `a part of tensor '${value.node.name}'`;
  if (isRealSizes(value)) return 'the real sizes';
  if (isRealSizeEntry(value)) return `real size '${value.realSize}'`;
  if (Array.isArray(value)) return 'a list';
  return JSON.stringify(value);
}

export abstract class Expressions extends CompilerState {
  abstract callInstance(instance: Instance, args: Value[], kwargs: Map<string, Value>, env: Env, make: Make, top: boolean, at: number): Value;
  abstract callList(list: InstanceList, args: Value[], kwargs: Map<string, Value>, env: Env, make: Make, top: boolean, at: number): Value;
  private readonly parsed = new Map<string, Ast>();
  private readonly anonCount = new Map<string, number>();

  parse(text: unknown, env: Env): Ast {
    if (typeof text === 'number') return { k: 'num', v: text, pos: 0 };
    if (typeof text === 'boolean') return { k: 'bool', v: text, pos: 0 };
    if (text === null) return { k: 'none', pos: 0 };
    if (typeof text !== 'string') this.fail(env, 'expected an expression');
    const cached = this.parsed.get(text);
    if (cached !== undefined) return cached;
    try {
      const ast = parseExpr(text);
      this.parsed.set(text, ast);
      return ast;
    } catch (error) {
      if (error instanceof ParseError) this.fail(env, `${error.message} in '${text}'`, error.offset);
      throw error;
    }
  }

  expression(text: unknown, env: Env, make?: Make): Value {
    return this.evaluate(this.parse(text, env), env, make === undefined ? REF : { k: 'graph', make }, true);
  }

  reference(text: unknown, env: Env): Value {
    return this.evaluate(this.parse(text, env), env, REF, false);
  }

  computed(text: unknown, env: Env, program: ProgramBuilder, reads: boolean): Value {
    const mode: Mode = { k: 'eager', program, reads };
    const value = this.evaluate(this.parse(text, env), env, mode, true);
    return value instanceof Node ? this.read(value, false, env, mode, 0) : value;
  }

  node(text: unknown, env: Env): Node {
    const found = this.reference(text, env);
    if (!(found instanceof Node)) this.fail(env, `'${String(text)}' is ${describe(found)}, not a tensor`);
    return found;
  }

  number(text: unknown, env: Env): Num {
    const found = this.expression(text, env);
    if (!isNum(found)) this.fail(env, `'${String(text)}' is ${describe(found)}, not a number`);
    return found;
  }

  lookup(name: string, env: Env, at: number): Value {
    if (env.locals.has(name)) return env.locals.get(name) as Value;
    const call = env.call;
    if (call !== undefined) {
      if (call.values.has(name)) return call.values.get(name) as Value;
      const bound = call.bound.get(name);
      if (bound !== undefined) return bound;
      if (env.instance.members.has(name)) return env.instance.members.get(name) as Value;
    }
    const arg = env.instance.args.get(name);
    if (arg !== undefined) return arg;
    if (name === 'real' && env.instance.path === '' && this.realSizes !== undefined) return { realSizes: this.realSizes };
    const module = env.instance.def.namespace.get(name);
    if (module !== undefined) return { module };
    if ((DTYPES as readonly string[]).includes(name)) return { dtype: name as Dtype };
    if (call === undefined && (env.instance.members.has(name) || (env.instance.def.decl.inputs as Record<string, unknown> | undefined)?.[name] !== undefined)) {
      this.fail(env, `'${name}' cannot be used here; init and args only see the module's args`, at);
    }
    return this.fail(env, `name '${name}' is not defined in ${env.instance.path === '' ? 'the page model' : `'${env.instance.path}' (${env.instance.def.name})`}`, at);
  }

  int(ast: Ast, env: Env, offset: number): number {
    const value = this.evaluate(ast, env, REF, true);
    if (!isNum(value) || !Number.isInteger(value.toy)) this.fail(env, 'expected an integer', offset + ast.pos);
    return value.toy;
  }

  evaluate(ast: Ast, env: Env, mode: Mode, top: boolean): Value {
    switch (ast.k) {
      case 'num':
        return num(ast.v);
      case 'str':
        return ast.v;
      case 'bool':
        return ast.v;
      case 'none':
        return null;
      case 'name':
        return this.lookup(ast.name, env, ast.pos);
      case 'member':
        return this.member(ast, env, mode);
      case 'index':
        return this.index(ast, env, mode, top);
      case 'list':
        return ast.items.map((item) => this.evaluate(item, env, mode, false));
      case 'neg': {
        const arg = this.evaluate(ast.arg, env, mode, false);
        if (isNum(arg)) return combine([arg], (x) => -x);
        if (isTensor(arg)) return this.apply('neg', [arg], new Map(), env, mode, top, ast.pos);
        return this.fail(env, `cannot negate ${describe(arg)}`, ast.pos);
      }
      case 'bin':
        return this.binary(ast, env, mode, top);
      case 'call':
        return this.call(ast, env, mode, top);
    }
  }

  member(ast: Extract<Ast, { k: 'member' }>, env: Env, mode: Mode): Value {
    const base = this.evaluate(ast.obj, env, mode, false);
    if (base instanceof Instance) {
      if (base.members.has(ast.name)) return base.members.get(ast.name) as Value;
      const call = this.onlyCall(base, env, ast.pos);
      const value = call.values.get(ast.name);
      if (value === undefined) this.fail(env, `'${ast.name}' is not defined in '${base.path}' (${base.def.name})`, ast.pos);
      return value;
    }
    if (base instanceof Result) {
      const output = base.outputs.get(ast.name);
      if (output === undefined) this.fail(env, `${base.call.instance.def.name} has no output '${ast.name}'; its outputs are ${[...base.outputs.keys()].join(', ')}`, ast.pos);
      return output;
    }
    if (isRealSizeEntry(base)) {
      const value = base.values[ast.name];
      if (value === undefined) this.fail(env, `real size '${base.realSize}' has no '${ast.name}'; it has ${Object.keys(base.values).join(', ')}`, ast.pos);
      return typeof value === 'number' ? num(value) : value;
    }
    if (isTensor(base) && ast.name === 'dtype') return { dtype: base.dtype };
    if (isTensor(base) && ast.name === 'T') return this.transposed(base, env, mode, ast.pos);
    if (ast.name === 'grad' && (base instanceof Node || isView(base))) {
      if (mode.k === 'graph') this.fail(env, "'.grad' is not available in forward", ast.pos);
      const view = base instanceof Node ? { node: base, grad: false, index: [] } : base;
      if (view.grad) this.fail(env, "'.grad' is already taken", ast.pos);
      if (view.node.grad === null) this.fail(env, `${view.node.kind} '${view.node.name}' has no gradient`, ast.pos);
      if (mode.k === 'eager') {
        if (view.index.length > 0) this.fail(env, "'.grad' of a part of a tensor cannot be computed with", ast.pos);
        return this.read(view.node, true, env, mode, ast.pos);
      }
      return { ...view, grad: true };
    }
    return this.fail(env, `${describe(base)} has no member '${ast.name}'`, ast.pos);
  }

  onlyCall(instance: Instance, env: Env, at: number) {
    if (instance.calls.length === 0) this.fail(env, `'${instance.path}' is never called, so it has no results`, at);
    if (instance.calls.length > 1) this.fail(env, `'${instance.path}' is called ${instance.calls.length} times, so the name does not pick one call`, at);
    return instance.calls[0];
  }

  transposed(base: Node | Eager, env: Env, mode: Mode, at: number): Value {
    if (base instanceof Node && mode.k === 'ref') {
      const existing = this.g.node(`${base.name}.T`);
      if (existing === undefined) this.fail(env, `'${base.name}.T' is not computed on this page`, at);
      return existing;
    }
    if (base instanceof Node && mode.k === 'graph') {
      try {
        return base.T;
      } catch (error) {
        return this.fail(env, (error as Error).message, at);
      }
    }
    return this.apply('transpose', [base, num(0), num(1)], new Map(), env, mode, false, at);
  }

  read(node: Node, grad: boolean, env: Env, mode: Mode, at: number): Eager {
    if (mode.k !== 'eager' || !mode.reads) return this.fail(env, `tensor '${node.name}' cannot be read here; this value only uses numbers and args`, at);
    const value = grad ? node.grad : node.value;
    if (value === null) return this.fail(env, `${node.kind} '${node.name}' has no gradient`, at);
    return new Eager(value, node.dtype, mode.program.read(node.name, grad));
  }

  index(ast: Extract<Ast, { k: 'index' }>, env: Env, mode: Mode, top: boolean): Value {
    const base = this.evaluate(ast.obj, env, mode, false);
    if (base instanceof InstanceList || Array.isArray(base)) return this.listIndex(base instanceof InstanceList ? base.items : base, base instanceof InstanceList, ast, env);
    if (isRealSizes(base)) {
      const key = ast.items.length === 1 && ast.items[0].k === 'at' ? this.evaluate(ast.items[0].value, env, REF, false) : undefined;
      if (typeof key !== 'string') this.fail(env, 'real takes one name, such as real[\'qwen3-8b\']', ast.pos);
      const values = base.realSizes[key];
      if (values === undefined) this.fail(env, `real has no '${key}'; it has ${Object.keys(base.realSizes).join(', ')}`, ast.pos);
      return { realSize: key, values };
    }
    if (mode.k === 'ref') {
      if (!(base instanceof Node) && !isView(base)) return this.fail(env, `cannot index ${describe(base)}`, ast.pos);
      if (ast.items.some(isNone)) this.fail(env, 'None adds a dimension, which only forward and computed values can do', ast.pos);
      const view: TensorView = base instanceof Node ? { node: base, grad: false, index: [] } : base;
      return { ...view, index: composeIndex(view.index, view.node.shape, this.indexItems(ast.items, viewShape(view), env, ast.pos)) };
    }
    if (!isTensor(base)) return this.fail(env, `cannot index ${describe(base)}`, ast.pos);
    const steps = this.indexSteps(base, ast, env, mode);
    if (steps.length === 0) {
      if (mode.k === 'graph') this.fail(env, 'this index selects the whole tensor', ast.pos);
      return base;
    }
    let current: Value = base;
    steps.forEach((step, k) => {
      const args: Value[] = step.op === 'narrow' ? [current, num(step.dim), step.start, step.length] : step.op === 'select' ? [current, num(step.dim), num(step.index)] : [current, num(step.dim)];
      current = this.apply(step.op, args, new Map(), env, mode, top && k === steps.length - 1, ast.pos);
    });
    return current;
  }

  private listIndex(items: readonly Value[], modules: boolean, ast: Extract<Ast, { k: 'index' }>, env: Env): Value {
    const item = ast.items[0];
    if (ast.items.length !== 1 || item.k === 'ellipsis' || isNone(item)) this.fail(env, 'a list takes one index or slice', ast.pos);
    const clamp = (v: number) => Math.min(items.length, Math.max(0, v < 0 ? items.length + v : v));
    if (item.k === 'slice') {
      if (modules) this.fail(env, 'a module list takes one index', ast.pos);
      const start = item.start === undefined ? 0 : clamp(this.int(item.start, env, 0));
      const stop = item.stop === undefined ? items.length : clamp(this.int(item.stop, env, 0));
      return items.slice(start, stop);
    }
    const i = this.int(item.value, env, 0);
    const at = i < 0 ? items.length + i : i;
    if (at < 0 || at >= items.length) this.fail(env, `index ${i} is out of range for ${items.length} items`, ast.pos);
    return items[at];
  }

  private indexSteps(base: Node | Eager, ast: Extract<Ast, { k: 'index' }>, env: Env, mode: Mode): IndexStep[] {
    const shape = base instanceof Node ? base.shape : base.value.shape;
    const items = ast.items;
    if (items.filter((item) => item.k === 'ellipsis').length > 1) this.fail(env, "an index can use '...' once", ast.pos);
    const consumed = items.filter((item) => item.k !== 'ellipsis' && !isNone(item)).length;
    if (consumed > shape.length) this.fail(env, `${consumed} indices for a tensor with shape (${shape.join(', ')})`, ast.pos);
    const ellipsis = items.findIndex((item) => item.k === 'ellipsis');
    const fill: undefined[] = Array(shape.length - consumed).fill(undefined);
    const expanded: (IndexItem | undefined)[] = ellipsis < 0 ? [...items, ...fill] : [...items.slice(0, ellipsis), ...fill, ...items.slice(ellipsis + 1)];
    const steps: IndexStep[] = [];
    let dim = 0;
    let baseDim = 0;
    for (const item of expanded) {
      if (isNone(item)) {
        steps.push({ op: 'unsqueeze', dim: dim++ });
        continue;
      }
      const d = baseDim++;
      const toy = shape[d];
      if (item !== undefined && item.k === 'at') {
        const i = this.int(item.value, env, 0);
        if (i < -toy || i >= toy) this.fail(env, `index ${i} is out of range for dimension ${d} of size ${toy}`, ast.pos);
        steps.push({ op: 'select', dim, index: i });
        continue;
      }
      const slice = item?.k === 'slice' ? item : undefined;
      if (slice === undefined || (slice.start === undefined && slice.stop === undefined)) {
        dim++;
        continue;
      }
      const size: Num = base instanceof Node && mode.k === 'graph' ? { toy, real: base.realShape[d], derive: (_sizes, shapes) => shapes[base.id][d] } : num(toy);
      const bound = (bound: Ast | undefined, fallback: Num): Num => {
        if (bound === undefined) return fallback;
        const value = this.evaluate(bound, env, REF, true);
        if (!isNum(value) || !Number.isInteger(value.toy)) return this.fail(env, 'a slice bound must be an integer', bound.pos);
        return combine([size, value], (s, v) => Math.min(s, Math.max(0, v < 0 ? s + v : v)));
      };
      const start = bound(slice.start, num(0));
      const length = combine([bound(slice.stop, size), start], (stop, from) => Math.max(0, stop - from));
      steps.push({ op: 'narrow', dim: dim++, start, length });
    }
    return steps;
  }

  indexItems(items: IndexItem[], shape: readonly number[], env: Env, at: number): ViewIndex[] {
    const ellipsis = items.findIndex((item) => item.k === 'ellipsis');
    const given = items.filter((item) => item.k !== 'ellipsis');
    if (given.length > shape.length) this.fail(env, `${given.length} indices for a tensor with shape (${shape.join(', ')})`, at);
    const expanded: (IndexItem | undefined)[] = ellipsis < 0 ? [...given, ...Array(shape.length - given.length).fill(undefined)] : [...items.slice(0, ellipsis), ...Array(shape.length - given.length).fill(undefined), ...items.slice(ellipsis + 1)];
    return expanded.map((item, d): ViewIndex => {
      const size = shape[d];
      if (item === undefined || item.k === 'ellipsis') return { start: 0, stop: size };
      const bound = (ast: Ast | undefined, fallback: number) => {
        if (ast === undefined) return fallback;
        const v = this.int(ast, env, 0);
        return Math.min(size, Math.max(0, v < 0 ? size + v : v));
      };
      if (item.k === 'slice') return { start: bound(item.start, 0), stop: bound(item.stop, size) };
      const i = this.int(item.value, env, 0);
      const k = i < 0 ? size + i : i;
      if (k < 0 || k >= size) this.fail(env, `index ${i} is out of range for dimension ${d} of size ${size}`, at);
      return k;
    });
  }

  binary(ast: Extract<Ast, { k: 'bin' }>, env: Env, mode: Mode, top: boolean): Value {
    const a = this.evaluate(ast.a, env, mode, false);
    const b = this.evaluate(ast.b, env, mode, false);
    const at = ast.pos;
    const none = new Map<string, Value>();
    if (isNum(a) && isNum(b)) {
      const both = (fn: (x: number, y: number) => number): Num => combine([a, b], fn);
      switch (ast.op) {
        case '+': return both((x, y) => x + y);
        case '-': return both((x, y) => x - y);
        case '*': return both((x, y) => x * y);
        case '/': return both((x, y) => x / y);
        case '//': return both((x, y) => Math.floor(x / y));
        case '**': return both((x, y) => x ** y);
        default: return this.fail(env, "'@' needs two tensors", at);
      }
    }
    const operand = (v: Value) => isTensor(v) || isNum(v);
    if (ast.op === '@') {
      if (!isTensor(a) || !isTensor(b)) this.fail(env, "'@' needs two tensors", at);
      return this.apply('matmul', [a, b], none, env, mode, top, at);
    }
    if (ast.op === '**' && operand(a) && operand(b)) return this.apply('pow', [a, b], none, env, mode, top, at);
    if (isTensor(a) && operand(b) && ast.op !== '**') return this.apply(OPS[ast.op], [a, b], none, env, mode, top, at);
    if (isNum(a) && isTensor(b)) {
      if (ast.op === '+' || ast.op === '*') return this.apply(OPS[ast.op], [b, a], none, env, mode, top, at);
      if (ast.op === '-') return this.apply('rsub', [b, a], none, env, mode, top, at);
      if (ast.op === '/') return this.apply('mul', [this.apply('reciprocal', [b], none, env, mode, false, at), a], none, env, mode, top, at);
    }
    return this.fail(env, `'${ast.op}' cannot combine ${describe(a)} and ${describe(b)}`, at);
  }

  call(ast: Extract<Ast, { k: 'call' }>, env: Env, mode: Mode, top: boolean): Value {
    const at = ast.pos;
    const args = (): Value[] => ast.args.map((arg) => this.evaluate(arg, env, mode, false));
    const kwargs = (): Map<string, Value> => new Map(ast.kwargs.map(([key, arg]) => [key, this.evaluate(arg, env, mode, false)]));
    if (ast.fn.k === 'member') {
      const target = this.evaluate(ast.fn.obj, env, mode, false);
      const method = ast.fn.name;
      if (!isTensor(target)) this.fail(env, `${describe(target)} has no method '${method}'`, ast.fn.pos);
      const given = args();
      const options = kwargs();
      if (method in METHOD_CASTS || method === 'to') {
        let dtype: Dtype;
        if (method === 'to') {
          if (given.length !== 1 || !isDtype(given[0]) || options.size > 0) this.fail(env, "'.to' takes one dtype, such as x.to(fp32) or x.to(y.dtype)", at);
          dtype = given[0].dtype;
        } else {
          if (given.length > 0 || options.size > 0) this.fail(env, `'.${method}()' takes no arguments`, at);
          dtype = METHOD_CASTS[method];
        }
        if (dtype === target.dtype) return target;
        return this.apply('identity', [target], new Map(), env, mode, top, at, dtype);
      }
      if (opDef(method) === undefined && compositeDef(method) === undefined) this.fail(env, `tensors have no method '${method}'`, ast.fn.pos);
      return this.apply(method, [target, ...given], options, env, mode, top, at);
    }
    if (ast.fn.k !== 'name') return this.fail(env, 'only names and methods can be called', at);
    const name = ast.fn.name;
    const callee = env.locals.has(name) ? env.locals.get(name) : (env.call?.values.get(name) ?? env.instance.members.get(name));
    if (callee instanceof Instance || callee instanceof InstanceList) {
      if (mode.k !== 'graph') this.fail(env, `calling '${name}' runs a module; only forward can do that`, at);
      return callee instanceof Instance ? this.callInstance(callee, args(), kwargs(), env, mode.make, top, at) : this.callList(callee, args(), kwargs(), env, mode.make, top, at);
    }
    if (callee instanceof PageFunction) return callee.run(args(), kwargs(), (message) => this.fail(env, message, at));
    if (callee !== undefined) this.fail(env, `${describe(callee)} cannot be called`, at);
    if (opDef(name) === undefined && compositeDef(name) === undefined) this.fail(env, `unknown function '${name}'`, at);
    return this.apply(name, args(), kwargs(), env, mode, top, at);
  }

  apply(op: string, args: Value[], kwargs: Map<string, Value>, env: Env, mode: Mode, top: boolean, at: number, dtype?: Dtype): Value {
    if (mode.k === 'ref') this.fail(env, 'this expression computes a new tensor, which only forward and computed values can do', at);
    const def = opDef(op);
    const composite = compositeDef(op);
    if (def === undefined && composite === undefined) this.fail(env, `unknown op '${op}'`, at);
    const names = (def ?? (composite as CompositeDef)).args;
    const restAt = def?.rest === undefined ? -1 : names.indexOf(def.rest);
    try {
      const positional = args.map((value, i) => this.convert(op, def, restAt >= 0 && i >= restAt ? names[restAt] : (names[i] ?? `#${i}`), value, env, mode, at));
      const options: Record<string, unknown> = {};
      for (const [key, value] of kwargs) options[key] = this.convert(op, def, key, value, env, mode, at);
      if (dtype !== undefined) options.dtype = dtype;
      if (composite !== undefined) return this.compose(composite, positional, options, mode, top);
      return this.applyOp(def as AnyOpDef, positional, options, mode, top);
    } catch (error) {
      if (error instanceof DslError) throw error;
      return this.fail(env, (error as Error).message, at);
    }
  }

  private convert(op: string, def: AnyOpDef | undefined, argName: string, value: Value, env: Env, mode: Mode, at: number): unknown {
    const size = def?.sizes?.includes(argName) ?? false;
    const one = (v: Value): unknown => {
      if (isNum(v)) return size && mode.k === 'graph' ? this.dimSpec(v) : v.toy;
      if (v instanceof Node) return mode.k === 'eager' ? this.read(v, false, env, mode, at) : v;
      if (v instanceof Eager) {
        if (mode.k !== 'eager') throw new Error(`argument '${argName}' of ${op} is a computed tensor, which forward cannot use`);
        return v;
      }
      if (v === null || typeof v === 'boolean' || typeof v === 'string') return v;
      if (isDtype(v)) return v.dtype;
      if (Array.isArray(v)) return v.map(one);
      throw new Error(`argument '${argName}' of ${op} cannot be ${describe(v)}`);
    };
    return one(value);
  }

  private compose(def: CompositeDef, positional: unknown[], options: Record<string, unknown>, mode: Mode, top: boolean): Value {
    const { inputs, rest } = bindComposite<Node | Eager>(def, positional, options, mode.k === 'graph' ? isNode : isEager);
    return def.compose<Node | Eager>((op, args, last) => this.applyOp(opDef(op) as AnyOpDef, args, last ? rest : {}, mode, top && last), inputs);
  }

  private applyOp(def: AnyOpDef, positional: unknown[], options: Record<string, unknown>, mode: Mode, top: boolean): Node | Eager {
    if (mode.k === 'eager') return this.eagerOp(def, positional, options, mode.program);
    if (mode.k !== 'graph') throw new Error(`${def.name} cannot be computed here`);
    const make = mode.make;
    const { inputs, attrs } = bindOp(def, positional, options, isNode);
    const texAttrs = Object.fromEntries(Object.entries(attrs).map(([key, value]) => [key, def.sizes?.includes(key) ? toySize(value, this.g.dims) : value]));
    const count = top ? 0 : (this.anonCount.get(make.path) ?? 0) + 1;
    if (!top) this.anonCount.set(make.path, count);
    const name = top ? make.path : `${make.path}/${count}`;
    const tex = top ? make.tex : (explainOf(def.name)?.tex.forward({ out: '', in: inputs.map((n) => n.tex), attrs: texAttrs }) ?? `\\mathrm{${def.name}}`);
    const elem = top ? elemOf(tex) : `(${elemOf(tex)})`;
    const node = (this.g as unknown as Record<string, (...a: unknown[]) => Node>)[def.name](...positional, { ...options, name, tex, elem });
    if (!top) make.anon.push(node);
    return node;
  }

  private eagerOp(def: AnyOpDef, positional: unknown[], options: Record<string, unknown>, program: ProgramBuilder): Eager {
    const { inputs, attrs, rest } = bindOp(def, positional, options, isEager);
    const { dtype: given, ...unknown } = rest;
    const extra = Object.keys(unknown);
    if (extra.length > 0) throw new Error(`${def.name}: unknown option '${extra[0]}'`);
    def.shape(inputs.map((e) => [...e.value.shape]), attrs);
    const value = def.forward(inputs.map((e) => e.value), attrs);
    const named = inputs.map((e) => ({ name: describe(e), dtype: e.dtype }));
    const dtype = outputDtype(def, named, attrs, DEFAULT_DTYPE, def.name, given === undefined ? undefined : checkDtype(String(given), def.name));
    const refs = inputs.map((e) => e.ref);
    return new Eager(value, dtype, refs.every((ref) => ref !== undefined) ? program.op(def.name, refs as number[], attrs) : undefined);
  }
}

function toySize(value: unknown, dims: Readonly<Record<string, number>>): unknown {
  if (Array.isArray(value)) return value.map((v) => toySize(v, dims));
  if (typeof value === 'string') return dims[value] ?? value;
  if (isRecord(value) && typeof value.toy === 'number') return value.toy;
  return value;
}

export function viewShape(view: TensorView): number[] {
  return view.node.shape.flatMap((size, d) => {
    const item = view.index[d];
    if (item === undefined) return [size];
    return typeof item === 'number' ? [] : [item.stop - item.start];
  });
}

function composeIndex(current: ViewIndex[], shape: readonly number[], next: ViewIndex[]): ViewIndex[] {
  const out: ViewIndex[] = [];
  let k = 0;
  shape.forEach((size, d) => {
    const item = current[d] ?? { start: 0, stop: size };
    if (typeof item === 'number') {
      out.push(item);
      return;
    }
    const inner = next[k++] ?? { start: 0, stop: item.stop - item.start };
    out.push(typeof inner === 'number' ? item.start + inner : { start: item.start + inner.start, stop: item.start + inner.stop });
  });
  return out;
}
