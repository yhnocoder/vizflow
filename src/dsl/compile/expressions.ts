import { DTYPES, type Dtype, Node, opDef } from '../../autograd';
import { explainOf } from '../../explain';
import { type Ast, type IndexItem, ParseError, parseExpr } from '../expr';
import { DslError } from '../source';
import { elemOf, numberTex } from '../tex';
import { CompilerState } from './state';
import { type DtypeValue, type Env, Instance, InstanceList, type Make, type ModuleRef, type Num, Result, type TensorView, type Value, type ViewIndex } from './types';
import { combine, isNum, isRecord, num } from './util';

const METHOD_CASTS: Record<string, Dtype> = { float: 'fp32', double: 'fp64', half: 'fp16', bfloat16: 'bf16' };

export function isDtype(value: unknown): value is DtypeValue {
  return isRecord(value) && typeof value.dtype === 'string' && Object.keys(value).length === 1;
}

export function isModuleRef(value: unknown): value is ModuleRef {
  return isRecord(value) && 'module' in value && Object.keys(value).length === 1;
}

export function isView(value: unknown): value is TensorView {
  return isRecord(value) && value.node instanceof Node && Array.isArray(value.index);
}

export function describe(value: Value): string {
  if (value instanceof Node) return `tensor '${value.name}'`;
  if (value instanceof Instance) return `module instance '${value.path}'`;
  if (value instanceof InstanceList) return `module list '${value.path}'`;
  if (value instanceof Result) return 'a result with named outputs';
  if (isNum(value)) return `the number ${value.toy}`;
  if (isDtype(value)) return `dtype ${value.dtype}`;
  if (isModuleRef(value)) return `module ${value.module.name}`;
  if (isView(value)) return `a part of tensor '${value.node.name}'`;
  if (Array.isArray(value)) return 'a list';
  return JSON.stringify(value);
}

export abstract class Expressions extends CompilerState {
  abstract callInstance(instance: Instance, args: Value[], kwargs: Map<string, Value>, env: Env, make: Make, top: boolean, at: number): Value;
  abstract callList(list: InstanceList, args: Value[], kwargs: Map<string, Value>, env: Env, make: Make, top: boolean, at: number): Value;
  abstract constantNode(fn: string, args: Value[], env: Env, make: Make, top: boolean, at: number): Node | undefined;

  parse(text: unknown, env: Env): Ast {
    if (typeof text === 'number') return { k: 'num', v: text, pos: 0 };
    if (typeof text === 'boolean') return { k: 'bool', v: text, pos: 0 };
    if (typeof text !== 'string') this.fail(env, 'expected an expression');
    try {
      return parseExpr(text);
    } catch (error) {
      if (error instanceof ParseError) this.fail(env, `${error.message} in '${text}'`, error.offset);
      throw error;
    }
  }

  expression(text: unknown, env: Env, make?: Make): Value {
    return this.evaluate(this.parse(text, env), env, make, true);
  }

  reference(text: unknown, env: Env): Value {
    return this.evaluate(this.parse(text, env), env, undefined, false);
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
    const local = env.locals.get(name);
    if (local !== undefined) return local;
    const call = env.call;
    if (call !== undefined) {
      const value = call.values.get(name) ?? call.bound.get(name) ?? env.instance.members.get(name);
      if (value !== undefined) return value;
    }
    const arg = env.instance.args.get(name);
    if (arg !== undefined) return arg;
    const module = env.instance.def.namespace.get(name);
    if (module !== undefined) return { module };
    if ((DTYPES as readonly string[]).includes(name)) return { dtype: name as Dtype };
    if (call === undefined && (env.instance.members.has(name) || (env.instance.def.decl.inputs as Record<string, unknown> | undefined)?.[name] !== undefined)) {
      this.fail(env, `'${name}' cannot be used here; init and args only see the module's args`, at);
    }
    return this.fail(env, `name '${name}' is not defined in ${env.instance.path === '' ? 'the page model' : `'${env.instance.path}' (${env.instance.def.name})`}`, at);
  }

  int(ast: Ast, env: Env, offset: number): number {
    const value = this.evaluate(ast, env, undefined, true);
    if (!isNum(value) || !Number.isInteger(value.toy)) this.fail(env, 'expected an integer', offset + ast.pos);
    return value.toy;
  }

  evaluate(ast: Ast, env: Env, make: Make | undefined, top: boolean): Value {
    switch (ast.k) {
      case 'num':
        return num(ast.v);
      case 'str':
        return ast.v;
      case 'bool':
        return ast.v;
      case 'name':
        return this.lookup(ast.name, env, ast.pos);
      case 'member':
        return this.member(ast, env, make);
      case 'index':
        return this.index(ast, env, make, top);
      case 'list':
        return ast.items.map((item) => this.evaluate(item, env, make, false));
      case 'neg': {
        const arg = this.evaluate(ast.arg, env, make, false);
        if (isNum(arg)) return combine([arg], (x) => -x);
        if (arg instanceof Node) return this.apply('neg', [arg], new Map(), env, make, top, ast.pos);
        return this.fail(env, `cannot negate ${describe(arg)}`, ast.pos);
      }
      case 'bin':
        return this.binary(ast, env, make, top);
      case 'call':
        return this.call(ast, env, make, top);
    }
  }

  member(ast: Extract<Ast, { k: 'member' }>, env: Env, make: Make | undefined): Value {
    const base = this.evaluate(ast.obj, env, make, false);
    if (base instanceof Instance) {
      const member = base.members.get(ast.name);
      if (member !== undefined) return member;
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
    if (base instanceof Node && ast.name === 'T') return this.transposed(base, env, make, ast.pos);
    if (base instanceof Node && ast.name === 'dtype') return { dtype: base.dtype };
    if (make === undefined && ast.name === 'grad' && (base instanceof Node || isView(base))) {
      const view = base instanceof Node ? { node: base, grad: false, index: [] } : base;
      if (view.grad) this.fail(env, "'.grad' is already taken", ast.pos);
      if (view.node.grad === null) this.fail(env, `${view.node.kind} '${view.node.name}' has no gradient`, ast.pos);
      return { ...view, grad: true };
    }
    return this.fail(env, `${describe(base)} has no member '${ast.name}'`, ast.pos);
  }

  onlyCall(instance: Instance, env: Env, at: number) {
    if (instance.calls.length === 0) this.fail(env, `'${instance.path}' is never called, so it has no results`, at);
    if (instance.calls.length > 1) this.fail(env, `'${instance.path}' is called ${instance.calls.length} times; name the call with { producer_of: <result> }`, at);
    return instance.calls[0];
  }

  transposed(node: Node, env: Env, make: Make | undefined, at: number): Node {
    if (make === undefined) {
      const existing = this.g.node(`${node.name}.T`);
      if (existing === undefined) this.fail(env, `'${node.name}.T' is not computed on this page`, at);
      return existing;
    }
    try {
      return node.T;
    } catch (error) {
      return this.fail(env, (error as Error).message, at);
    }
  }

  index(ast: Extract<Ast, { k: 'index' }>, env: Env, make: Make | undefined, top: boolean): Value {
    const base = this.evaluate(ast.obj, env, make, false);
    if (base instanceof InstanceList || Array.isArray(base)) {
      const items = base instanceof InstanceList ? base.items : base;
      if (ast.items.length !== 1 || ast.items[0].k !== 'at') this.fail(env, 'a list takes one index', ast.pos);
      const i = this.int(ast.items[0].value, env, 0);
      const at = i < 0 ? items.length + i : i;
      if (at < 0 || at >= items.length) this.fail(env, `index ${i} is out of range for ${items.length} items`, ast.pos);
      return items[at];
    }
    if (!(base instanceof Node) && !isView(base)) return this.fail(env, `cannot index ${describe(base)}`, ast.pos);
    const view: TensorView = base instanceof Node ? { node: base, grad: false, index: [] } : base;
    const shape = viewShape(view);
    const resolved = this.indexItems(ast.items, shape, env, ast.pos);
    if (make === undefined) return { ...view, index: composeIndex(view.index, view.node.shape, resolved) };
    if (!(base instanceof Node)) this.fail(env, 'cannot index a part of a tensor here', ast.pos);
    const lengths = this.sliceLengths(ast.items, base, env);
    let current: Node = base;
    let dim = 0;
    resolved.forEach((item, k) => {
      const last = k === resolved.length - 1;
      if (typeof item === 'number') {
        current = this.apply('select', [current, num(dim), num(item)], new Map(), env, make, top && last, ast.pos);
      } else {
        if (!(item.start === 0 && item.stop === current.shape[dim])) {
          current = this.apply('narrow', [current, num(dim), num(item.start), lengths[k] ?? num(item.stop - item.start)], new Map(), env, make, top && last, ast.pos);
        }
        dim++;
      }
    });
    if (current === base) this.fail(env, 'this index selects the whole tensor', ast.pos);
    return current;
  }

  sliceLengths(items: IndexItem[], node: Node, env: Env): (Num | undefined)[] {
    const ellipsis = items.findIndex((item) => item.k === 'ellipsis');
    const given = items.filter((item) => item.k !== 'ellipsis').length;
    const expanded: (IndexItem | undefined)[] = ellipsis < 0 ? [...items, ...Array(node.shape.length - given).fill(undefined)] : [...items.slice(0, ellipsis), ...Array(node.shape.length - given).fill(undefined), ...items.slice(ellipsis + 1)];
    return expanded.map((item, d) => {
      if (item === undefined || item.k !== 'slice') return undefined;
      const size: Num = { toy: node.shape[d], real: node.realShape[d], derive: (_sizes, shapes) => shapes[node.id][d] };
      const bound = (ast: Ast | undefined, fallback: Num): Num => {
        if (ast === undefined) return fallback;
        const value = this.evaluate(ast, env, undefined, true);
        if (!isNum(value)) return this.fail(env, 'a slice bound must be a number', ast.pos);
        return value.toy < 0 ? combine([size, value], (s, v) => s + v) : value;
      };
      return combine([bound(item.stop, size), bound(item.start, num(0))], (stop, start) => stop - start);
    });
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

  binary(ast: Extract<Ast, { k: 'bin' }>, env: Env, make: Make | undefined, top: boolean): Value {
    const a = this.evaluate(ast.a, env, make, false);
    const b = this.evaluate(ast.b, env, make, false);
    const at = ast.pos;
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
    if (ast.op === '@') {
      if (!(a instanceof Node) || !(b instanceof Node)) this.fail(env, "'@' needs two tensors", at);
      return this.apply('matmul', [a, b], new Map(), env, make, top, at);
    }
    if (ast.op === '**') {
      if (!(a instanceof Node) || !isNum(b)) this.fail(env, "'**' needs a tensor and a number", at);
      return this.apply('pow', [a, b], new Map(), env, make, top, at);
    }
    const name = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '//': 'floor_divide' }[ast.op];
    if (a instanceof Node && (b instanceof Node || isNum(b))) return this.apply(name, [a, b], new Map(), env, make, top, at);
    if (isNum(a) && b instanceof Node && (ast.op === '+' || ast.op === '*')) return this.apply(name, [b, a], new Map(), env, make, top, at);
    return this.fail(env, `'${ast.op}' cannot combine ${describe(a)} and ${describe(b)}`, at);
  }

  call(ast: Extract<Ast, { k: 'call' }>, env: Env, make: Make | undefined, top: boolean): Value {
    const at = ast.pos;
    const args = (): Value[] => ast.args.map((arg) => this.evaluate(arg, env, make, false));
    const kwargs = (): Map<string, Value> => new Map(ast.kwargs.map(([key, arg]) => [key, this.evaluate(arg, env, make, false)]));
    if (ast.fn.k === 'member') {
      const target = this.evaluate(ast.fn.obj, env, make, false);
      const method = ast.fn.name;
      if (!(target instanceof Node)) this.fail(env, `${describe(target)} has no method '${method}'`, ast.fn.pos);
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
        return this.apply('identity', [target], new Map(), env, make, top, at, dtype);
      }
      if (opDef(method) === undefined) this.fail(env, `tensors have no method '${method}'`, ast.fn.pos);
      return this.apply(method, [target, ...given], options, env, make, top, at);
    }
    if (ast.fn.k !== 'name') return this.fail(env, 'only names and methods can be called', at);
    const name = ast.fn.name;
    const callee = env.locals.get(name) ?? env.call?.values.get(name) ?? env.instance.members.get(name);
    if (callee instanceof Instance || callee instanceof InstanceList) {
      if (make === undefined) this.fail(env, `calling '${name}' runs a module; only forward can do that`, at);
      return callee instanceof Instance ? this.callInstance(callee, args(), kwargs(), env, make, top, at) : this.callList(callee, args(), kwargs(), env, make, top, at);
    }
    if (callee !== undefined) this.fail(env, `${describe(callee)} cannot be called`, at);
    const given = args();
    const options = kwargs();
    if (make !== undefined) {
      if (options.size > 0 && (name === 'arange' || name === 'causal_mask')) this.fail(env, `${name} takes no keyword arguments`, at);
      const constant = this.constantNode(name, given, env, make, top, at);
      if (constant !== undefined) return constant;
    }
    if (opDef(name) === undefined) this.fail(env, `unknown function '${name}'`, at);
    return this.apply(name, given, options, env, make, top, at);
  }

  apply(op: string, args: Value[], kwargs: Map<string, Value>, env: Env, make: Make | undefined, top: boolean, at: number, dtype?: Dtype): Node {
    if (make === undefined) this.fail(env, 'this expression computes a new tensor; only forward can do that', at);
    const def = opDef(op);
    if (def === undefined) this.fail(env, `unknown op '${op}'`, at);
    const convert = (argName: string, value: Value): unknown => {
      if (isNum(value)) return def.sizes?.includes(argName) ? this.dimSpec(value) : value.toy;
      if (Array.isArray(value) && value.every(isNum)) return value.map((n) => (def.sizes?.includes(argName) ? this.dimSpec(n) : n.toy));
      if (value instanceof Node || typeof value === 'boolean' || typeof value === 'string') return value;
      if (Array.isArray(value) && value.every((v) => v instanceof Node)) return value;
      return this.fail(env, `argument '${argName}' of ${op} cannot be ${describe(value)}`, at);
    };
    const restAt = def.rest === undefined ? -1 : def.args.indexOf(def.rest);
    const positional = args.map((value, i) => convert(restAt >= 0 && i >= restAt ? def.args[restAt] : (def.args[i] ?? `#${i}`), value));
    const options: Record<string, unknown> = {};
    for (const [key, value] of kwargs) options[key] = convert(key, value);
    const attrs: Record<string, unknown> = { ...(def.defaults ?? {}), ...options };
    def.args.forEach((argName, i) => {
      if (positional[i] !== undefined && (!def.tensors.includes(argName) || typeof positional[i] === 'number')) attrs[argName] = positional[i];
    });
    if (restAt >= 0 && positional.length > restAt) attrs[def.args[restAt]] = positional.slice(restAt);
    const name = top ? make.path : `${make.path}/${++make.counter}`;
    const operandTex = def.args.flatMap((argName, i) => {
      const value = positional[i];
      if (value instanceof Node) return [value.tex];
      if (Array.isArray(value) && value.every((n) => n instanceof Node)) return value.map((n) => n.tex);
      return def.tensors.includes(argName) && typeof value === 'number' ? [numberTex(value)] : [];
    });
    const tex = top ? make.tex : (explainOf(op)?.tex.forward({ out: '', in: operandTex, attrs }) ?? `\\mathrm{${op}}`);
    const elem = top ? elemOf(tex) : `(${elemOf(tex)})`;
    let node: Node;
    try {
      node = (this.g as unknown as Record<string, (...a: unknown[]) => Node>)[op](...positional, { ...options, name, tex, elem, ...(dtype === undefined ? {} : { dtype }) });
    } catch (error) {
      if (error instanceof DslError) throw error;
      return this.fail(env, (error as Error).message, at);
    }
    if (!top) make.anon.push(node);
    return node;
  }
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
