import { Node, opDef } from '../../autograd';
import { explainOf } from '../../explain';
import { type Ast, ParseError, parseExpr } from '../expr';
import { DslError } from '../source';
import { elemOf, numberTex } from '../tex';
import { BUFFER_ARITY, BUFFER_FUNCTIONS } from './buffer-functions';
import { CompilerState } from './state';
import { type Env, type Make, type Num, type OpValue, type Resolved, Scope, type Value } from './types';
import { isNum, isRecord, num } from './util';

function isOp(value: unknown): value is OpValue {
  return isRecord(value) && typeof value.op === 'string' && !(value instanceof Node);
}

export class ExpressionCompiler extends CompilerState {
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
      case 'list': {
        const items = ast.items.map((item) => this.evaluate(item, env, make, offset));
        if (items.length === 0 || !items.every((item) => item instanceof Node)) this.fail(env, 'a list must hold one or more nodes', offset + ast.pos);
        return items as Node[];
      }
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
      const arity = BUFFER_ARITY[fnName];
      if ((arity === 'any' ? args.length === 0 : args.length !== arity) || ast.kwargs.length > 0 || !args.every(isNum)) {
        this.fail(env, `${fnName} takes ${arity === 'any' ? 'one or more' : arity} numbers`, at);
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
      const dtype = top && make.dtype !== undefined ? { dtype: make.dtype } : {};
      node = (this.g as unknown as Record<string, (...a: unknown[]) => Node>)[op](...positional, { ...options, name, tex, elem, ...dtype });
    } catch (error) {
      if (error instanceof DslError) throw error;
      this.fail(env, (error as Error).message, at);
    }
    if (!top) make.anon.push(node);
    return node;
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
}
