import { DTYPES, type Dtype, Node } from '../../autograd';
import type { Nested } from '../../tensor';
import type { Ast } from '../expr';
import { type ModuleDef, PACKAGE_PREFIX } from '../load';
import { DslError, type Where, child, fail } from '../source';
import { addMark, defaultTex, elemOf } from '../tex';
import { Expressions, describe, isDtype, isModuleRef } from './expressions';
import { type Alias, Call, type DtypeValue, type Env, Instance, InstanceList, ListCall, type Make, type Num, Result, type Value } from './types';
import { checkKeys, combine, isNum, isRecord, join, num, record } from './util';

const ARG_TYPES: ReadonlySet<string> = new Set(['dim', 'int', 'float', 'bool', 'dtype', 'module']);
const INITS: ReadonlySet<string> = new Set(['ones', 'zeros', 'normal', 'uniform']);
const DEFAULT_INIT = 'uniform';
const MODULE_KEYS: ReadonlySet<string> = new Set(['module', 'import', 'args', 'inputs', 'outputs', 'init', 'forward', 'tex', 'formula', 'fold', 'steps', 'unshown']);
const CHILD_KEYS: ReadonlySet<string> = new Set(['use', 'args', 'count']);
const WEIGHT_KEYS: ReadonlySet<string> = new Set(['shape', 'init']);
const BUFFER_KEYS: ReadonlySet<string> = new Set(['type', 'shape', 'dtype', 'value']);
const FIELD_KEYS: ReadonlySet<string> = new Set(['type', 'shape', 'dtype']);
const LIST_USE = /^list\[(.+)\]$/;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

type Constant = number | Constant[];

export const LINEAR = 'Linear';

export function listMark(i: number): string {
  return `^{[${i + 1}]}`;
}

function mapConstant(a: Constant, fn: (x: number) => number): Constant {
  return Array.isArray(a) ? a.map((x) => mapConstant(x, fn)) : fn(a);
}

function zipConstant(a: Constant, b: Constant, fn: (x: number, y: number) => number): Constant {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) throw new Error(`cannot combine lists of length ${a.length} and ${b.length}`);
    return a.map((x, i) => zipConstant(x, b[i], fn));
  }
  if (Array.isArray(a)) return a.map((x) => zipConstant(x, b, fn));
  if (Array.isArray(b)) return b.map((y) => zipConstant(a, y, fn));
  return fn(a, b);
}

function shapeOfConstant(value: Constant): number[] {
  return Array.isArray(value) ? [value.length, ...(value.length > 0 ? shapeOfConstant(value[0]) : [])] : [];
}

export abstract class Modules extends Expressions {
  readonly calls: Call[] = [];
  readonly producers = new Map<Node, Call>();
  readonly anon = new Map<Node, Node[]>();
  readonly instances: Instance[] = [];

  instanceEnv(instance: Instance, where: Where = instance.def.where): Env {
    return { instance, call: undefined, locals: new Map(), where };
  }

  callEnv(call: Call, where: Where = call.instance.def.where): Env {
    return { instance: call.instance, call, locals: new Map(), where };
  }

  decl(def: ModuleDef): Record<string, unknown> {
    return def.decl;
  }

  fieldOf(def: ModuleDef, key: string): Record<string, unknown> {
    return record(def.decl[key], child(def.where, def.decl, key), key);
  }

  instantiate(def: ModuleDef, path: string, name: string, parent: Instance | undefined, marks: readonly string[], given: Map<string, { value: Value; where: Where }>, where: Where): Instance {
    checkKeys(def.decl, MODULE_KEYS, def.where);
    const instance = new Instance(def, path, name, parent, marks);
    this.instances.push(instance);
    const argsWhere = child(def.where, def.decl, 'args');
    const declared = this.fieldOf(def, 'args');
    for (const [key, { where: at }] of given) {
      if (!(key in declared)) fail(at, `module ${def.name} has no arg '${key}'; its args are ${Object.keys(declared).join(', ') || '(none)'}`);
    }
    const env = this.instanceEnv(instance, argsWhere);
    for (const [key, spec] of Object.entries(declared)) {
      const at = child(argsWhere, declared, key);
      const { type, fallback } = argSpec(spec, at);
      const provided = given.get(key);
      let value: Value;
      if (provided !== undefined) value = provided.value;
      else if (fallback !== undefined) value = this.expression(fallback, { ...env, where: child(at, spec as object, 'default') });
      else return fail(where, `module ${def.name} needs arg '${key}'`);
      instance.args.set(key, checkArg(type, key, value, provided?.where ?? at));
    }
    if (def.name === LINEAR && def.file.startsWith(PACKAGE_PREFIX) && instance.args.get('bias') === true) {
      fail(given.get('bias')?.where ?? where, 'Linear with bias: true is not supported yet; pass bias: false');
    }
    this.initMembers(instance);
    return instance;
  }

  private initMembers(instance: Instance): void {
    const def = instance.def;
    const initWhere = child(def.where, def.decl, 'init');
    const init = this.fieldOf(def, 'init');
    for (const [key, spec] of Object.entries(init)) {
      const at = child(initWhere, init, key);
      if (key === 'weights') {
        for (const [weight, decl] of Object.entries(record(spec, at, 'weights'))) this.weight(instance, weight, decl, child(at, spec as object, weight));
      } else if (key === 'buffers') {
        for (const [buffer, decl] of Object.entries(record(spec, at, 'buffers'))) this.buffer(instance, buffer, decl, child(at, spec as object, buffer));
      } else {
        this.childMember(instance, key, spec, at);
      }
    }
  }

  private define(instance: Instance, name: string, value: Instance | InstanceList | Node, where: Where): void {
    if (instance.members.has(name) || instance.args.has(name)) fail(where, `name '${name}' is already defined in ${instance.def.name}`);
    instance.members.set(name, value);
  }

  private shape(spec: unknown, env: Env): Num[] {
    if (!Array.isArray(spec)) this.fail(env, 'shape must be a list');
    return spec.map((entry, i) => {
      const at = child(env.where, spec, i);
      const value = this.number(entry, { ...env, where: at });
      if (!Number.isInteger(value.toy) || value.toy <= 0) fail(at, `shape entry ${JSON.stringify(entry)} is not a positive integer`);
      return value;
    });
  }

  private weight(instance: Instance, name: string, decl: unknown, at: Where): void {
    const spec = record(decl, at, `weight '${name}'`);
    checkKeys(spec, WEIGHT_KEYS, at);
    const env = this.instanceEnv(instance, child(at, spec, 'shape'));
    const shape = this.shape(spec.shape, env);
    const init = spec.init ?? DEFAULT_INIT;
    if (typeof init !== 'string' || !INITS.has(init)) fail(child(at, spec, 'init'), `init must be one of ${[...INITS].join(', ')}`);
    const path = join(instance.path, name);
    const tex = this.memberTex(instance, name);
    let node: Node;
    try {
      node = this.g.param(path, shape.map((d) => this.dimSpec(d)), this.initValues(init, shape.map((d) => d.toy)), { tex, elem: elemOf(tex) });
    } catch (error) {
      return fail(at, (error as Error).message);
    }
    this.define(instance, name, node, at);
    this.nodes.set(path, node);
  }

  private buffer(instance: Instance, name: string, decl: unknown, at: Where): void {
    const spec = record(decl, at, `buffer '${name}'`);
    checkKeys(spec, BUFFER_KEYS, at);
    if (spec.type !== 'tensor') fail(child(at, spec, 'type'), "a buffer needs type: tensor");
    const env = this.instanceEnv(instance, child(at, spec, 'shape'));
    const shape = this.shape(spec.shape, env);
    const dtype = this.expression(spec.dtype, { ...env, where: child(at, spec, 'dtype') });
    if (!isDtype(dtype)) fail(child(at, spec, 'dtype'), `dtype must be one of ${DTYPES.join(', ')}`);
    const valueWhere = child(at, spec, 'value');
    let values: Constant;
    try {
      values = this.constant(this.parse(spec.value, { ...env, where: valueWhere }), { ...env, where: valueWhere });
    } catch (error) {
      if (error instanceof DslError) throw error;
      return fail(valueWhere, (error as Error).message);
    }
    const actual = shapeOfConstant(values);
    if (actual.join(',') !== shape.map((d) => d.toy).join(',')) fail(valueWhere, `the value has shape (${actual.join(', ')}), but the buffer declares (${shape.map((d) => d.toy).join(', ')})`);
    const path = join(instance.path, name);
    const tex = this.memberTex(instance, name);
    let node: Node;
    try {
      node = this.g.buffer(path, shape.map((d) => this.dimSpec(d)), values as Nested, { dtype: dtype.dtype, tex, elem: elemOf(tex) });
    } catch (error) {
      return fail(at, (error as Error).message);
    }
    this.define(instance, name, node, at);
    this.nodes.set(path, node);
  }

  private constant(ast: Ast, env: Env): Constant {
    switch (ast.k) {
      case 'num':
        return ast.v;
      case 'name': {
        const value = this.lookup(ast.name, env, ast.pos);
        if (!isNum(value)) return this.fail(env, `'${ast.name}' is ${describe(value)}, not a number`, ast.pos);
        return value.toy;
      }
      case 'neg':
        return mapConstant(this.constant(ast.arg, env), (x) => -x);
      case 'list':
        return ast.items.map((item) => this.constant(item, env));
      case 'bin': {
        const a = this.constant(ast.a, env);
        const b = this.constant(ast.b, env);
        const ops: Record<string, (x: number, y: number) => number> = { '+': (x, y) => x + y, '-': (x, y) => x - y, '*': (x, y) => x * y, '/': (x, y) => x / y, '//': (x, y) => Math.floor(x / y), '**': (x, y) => x ** y };
        const fn = ops[ast.op];
        if (fn === undefined) return this.fail(env, `'${ast.op}' is not available in a buffer value`, ast.pos);
        return zipConstant(a, b, fn);
      }
      case 'call': {
        if (ast.fn.k === 'member' && ['float', 'double'].includes(ast.fn.name) && ast.args.length === 0) return this.constant(ast.fn.obj, env);
        if (ast.fn.k === 'name' && ast.fn.name === 'arange' && ast.kwargs.length === 0) {
          const args = ast.args.map((arg) => this.constant(arg, env));
          if (args.length < 1 || args.length > 3 || args.some((a) => typeof a !== 'number')) return this.fail(env, 'arange takes one to three numbers', ast.pos);
          const [start, stop, step] = args.length === 1 ? [0, args[0] as number, 1] : [args[0] as number, args[1] as number, (args[2] as number | undefined) ?? 1];
          return Array.from({ length: Math.max(0, Math.ceil((stop - start) / step)) }, (_, i) => start + i * step);
        }
        return this.fail(env, 'a buffer value can only use numbers, args, arange and arithmetic', ast.pos);
      }
      default:
        return this.fail(env, 'a buffer value can only use numbers, args, arange and arithmetic', ast.pos);
    }
  }

  private childMember(instance: Instance, name: string, spec: unknown, at: Where): void {
    const env = this.instanceEnv(instance, at);
    const decl: Record<string, unknown> = typeof spec === 'string' ? { use: spec } : record(spec, at, `'${name}'`);
    checkKeys(decl, CHILD_KEYS, at);
    const useWhere = typeof spec === 'string' ? at : child(at, decl, 'use');
    if (typeof decl.use !== 'string') fail(useWhere, 'use must name a module');
    const listed = LIST_USE.exec(decl.use);
    const used = this.expression(listed === null ? decl.use : listed[1], { ...env, where: useWhere });
    if (!isModuleRef(used)) fail(useWhere, `'${decl.use}' is ${describe(used)}, not a module`);
    const given = this.childArgs(decl.args, { ...env, where: child(at, decl, 'args') });
    const path = join(instance.path, name);
    if (listed === null) {
      if (decl.count !== undefined) fail(child(at, decl, 'count'), 'count needs use: list[Module]');
      this.define(instance, name, this.instantiate(used.module, path, name, instance, instance.marks, given, useWhere), at);
      return;
    }
    if (decl.count === undefined) fail(at, `'${name}' is a list, so it needs count`);
    const count = this.number(decl.count, { ...env, where: child(at, decl, 'count') });
    if (!Number.isInteger(count.toy) || count.toy < 1) fail(child(at, decl, 'count'), 'count must be a positive integer');
    const list = new InstanceList(path, name, count);
    const make = () => {
      for (let i = 0; i < count.toy; i++) list.items.push(this.instantiate(used.module, `${path}[${i}]`, name, instance, [...instance.marks, listMark(i)], given, useWhere));
    };
    const real = this.dimSpec(count);
    if (real === count.toy) make();
    else this.g.realCopies(count.toy, real, make);
    this.define(instance, name, list, at);
  }

  private childArgs(spec: unknown, env: Env): Map<string, { value: Value; where: Where }> {
    const out = new Map<string, { value: Value; where: Where }>();
    if (spec === undefined) return out;
    const put = (key: string, text: unknown, where: Where) => {
      if (out.has(key)) fail(where, `arg '${key}' is given twice`);
      out.set(key, { value: this.expression(text, { ...env, where }), where });
    };
    if (Array.isArray(spec)) {
      spec.forEach((item, i) => {
        const at = child(env.where, spec, i);
        if (typeof item === 'string') {
          if (!NAME.test(item)) fail(at, `'${item}' is not a name; write { name: value } to give a value`);
          put(item, item, at);
          return;
        }
        if (!isRecord(item) || Object.keys(item).length !== 1) fail(at, 'an args list item is a name or a mapping with one key');
        const key = Object.keys(item)[0];
        put(key, item[key], child(at, item, key));
      });
      return out;
    }
    const map = record(spec, env.where, 'args');
    for (const [key, text] of Object.entries(map)) put(key, text, child(env.where, map, key));
    return out;
  }

  outerTex(instance: Instance, name: string): string | undefined {
    const chain: Instance[] = [];
    for (let at = instance.parent; at !== undefined; at = at.parent) chain.unshift(at);
    for (const ancestor of chain) {
      const relative = instance.path.slice(ancestor.path === '' ? 0 : ancestor.path.length + 1);
      const key = `${relative}.${name}`;
      const texMap = this.fieldOf(ancestor.def, 'tex');
      const found = texMap[key];
      if (found !== undefined) {
        if (typeof found !== 'string') fail(child(child(ancestor.def.where, ancestor.def.decl, 'tex'), texMap, key), 'tex must be a string');
        return this.marked(this.expandTex(found, this.instanceEnv(ancestor)), ancestor.marks, found);
      }
    }
    return undefined;
  }

  ownTex(instance: Instance, name: string, env: Env): string {
    const texMap = this.fieldOf(instance.def, 'tex');
    const own = texMap[name];
    if (own !== undefined) {
      const at = child(child(instance.def.where, instance.def.decl, 'tex'), texMap, name);
      if (typeof own !== 'string') fail(at, 'tex must be a string');
      return this.marked(this.expandTex(own, { ...env, where: at }), instance.marks, own);
    }
    return this.marked(defaultTex(name), instance.marks, '');
  }

  memberTex(instance: Instance, name: string): string {
    return this.outerTex(instance, name) ?? this.ownTex(instance, name, this.instanceEnv(instance));
  }

  marked(tex: string, marks: readonly string[], source: string): string {
    if (source.includes('\\viz')) return tex;
    return marks.reduce((acc, mark) => addMark(acc, mark), tex);
  }

  abstract expandTex(tex: string, env: Env): string;

  resultAlias(call: Call, name: string, env: Env): Alias {
    const given = call.aliases.get(name);
    const outer = this.outerTex(call.instance, name);
    return {
      path: given?.path ?? join(call.prefix, name),
      tex: outer ?? given?.tex ?? this.ownTex(call.instance, name, env),
    };
  }

  callInstance(instance: Instance, args: Value[], kwargs: Map<string, Value>, env: Env, make: Make, top: boolean, at: number): Value {
    const outputs = Object.keys(this.fieldOf(instance.def, 'outputs'));
    let aliases = new Map<string, Alias>();
    if (top && outputs.length === 1) aliases.set(outputs[0], { path: make.path, tex: make.tex });
    if (top && outputs.length > 1) aliases = new Map(env.call?.resultAliases.get(make.name) ?? []);
    const index = instance.calls.length + 1;
    const prefix = index === 1 ? instance.path : `${instance.path}#${index}`;
    const call = new Call(instance, prefix, env.call, aliases);
    const before = this.g.nodes.length;
    this.bindInputs(call, args, kwargs, env, at);
    instance.calls.push(call);
    env.call?.children.push(call);
    this.calls.push(call);
    this.runForward(call);
    call.created = this.g.nodes.slice(before);
    const result = this.checkOutputs(call);
    for (const node of call.outputs.values()) if (!this.producers.has(node)) this.producers.set(node, call);
    return result;
  }

  bindInputs(call: Call, args: Value[], kwargs: Map<string, Value>, env: Env, at: number): void {
    const def = call.instance.def;
    const inputsWhere = child(def.where, def.decl, 'inputs');
    const declared = this.fieldOf(def, 'inputs');
    const names = Object.keys(declared);
    if (args.length > names.length) this.fail(env, `${def.name} takes ${names.length} inputs (${names.join(', ')}), got ${args.length}`, at);
    const given = new Map<string, Value>();
    args.forEach((value, i) => given.set(names[i], value));
    for (const [key, value] of kwargs) {
      if (!(key in declared)) this.fail(env, `${def.name} has no input '${key}'; its inputs are ${names.join(', ')}`, at);
      if (given.has(key)) this.fail(env, `input '${key}' of ${def.name} is given twice`, at);
      given.set(key, value);
    }
    const callEnv = this.callEnv(call, inputsWhere);
    for (const name of names) {
      const value = given.get(name);
      if (value === undefined) this.fail(env, `input '${name}' of ${def.name} is not given`, at);
      if (!(value instanceof Node)) this.fail(env, `input '${name}' of ${def.name} must be a tensor, got ${describe(value)}`, at);
      const fieldWhere = child(inputsWhere, declared, name);
      this.bindField(call, name, value, declared[name], { ...callEnv, where: fieldWhere }, (message) => this.fail(env, `input '${name}' of ${def.name}: ${message}`, at), true);
      call.values.set(name, value);
      call.inputs.set(name, value);
    }
  }

  bindField(call: Call, name: string, node: Node, spec: unknown, env: Env, mismatch: (message: string) => never, bind: boolean): void {
    const field = record(spec, env.where, `'${name}'`);
    checkKeys(field, FIELD_KEYS, env.where);
    if (field.type === 'list') return;
    if (field.type !== 'tensor') fail(child(env.where, field, 'type'), 'type must be tensor or list');
    const shape = field.shape;
    const shapeWhere = child(env.where, field, 'shape');
    if (!Array.isArray(shape)) fail(shapeWhere, 'shape must be a list');
    const rest = shape.indexOf('...');
    if (rest !== shape.lastIndexOf('...')) fail(shapeWhere, "a shape can hold '...' once");
    if (rest < 0 ? shape.length !== node.shape.length : shape.length - 1 > node.shape.length) mismatch(`expected ${rest < 0 ? '' : 'at least '}${shape.length - (rest < 0 ? 0 : 1)} dimensions (${shape.join(', ')}), got shape (${node.shape.join(', ')})`);
    shape.forEach((entry, k) => {
      if (k === rest) return;
      const d = rest >= 0 && k > rest ? node.shape.length - (shape.length - k) : k;
      const at = child(shapeWhere, shape, k);
      if (bind && typeof entry === 'string' && NAME.test(entry) && !call.instance.args.has(entry) && !call.bound.has(entry) && !call.values.has(entry)) {
        call.bound.set(entry, { toy: node.shape[d], real: node.realShape[d], derive: (_sizes, shapes) => shapes[node.id][d] });
        return;
      }
      const expected = this.number(entry, { ...env, where: at });
      if (expected.toy !== node.shape[d]) mismatch(`dimension ${d} is ${node.shape[d]}, but ${String(entry)} is ${expected.toy}`);
    });
    const dtypeWhere = child(env.where, field, 'dtype');
    const dtype = field.dtype;
    if (typeof dtype !== 'string') fail(dtypeWhere, 'dtype must be a dtype name or an expression');
    if (bind && NAME.test(dtype) && !(DTYPES as readonly string[]).includes(dtype) && !call.instance.args.has(dtype) && !call.bound.has(dtype)) {
      call.bound.set(dtype, { dtype: node.dtype });
      return;
    }
    const expected = this.expression(dtype, { ...env, where: dtypeWhere });
    if (!isDtype(expected)) fail(dtypeWhere, `'${dtype}' is ${describe(expected)}, not a dtype`);
    if (expected.dtype !== node.dtype) mismatch(`expected dtype ${expected.dtype}, got ${node.dtype}`);
  }

  runForward(call: Call): void {
    const def = call.instance.def;
    const forwardWhere = child(def.where, def.decl, 'forward');
    const forward = def.decl.forward ?? [];
    if (!Array.isArray(forward)) fail(forwardWhere, 'forward must be a list of { name: expression }');
    const items = forward.map((item, i) => {
      const at = child(forwardWhere, forward, i);
      if (!isRecord(item) || Object.keys(item).length !== 1) fail(at, 'a forward item must have exactly one name');
      const name = Object.keys(item)[0];
      if (!NAME.test(name)) fail(at, `'${name}' is not a valid name`);
      return { name, text: item[name], where: child(at, item, name) };
    });
    const seen = new Set<string>();
    for (const { name, where } of items) {
      if (seen.has(name)) fail(where, `name '${name}' is already defined in ${def.name}`);
      seen.add(name);
    }
    this.scanAliases(call, items);
    for (const { name, text, where } of items) {
      const env = this.callEnv(call, where);
      if (call.values.has(name) || call.bound.has(name) || call.instance.members.has(name) || call.instance.args.has(name)) fail(where, `name '${name}' is already defined in ${def.name}`);
      const alias = this.resultAlias(call, name, env);
      const make: Make = { name, path: alias.path, tex: alias.tex, anon: [], counter: 0 };
      const value = this.expression(text, env, make);
      if (!(value instanceof Node) && !(value instanceof Result)) fail(where, `'${name}' must be a tensor or a module result, got ${describe(value)}`);
      call.values.set(name, value);
      call.results.push(name);
      if (value instanceof Node) {
        if (make.anon.length > 0) {
          call.anon.set(value, make.anon);
          this.anon.set(value, [...(this.anon.get(value) ?? []), ...make.anon]);
        }
        for (const node of make.anon) this.nodes.set(node.name, node);
        this.nodes.set(join(call.prefix, name), value);
        this.nodes.set(value.name, value);
      }
    }
  }

  private scanAliases(call: Call, items: { name: string; text: unknown; where: Where }[]): void {
    const results = new Set<string>();
    for (const { name, text, where } of items) {
      if (typeof text !== 'string') continue;
      const ast = this.parse(text, this.callEnv(call, where));
      if (ast.k === 'call' && ast.fn.k === 'name') {
        const callee = call.instance.members.get(ast.fn.name);
        if (callee instanceof Instance && Object.keys(this.fieldOf(callee.def, 'outputs')).length > 1) results.add(name);
      }
      if (ast.k === 'member' && ast.obj.k === 'name' && results.has(ast.obj.name)) {
        const map = call.resultAliases.get(ast.obj.name) ?? new Map<string, Alias>();
        map.set(ast.name, this.resultAlias(call, name, this.callEnv(call, where)));
        call.resultAliases.set(ast.obj.name, map);
      }
    }
  }

  checkOutputs(call: Call): Value {
    const def = call.instance.def;
    const outputsWhere = child(def.where, def.decl, 'outputs');
    const declared = this.fieldOf(def, 'outputs');
    const names = Object.keys(declared);
    if (names.length === 0) fail(outputsWhere, `module ${def.name} needs at least one output`);
    for (const name of names) {
      const at = child(outputsWhere, declared, name);
      const value = call.values.get(name);
      if (!(value instanceof Node)) fail(at, value === undefined ? `output '${name}' of ${def.name} is not computed in forward` : `output '${name}' of ${def.name} must be a tensor, got ${describe(value)}`);
      this.bindField(call, name, value, declared[name], { ...this.callEnv(call), where: at }, (message) => fail(at, `output '${name}' of ${def.name}: ${message}`), false);
      call.outputs.set(name, value);
    }
    return names.length === 1 ? (call.outputs.get(names[0]) as Node) : new Result(call, call.outputs);
  }

  callList(list: InstanceList, args: Value[], kwargs: Map<string, Value>, env: Env, make: Make, top: boolean, at: number): Value {
    if (args.length !== 1) this.fail(env, `calling the list '${list.path}' takes one positional input, which passes from layer to layer; give the other inputs by keyword`, at);
    const listCall = new ListCall(list);
    list.calls.push(listCall);
    const run = (): Value => {
      let current = args[0];
      list.items.forEach((item, i) => {
        const last = i === list.items.length - 1;
        current = this.callInstance(item, [current], kwargs, env, make, top && last, at);
        listCall.calls.push(item.calls[item.calls.length - 1]);
        if (!(current instanceof Node)) this.fail(env, `a module list passes one tensor from layer to layer, but ${item.def.name} returns several outputs`, at);
      });
      return current;
    };
    const real = this.dimSpec(list.count);
    return real === list.count.toy ? run() : this.g.realCopies(list.count.toy, real, run);
  }

  constantNode(fn: string, args: Value[], env: Env, make: Make, top: boolean, at: number): Node | undefined {
    if (fn !== 'arange' && fn !== 'causal_mask') return undefined;
    if (!args.every(isNum)) this.fail(env, `${fn} takes numbers`, at);
    const nums = args as Num[];
    let shape: Num[];
    let values: Nested;
    let dtype: Dtype;
    if (fn === 'arange') {
      if (nums.length < 1 || nums.length > 3) this.fail(env, 'arange takes one to three numbers', at);
      const [start, stop, step] = nums.length === 1 ? [num(0), nums[0], num(1)] : [nums[0], nums[1], nums[2] ?? num(1)];
      const length = combine([stop, start, step], (b, a, s) => Math.max(0, Math.ceil((b - a) / s)));
      shape = [length];
      values = Array.from({ length: length.toy }, (_, i) => start.toy + i * step.toy);
      dtype = 'int64';
    } else {
      if (nums.length !== 1) this.fail(env, 'causal_mask takes the sequence length', at);
      const T = nums[0];
      shape = [T, T];
      values = Array.from({ length: T.toy }, (_, r) => Array.from({ length: T.toy }, (_, c) => (c > r ? -Infinity : 0)));
      dtype = this.g.dtype;
    }
    const name = top ? make.path : `${make.path}/${++make.counter}`;
    const tex = top ? make.tex : `\\mathrm{${fn.replace(/_/g, '\\_')}}`;
    let node: Node;
    try {
      node = this.g.buffer(name, shape.map((d) => this.dimSpec(d)), values, { dtype, tex, elem: elemOf(tex) });
    } catch (error) {
      return this.fail(env, (error as Error).message, at);
    }
    if (!top) make.anon.push(node);
    return node;
  }
}

function argSpec(spec: unknown, at: Where): { type: string; fallback: unknown } {
  if (typeof spec === 'string') {
    if (!ARG_TYPES.has(spec)) fail(at, `arg type must be one of ${[...ARG_TYPES].join(', ')}`);
    return { type: spec, fallback: undefined };
  }
  const field = record(spec, at, 'an arg');
  checkKeys(field, new Set(['type', 'default']), at);
  if (typeof field.type !== 'string' || !ARG_TYPES.has(field.type)) fail(child(at, field, 'type'), `arg type must be one of ${[...ARG_TYPES].join(', ')}`);
  return { type: field.type, fallback: field.default };
}

function checkArg(type: string, name: string, value: Value, where: Where): Value {
  const wrong = (what: string): never => fail(where, `arg '${name}' must be ${what}, got ${describe(value)}`);
  switch (type) {
    case 'dim':
      if (!isNum(value) || !Number.isInteger(value.toy) || value.toy <= 0) wrong('a positive integer');
      return value;
    case 'int':
      if (!isNum(value) || !Number.isInteger(value.toy)) wrong('an integer');
      return value;
    case 'float':
      if (!isNum(value)) wrong('a number');
      return value;
    case 'bool':
      if (typeof value !== 'boolean') wrong('True or False');
      return value;
    case 'dtype':
      if (!isDtype(value)) wrong('a dtype');
      return value as DtypeValue;
    default:
      if (!isModuleRef(value)) wrong('a module');
      return value;
  }
}
