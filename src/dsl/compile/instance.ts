import { DTYPES, Node, ProgramBuilder } from '../../autograd';
import { type Nested, Tensor } from '../../tensor';
import type { Ast } from '../expr';
import type { ModuleDef } from '../load';
import { type Where, child, fail } from '../source';
import { addMark, defaultTex, elemOf } from '../tex';
import { Expressions, describe, isDtype, isModuleRef } from './expressions';
import { type Alias, Call, type DtypeValue, Eager, type Env, Instance, InstanceList, ListCall, type Make, type Member, type Num, Result, type Value } from './types';
import { templateExpressions, vizReferences } from './template';
import { checkKeys, fieldSpec, isNum, isRecord, join, record } from './util';

const ARG_TYPES: ReadonlySet<string> = new Set(['dim', 'int', 'float', 'bool', 'dtype', 'module']);
const INITS: ReadonlySet<string> = new Set(['ones', 'zeros', 'normal', 'uniform']);
const MODULE_KEYS: ReadonlySet<string> = new Set(['module', 'import', 'args', 'inputs', 'outputs', 'init', 'forward', 'tex', 'formula', 'fold', 'layout', 'steps', 'unshown']);
const CHILD_KEYS: ReadonlySet<string> = new Set(['use', 'args', 'count']);
const TENSOR_KEYS: ReadonlySet<string> = new Set(['shape', 'dtype', 'init', 'value', 'present']);
const FIELD_KEYS: ReadonlySet<string> = new Set(['type', 'shape', 'dtype']);
const LIST_USE = /^list\[(.+)\]$/;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

type Reader = (text: string, where: Where) => void;

function countNames(ast: Ast | undefined, counts: Map<string, number>): void {
  if (ast === undefined) return;
  switch (ast.k) {
    case 'name':
      counts.set(ast.name, (counts.get(ast.name) ?? 0) + 1);
      return;
    case 'member':
      countNames(ast.obj, counts);
      return;
    case 'index':
      countNames(ast.obj, counts);
      for (const item of ast.items) {
        if (item.k === 'at') countNames(item.value, counts);
        if (item.k === 'slice') for (const bound of [item.start, item.stop]) countNames(bound, counts);
      }
      return;
    case 'call':
      for (const part of [ast.fn, ...ast.args, ...ast.kwargs.map(([, value]) => value)]) countNames(part, counts);
      return;
    case 'list':
      for (const item of ast.items) countNames(item, counts);
      return;
    case 'neg':
      countNames(ast.arg, counts);
      return;
    case 'bin':
      countNames(ast.a, counts);
      countNames(ast.b, counts);
      return;
    default:
      return;
  }
}

function walk(value: unknown, where: Where, read: Reader): void {
  if (typeof value === 'string') read(value, where);
  else if (Array.isArray(value)) value.forEach((item, i) => walk(item, child(where, value, i), read));
  else if (isRecord(value)) for (const [key, item] of Object.entries(value)) walk(item, child(where, value, key), read);
}

export function listMark(i: number): string {
  return `^{[${i}]}`;
}

export abstract class Modules extends Expressions {
  readonly calls: Call[] = [];
  readonly producers = new Map<Node, Call>();
  readonly anon = new Map<Node, Node[]>();
  readonly instances: Instance[] = [];
  private readonly structuralNames = new Map<ModuleDef, Map<string, number>>();

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
        for (const [weight, decl] of Object.entries(record(spec, at, 'weights'))) this.tensorMember(instance, weight, decl, child(at, spec as object, weight), 'param');
      } else if (key === 'buffers') {
        for (const [buffer, decl] of Object.entries(record(spec, at, 'buffers'))) this.tensorMember(instance, buffer, decl, child(at, spec as object, buffer), 'buffer');
      } else {
        this.childMember(instance, key, spec, at);
      }
    }
  }

  private define(instance: Instance, name: string, value: Member, where: Where): void {
    if (instance.members.has(name)) fail(where, `name '${name}' is already defined in ${instance.def.name}`);
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

  private tensorMember(instance: Instance, name: string, decl: unknown, at: Where, kind: 'param' | 'buffer'): void {
    const spec = record(decl, at, `'${name}'`);
    checkKeys(spec, TENSOR_KEYS, at);
    if (spec.present !== undefined) {
      const present = this.expression(spec.present, this.instanceEnv(instance, child(at, spec, 'present')));
      if (typeof present !== 'boolean') fail(child(at, spec, 'present'), `present must be True or False, got ${describe(present)}`);
      if (!present) {
        this.define(instance, name, null, at);
        return;
      }
    }
    if ((spec.init === undefined) === (spec.value === undefined)) fail(at, `'${name}' needs exactly one of init or value`);
    const env = this.instanceEnv(instance, child(at, spec, 'shape'));
    const shape = this.shape(spec.shape, env);
    let dtype = this.g.dtype;
    if (spec.dtype !== undefined) {
      const declared = this.expression(spec.dtype, { ...env, where: child(at, spec, 'dtype') });
      if (!isDtype(declared)) fail(child(at, spec, 'dtype'), `dtype must be one of ${DTYPES.join(', ')}`);
      dtype = declared.dtype;
    }
    let values: Nested;
    if (spec.init !== undefined) {
      if (typeof spec.init !== 'string' || !INITS.has(spec.init)) fail(child(at, spec, 'init'), `init must be one of ${[...INITS].join(', ')}`);
      values = this.initValues(spec.init, shape.map((d) => d.toy));
    } else {
      const valueWhere = child(at, spec, 'value');
      const computed = this.computed(spec.value, { ...env, where: valueWhere }, new ProgramBuilder(), false);
      if (!(computed instanceof Eager) && !isNum(computed)) fail(valueWhere, `the value is ${describe(computed)}, not a tensor`);
      const tensor = computed instanceof Eager ? computed.value : new Tensor([], [computed.toy]);
      if (computed instanceof Eager && computed.dtype !== dtype) fail(valueWhere, `the value has dtype ${computed.dtype}, but '${name}' declares ${dtype}`);
      if (tensor.shape.join(',') !== shape.map((d) => d.toy).join(',')) fail(valueWhere, `the value has shape (${tensor.shape.join(', ')}), but '${name}' declares (${shape.map((d) => d.toy).join(', ')})`);
      values = tensor.toNested();
    }
    const path = join(instance.path, name);
    const tex = this.memberTex(instance, name);
    let node: Node;
    try {
      const make = kind === 'param' ? this.g.param.bind(this.g) : this.g.buffer.bind(this.g);
      node = make(path, shape.map((d) => this.dimSpec(d)), values, { dtype, tex, elem: elemOf(tex) });
    } catch (error) {
      return fail(at, (error as Error).message);
    }
    this.define(instance, name, node, at);
    this.nodes.set(path, node);
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
    const field = record(fieldSpec(spec), env.where, `'${name}'`);
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
      if (k === rest || entry === '_') return;
      const d = rest >= 0 && k > rest ? node.shape.length - (shape.length - k) : k;
      const at = child(shapeWhere, shape, k);
      if (bind && typeof entry === 'string' && NAME.test(entry) && !call.instance.args.has(entry) && !call.bound.has(entry) && !call.values.has(entry)) {
        this.checkUsed(call, entry, at);
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
      this.checkUsed(call, dtype, dtypeWhere);
      call.bound.set(dtype, { dtype: node.dtype });
      return;
    }
    const expected = this.expression(dtype, { ...env, where: dtypeWhere });
    if (!isDtype(expected)) fail(dtypeWhere, `'${dtype}' is ${describe(expected)}, not a dtype`);
    if (expected.dtype !== node.dtype) mismatch(`expected dtype ${expected.dtype}, got ${node.dtype}`);
  }

  private checkUsed(call: Call, name: string, where: Where): void {
    if ((this.namesOf(call).get(name) ?? 0) < 2) {
      fail(where, `'${name}' takes its value from this input but is used nowhere else in ${call.instance.def.name}; if it should be an arg, check the spelling, and write _ for a dimension that needs no name`);
    }
  }

  private namesOf(call: Call): Map<string, number> {
    const def = call.instance.def;
    const cached = this.structuralNames.get(def);
    if (cached !== undefined) return cached;
    const counts = new Map<string, number>();
    this.structuralNames.set(def, counts);
    const decl = def.decl;
    const section = (key: string) => child(def.where, decl, key);
    const expression: Reader = (text, where) => {
      if (text !== '...') countNames(this.parse(text, this.callEnv(call, where)), counts);
    };
    const latex: Reader = (text, where) => vizReferences(text).forEach((ref) => expression(ref, where));
    const template: Reader = (text, where) => templateExpressions(text).forEach((expr) => expression(expr, where));
    const fields = (value: unknown, where: Where, keys: readonly string[]) => {
      if (isRecord(value)) for (const key of keys) walk(value[key], child(where, value, key), expression);
    };
    const each = (value: unknown, where: Where, visit: (item: unknown, at: Where) => void) => {
      if (isRecord(value)) for (const [key, item] of Object.entries(value)) visit(item, child(where, value, key));
    };
    const options = (value: unknown, where: Where): void => {
      each(value, where, (item, at) => {
        const key = at.key as string;
        if (key === 'steps' && Array.isArray(item)) item.forEach((step, i) => options(step, child(at, item, i)));
        else if (key === 'edit') each(item, at, options);
        else if (key === 'rules' && Array.isArray(item)) item.forEach((rule, i) => options(rule, child(at, item, i)));
        else if (key === 'defaults' || key === 'fold' || key === 'backward') options(item, at);
        else if (key === 'flat' || key === 'brace') return;
        else if (key === 'notes' && Array.isArray(item)) item.forEach((note, i) => {
          fields(note, child(at, item, i), ['at']);
          if (isRecord(note)) walk(note.text, child(child(at, item, i), note, 'text'), template);
        });
        else if (key === 'show' || key === 'expand') walk(item, at, expression);
        else if (key === 'formula') walk(item, at, latex);
        else if (key !== 'id' && key !== 'loop' && key !== 'side') walk(item, at, template);
      });
    };
    each(decl.args, section('args'), (arg, at) => fields(arg, at, ['default']));
    for (const key of ['inputs', 'outputs']) each(decl[key], section(key), (field, at) => fields(fieldSpec(field), at, ['shape', 'dtype']));
    each(decl.init, section('init'), (entry, at) => {
      if (at.key === 'weights' || at.key === 'buffers') each(entry, at, (tensor, w) => fields(tensor, w, ['shape', 'dtype', 'value', 'present']));
      else fields(entry, at, ['args', 'count']);
    });
    walk(decl.forward, section('forward'), expression);
    walk(decl.unshown, section('unshown'), expression);
    walk(decl.layout, section('layout'), expression);
    walk(decl.tex, section('tex'), latex);
    walk(decl.formula, section('formula'), latex);
    options(decl.fold, section('fold'));
    if (Array.isArray(decl.steps)) decl.steps.forEach((step, i) => options(step, child(section('steps'), decl.steps as unknown[], i)));
    return counts;
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
      const make: Make = { name, path: alias.path, tex: alias.tex, anon: [] };
      const before = this.g.nodes.length;
      const calls = call.children.length;
      const value = this.expression(text, env, make);
      const inner = call.children.slice(calls);
      const nested = new Set(inner.flatMap((c) => c.created));
      call.lines.push({ nodes: this.g.nodes.slice(before).filter((n) => !nested.has(n)), calls: inner });
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
