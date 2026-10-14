import { DTYPES, Node, type NodeCopies, ProgramBuilder, type RealSize, type Size, isIntegerDtype } from '../../autograd';
import { type Nested, Tensor } from '../../tensor';
import type { Ast, IndexItem } from '../expr';
import type { ModuleDef } from '../load';
import { type Where, child, fail } from '../source';
import { addMark, defaultTex, elemOf } from '../tex';
import { Expressions, describe, isDtype, isModuleRef } from './expressions';
import { type Alias, Call, type DtypeValue, Eager, type Env, Instance, InstanceList, type Iterations, ListCall, type Make, type Member, type Mode, type Num, Result, type Value } from './types';
import { templateExpressions, vizReferences } from './template';
import { checkKeys, combine, fieldSpec, isNum, isRecord, join, num, rangeCount, rangeValues, realOf, record } from './util';

const ARG_TYPES: ReadonlySet<string> = new Set(['dim', 'int', 'float', 'bool', 'str', 'list', 'dtype', 'module']);
const INITS: ReadonlySet<string> = new Set(['ones', 'zeros', 'normal', 'uniform']);
const MODULE_KEYS: ReadonlySet<string> = new Set(['module', 'import', 'args', 'inputs', 'outputs', 'init', 'forward', 'tex', 'formula', 'fold', 'layout', 'steps', 'unshown', 'kernel', 'save', 'backward', 'backward_steps']);
const BRANCH_ITEM_KEYS: ReadonlySet<string> = new Set(['if', 'then', 'else']);
const LOOP_KEY = '\u0000loop:';
const CHILD_KEYS: ReadonlySet<string> = new Set(['use', 'args', 'count', 'index', 'present']);
const BRANCH_KEYS: ReadonlySet<string> = new Set(['if', 'then', 'else']);
const TENSOR_KEYS: ReadonlySet<string> = new Set(['shape', 'dtype', 'init', 'value', 'present']);
const FIELD_KEYS: ReadonlySet<string> = new Set(['type', 'shape', 'dtype', 'optional']);
const LIST_USE = /^list\[(.+)\]$/;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const WRITE = /^([A-Za-z_][A-Za-z0-9_]*)\[(.+)\]$/;
const LOOP_HEAD = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+range\((.+)\)\s*$/;

type ForwardItem =
  | { kind: 'assign'; name: string; target: string | undefined; text: unknown; where: Where }
  | { kind: 'loop'; variable: string; start: string; stop: string; step: string; body: ForwardItem[]; where: Where }
  | { kind: 'branch'; test: unknown; then: ForwardItem[]; other: ForwardItem[]; where: Where };

interface Nest {
  base: NodeCopies;
  drawn: number;
  real: Size;
}

function flatAssignments(items: ForwardItem[]): { name: string; text: unknown; where: Where; target?: string }[] {
  return items.flatMap((item) => (item.kind === 'loop' ? flatAssignments(item.body) : item.kind === 'branch' ? [...flatAssignments(item.then), ...flatAssignments(item.other)] : [item]));
}

function topLevelArgs(text: string): string[] {
  const args: string[] = [];
  let depth = 0;
  let from = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if ('([{'.includes(ch)) depth++;
    else if (')]}'.includes(ch)) depth--;
    else if (ch === ',' && depth === 0) {
      args.push(text.slice(from, i).trim());
      from = i + 1;
    }
  }
  args.push(text.slice(from).trim());
  return args;
}

function innerLoops(items: ForwardItem[]): Extract<ForwardItem, { kind: 'loop' }>[] {
  return items.flatMap((item) => (item.kind === 'loop' ? [item] : item.kind === 'branch' ? innerLoops([...item.then, ...item.other]) : []));
}

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
        if (item.k === 'slice') for (const bound of [item.start, item.stop, item.step]) countNames(bound, counts);
      }
      return;
    case 'call':
      for (const part of [ast.fn, ...ast.args, ...ast.kwargs.map(([, value]) => value)]) countNames(part, counts);
      return;
    case 'list':
      for (const item of ast.items) countNames(item, counts);
      return;
    case 'neg':
    case 'invert':
    case 'not':
      countNames(ast.arg, counts);
      return;
    case 'logic':
    case 'bin':
    case 'cmp':
      countNames(ast.a, counts);
      countNames(ast.b, counts);
      return;
    case 'cond':
      for (const part of [ast.test, ast.then, ast.other]) countNames(part, counts);
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
  private readonly nests = new Map<ForwardItem, Nest>();
  readonly backwardCalls = new Map<Call, Call>();

  inBackward(call: Call): boolean {
    for (let at: Call | undefined = call; at !== undefined; at = at.parent) if (at.parent !== undefined && this.backwardCalls.get(at.parent) === at) return true;
    return false;
  }

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
      const { type, fallback, options } = argSpec(spec, at);
      const provided = given.get(key);
      let value: Value;
      if (provided !== undefined) value = provided.value;
      else if (fallback !== undefined) value = this.expression(fallback, { ...env, where: child(at, spec as object, 'default') });
      else return fail(where, `module ${def.name} needs arg '${key}'`);
      const checked = checkArg(type, key, value, provided?.where ?? at);
      if (options !== undefined && !options.includes(checked as string)) fail(provided?.where ?? at, `arg '${key}' of ${def.name} is '${String(checked)}', which is not one of ${options.join(', ')}`);
      instance.args.set(key, checked);
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
      if (key === 'constants') {
        for (const [constant, text] of Object.entries(record(spec, at, 'constants'))) this.constantMember(instance, constant, text, child(at, spec as object, constant));
      } else if (key === 'weights') {
        for (const [weight, decl] of Object.entries(record(spec, at, 'weights'))) this.tensorMember(instance, weight, decl, child(at, spec as object, weight), 'param');
      } else if (key === 'buffers') {
        for (const [buffer, decl] of Object.entries(record(spec, at, 'buffers'))) this.tensorMember(instance, buffer, decl, child(at, spec as object, buffer), 'buffer');
      } else {
        this.childMember(instance, key, spec, at);
      }
    }
  }

  private constantMember(instance: Instance, name: string, text: unknown, at: Where): void {
    if (!NAME.test(name)) fail(at, `'${name}' is not a valid name`);
    if (instance.members.has(name) || instance.args.has(name)) fail(at, `name '${name}' is already defined in ${instance.def.name}`);
    const value = this.computed(text, { ...this.instanceEnv(instance, at), where: at }, new ProgramBuilder(), false);
    if (!isNum(value) && typeof value !== 'boolean' && typeof value !== 'string' && !(value instanceof Eager)) fail(at, `constant '${name}' must be a number, a bool, a str or a tensor computed at init, got ${describe(value)}`);
    instance.args.set(name, value);
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
    if ('if' in decl) {
      checkKeys(decl, BRANCH_KEYS, at);
      if (decl.then === undefined) fail(at, `'${name}' has 'if' but no 'then'`);
      const test = this.expression(decl.if, { ...env, where: child(at, decl, 'if') });
      if (typeof test !== 'boolean') fail(child(at, decl, 'if'), `if must be True or False, got ${describe(test)}`);
      const branch = test ? 'then' : 'else';
      if (decl[branch] === undefined) this.define(instance, name, null, at);
      else this.childMember(instance, name, decl[branch], child(at, decl, branch));
      return;
    }
    checkKeys(decl, CHILD_KEYS, at);
    if (decl.present !== undefined) {
      const present = this.expression(decl.present, { ...env, where: child(at, decl, 'present') });
      if (typeof present !== 'boolean') fail(child(at, decl, 'present'), `present must be True or False, got ${describe(present)}`);
      if (!present) {
        this.define(instance, name, null, at);
        return;
      }
    }
    const useWhere = typeof spec === 'string' ? at : child(at, decl, 'use');
    if (typeof decl.use !== 'string') fail(useWhere, 'use must name a module');
    const listed = LIST_USE.exec(decl.use);
    const used = this.expression(listed === null ? decl.use : listed[1], { ...env, where: useWhere });
    if (!isModuleRef(used)) fail(useWhere, `'${decl.use}' is ${describe(used)}, not a module`);
    const argsEnv = { ...env, where: child(at, decl, 'args') };
    const path = join(instance.path, name);
    if (listed === null) {
      if (decl.count !== undefined) fail(child(at, decl, 'count'), 'count needs use: list[Module]');
      if (decl.index !== undefined) fail(child(at, decl, 'index'), 'index needs use: list[Module]');
      const given = this.childArgs(decl.args, argsEnv);
      this.define(instance, name, this.instantiate(used.module, path, name, instance, instance.marks, given, useWhere), at);
      return;
    }
    if (decl.count === undefined) fail(at, `'${name}' is a list, so it needs count`);
    const count = this.number(decl.count, { ...env, where: child(at, decl, 'count') });
    if (!Number.isInteger(count.toy) || count.toy < 1) fail(child(at, decl, 'count'), 'count must be a positive integer');
    const indexWhere = child(at, decl, 'index');
    if (decl.index !== undefined && (typeof decl.index !== 'string' || !NAME.test(decl.index))) fail(indexWhere, 'index names the item index, such as index: layer_idx');
    const indexName = decl.index as string | undefined;
    const list = new InstanceList(path, name, count);
    const make = () => {
      for (let i = 0; i < count.toy; i++) {
        const locals = indexName === undefined ? argsEnv.locals : new Map([...argsEnv.locals, [indexName, num(i)]]);
        const given = this.childArgs(decl.args, { ...argsEnv, locals });
        list.items.push(this.instantiate(used.module, `${path}[${i}]`, name, instance, [...instance.marks, listMark(i)], given, useWhere));
      }
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
    if (top && outputs.length > 1) aliases = new Map([...(env.call?.resultAliases.get(make.name) ?? [])].map(([output, alias]) => [output, { ...alias, path: `${alias.path}${make.suffix ?? ''}` }]));
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
    this.markKernel(call, call.created);
    if (instance.def.decl.backward !== undefined) this.runBackward(call);
    return result;
  }

  private markKernel(call: Call, created: readonly Node[]): void {
    const def = call.instance.def;
    const kernel = def.decl.kernel;
    if (kernel === undefined) return;
    if (typeof kernel !== 'boolean') fail(child(def.where, def.decl, 'kernel'), 'kernel must be true or false');
    if (!kernel) return;
    const outputs = new Set(call.outputs.values());
    this.g.markOnChip(created.filter((n) => !outputs.has(n)));
  }

  private runBackward(call: Call): void {
    const def = call.instance.def;
    const where = child(def.where, def.decl, 'backward');
    const saveWhere = child(def.where, def.decl, 'save');
    const save = def.decl.save ?? [];
    if (!Array.isArray(save) || !save.every((name) => typeof name === 'string' && NAME.test(name))) fail(saveWhere, 'save lists forward names, like ctx.save_for_backward(q, k, v)');
    const backward = new Call(call.instance, join(call.prefix, 'backward'), call, new Map());
    this.backwardCalls.set(call, backward);
    for (const [name, value] of call.bound) backward.bound.set(name, value);
    const start = this.g.nodes.length;
    const gradOutputs: { leaf: Node; output: Node }[] = [];
    this.g.backwardPhase(() => {
      for (const [name, output] of call.outputs) {
        const gradName = `grad_${name}`;
        const own = this.fieldOf(def, 'tex')[gradName];
        const tex = typeof own === 'string' ? this.marked(this.expandTex(own, this.callEnv(call)), call.instance.marks, own) : `\\mathrm{d}${output.tex}`;
        const leaf = this.g.grad_of(output, { name: join(backward.prefix, gradName), tex, elem: elemOf(tex) });
        gradOutputs.push({ leaf, output });
        backward.values.set(gradName, leaf);
        this.nodes.set(leaf.name, leaf);
      }
      (save as string[]).forEach((name, i) => {
        const at = child(saveWhere, save, i);
        const value = call.values.get(name);
        if (!(value instanceof Node)) fail(at, value === undefined ? `'${name}' is not defined in the forward of ${def.name}` : `'${name}' is ${describe(value)}; save takes tensors`);
        if (backward.values.has(name)) fail(at, `'${name}' is saved twice`);
        const node = this.g.saved(value, { name: join(backward.prefix, name), tex: value.tex, elem: value.elem });
        backward.values.set(name, node);
        this.nodes.set(node.name, node);
      });
      backward.lines.push({ nodes: this.g.nodes.slice(start), calls: [] });
      const items = this.forwardItems(def.decl.backward, where);
      this.scanAliases(backward, flatAssignments(items));
      this.runItems(backward, items, []);
      backward.finished = true;
    });
    const gradInputs: { input: Node; grad: Node }[] = [];
    for (const [name, input] of call.inputs) {
      const gradName = `grad_${name}`;
      const grad = backward.values.get(gradName);
      if (grad === undefined) {
        if (isIntegerDtype(input.dtype)) continue;
        fail(where, `the backward of ${def.name} must set ${gradName}, the gradient of input '${name}', or set it to None`);
      }
      if (grad === null) continue;
      if (!(grad instanceof Node)) fail(where, `${gradName} must be a tensor or None, got ${describe(grad)}`);
      gradInputs.push({ input, grad });
    }
    const nodes = this.g.nodes.slice(start);
    backward.created = nodes;
    call.lines.push({ nodes: [], calls: [backward] });
    try {
      this.g.defineFunction({ inputs: [...call.inputs.values()], outputs: [...call.outputs.values()], inside: call.created, gradOutputs, gradInputs, backward: nodes });
    } catch (error) {
      fail(where, (error as Error).message);
    }
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
      const optional = isRecord(declared[name]) && (declared[name] as Record<string, unknown>).optional === true;
      if (optional && (value === undefined || value === null)) {
        call.values.set(name, null);
        continue;
      }
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
    if (dtype === '_') return;
    if (bind && NAME.test(dtype) && !(DTYPES as readonly string[]).includes(dtype) && !call.instance.args.has(dtype) && !call.bound.has(dtype)) {
      if (!Array.isArray(spec)) this.checkUsed(call, dtype, dtypeWhere);
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
      if (at.key === 'constants') walk(entry, at, expression);
      else if (at.key === 'weights' || at.key === 'buffers') each(entry, at, (tensor, w) => fields(tensor, w, ['shape', 'dtype', 'value', 'present']));
      else {
        const member = (spec: unknown, where: Where): void => {
          fields(spec, where, ['args', 'count', 'present', 'if']);
          if (isRecord(spec)) for (const key of ['then', 'else']) if (spec[key] !== undefined) member(spec[key], child(where, spec, key));
        };
        member(entry, at);
      }
    });
    const loopOrExpression = (text: string, where: Where) => {
      const head = LOOP_HEAD.exec(text);
      if (head === null) expression(text, where);
      else for (const bound of topLevelArgs(head[2])) expression(bound, where);
    };
    for (const key of ['forward', 'backward']) walk(decl[key], section(key), loopOrExpression);
    walk(decl.save, section('save'), expression);
    walk(decl.unshown, section('unshown'), expression);
    walk(decl.layout, section('layout'), expression);
    walk(decl.tex, section('tex'), latex);
    walk(decl.formula, section('formula'), latex);
    options(decl.fold, section('fold'));
    for (const key of ['steps', 'backward_steps']) {
      const list = decl[key];
      if (Array.isArray(list)) list.forEach((step, i) => options(step, child(section(key), list, i)));
    }
    return counts;
  }

  runForward(call: Call): void {
    const def = call.instance.def;
    const forwardWhere = child(def.where, def.decl, 'forward');
    const items = this.forwardItems(def.decl.forward ?? [], forwardWhere);
    const seen = new Set<string>();
    for (const item of items) {
      if (item.kind !== 'assign') continue;
      if (item.target !== undefined) fail(item.where, `'${item.name}[...]' writes into a tensor, which forward does only inside a loop; outside a loop, give the result a new name`);
      if (seen.has(item.name)) fail(item.where, `name '${item.name}' is already defined in ${def.name}`);
      seen.add(item.name);
    }
    this.scanAliases(call, flatAssignments(items));
    this.runItems(call, items, []);
    call.finished = true;
  }

  private loopStep(item: Extract<ForwardItem, { kind: 'loop' }>, env: Env): Num {
    const step = this.number(item.step, env);
    if (!Number.isInteger(step.toy) || step.toy === 0 || Math.sign(step.real) !== Math.sign(step.toy)) fail(item.where, `the step of range(${item.start}, ${item.stop}, ${item.step}) must be a nonzero integer, got ${step.toy}`);
    return step;
  }

  private nest(call: Call, outer: Extract<ForwardItem, { kind: 'loop' }>, inner: Extract<ForwardItem, { kind: 'loop' }>, start: Num, stop: Num, step: Num, base: NodeCopies, env: Env): void {
    const names = new Map<string, number>();
    for (const bound of [inner.start, inner.stop]) countNames(this.parse(bound, env), names);
    if (!names.has(outer.variable)) return;
    const key = `${LOOP_KEY}${outer.variable}`;
    const saved = call.values.get(outer.variable);
    const spanAt = (value: Num): Num => {
      call.values.set(outer.variable, value);
      const from = this.number(inner.start, this.callEnv(call, inner.where));
      const to = this.number(inner.stop, this.callEnv(call, inner.where));
      return combine([from, to, innerStep], rangeCount);
    };
    const innerStep = this.loopStep(inner, env);
    let drawn = 0;
    for (const k of rangeValues(start.toy, stop.toy, step.toy)) drawn += spanAt(num(k)).toy;
    const symbolic = spanAt({ toy: 0, real: 0, derive: (sizes) => sizes[key] });
    if (saved === undefined) call.values.delete(outer.variable);
    else call.values.set(outer.variable, saved);
    if (drawn === 0) return;
    const span = realOf(symbolic);
    const first = realOf(start);
    const last = realOf(stop);
    const every = realOf(step);
    const derive: RealSize = (sizes, shapes) => {
      let total = 0;
      for (const k of rangeValues(first(sizes, shapes), last(sizes, shapes), every(sizes, shapes))) total += span({ ...sizes, [key]: k }, shapes);
      return total;
    };
    const real: Size = this.g.real === undefined ? drawn : { toy: drawn, real: derive(this.g.realSizes(), this.g.realShapes()), derive };
    this.nests.set(inner, { base, drawn, real });
  }

  private forwardItems(forward: unknown, where: Where): ForwardItem[] {
    if (!Array.isArray(forward)) fail(where, 'forward must be a list of { name: expression }');
    return forward.map((item, i) => {
      const at = child(where, forward, i);
      if (isRecord(item) && 'for' in item) {
        checkKeys(item, new Set(['for', 'do']), at);
        const match = typeof item.for === 'string' ? LOOP_HEAD.exec(item.for) : null;
        if (match === null) fail(child(at, item, 'for'), "for must look like 'i in range(n)'");
        const bounds = topLevelArgs(match[2]);
        if (bounds.length > 3) fail(child(at, item, 'for'), `range takes at most three arguments, got ${bounds.length}`);
        const [start, stop, step] = bounds.length === 1 ? ['0', bounds[0], '1'] : [bounds[0], bounds[1], bounds[2] ?? '1'];
        return { kind: 'loop', variable: match[1], start, stop, step, body: this.forwardItems(item.do, child(at, item, 'do')), where: child(at, item, 'for') };
      }
      if (isRecord(item) && 'if' in item) {
        checkKeys(item, BRANCH_ITEM_KEYS, at);
        if (item.then === undefined) fail(at, "an if block needs 'then'");
        return { kind: 'branch', test: item.if, then: this.forwardItems(item.then, child(at, item, 'then')), other: item.else === undefined ? [] : this.forwardItems(item.else, child(at, item, 'else')), where: child(at, item, 'if') };
      }
      if (!isRecord(item) || Object.keys(item).length !== 1) fail(at, 'a forward item must have exactly one name, be a loop { for: i in range(n), do: [...] }, or be a block { if: condition, then: [...], else: [...] }');
      const key = Object.keys(item)[0];
      const written = WRITE.exec(key);
      if (!NAME.test(key) && written === null) fail(at, `'${key}' is not a valid name`);
      return { kind: 'assign', name: written === null ? key : written[1], target: written === null ? undefined : written[2], text: item[key], where: child(at, item, key) };
    });
  }

  private iterationsOf(call: Call, name: string, loops: { variable: string }[], before: Value | undefined): Iterations {
    const iterations = call.iterations.get(name) ?? { variables: [], values: new Map<string, Value>(), before };
    if (loops.length > iterations.variables.length) iterations.variables = loops.map((loop) => loop.variable);
    call.iterations.set(name, iterations);
    return iterations;
  }

  private skipBranch(call: Call, items: ForwardItem[], loops: { variable: string; index: number }[]): void {
    for (const item of items) {
      if (item.kind === 'loop' || (item.kind === 'assign' && item.target !== undefined)) continue;
      if (item.kind === 'branch') {
        this.skipBranch(call, [...item.then, ...item.other], loops);
        continue;
      }
      if (loops.length === 0) {
        if (!call.values.has(item.name)) call.values.set(item.name, null);
        continue;
      }
      const iterations = this.iterationsOf(call, item.name, loops, call.values.get(item.name));
      const key = loops.map((loop) => loop.index).join(',');
      if (!iterations.values.has(key)) iterations.values.set(key, null);
    }
  }

  private runItems(call: Call, items: ForwardItem[], loops: { variable: string; index: number }[]): void {
    const def = call.instance.def;
    for (const item of items) {
      const env = this.callEnv(call, item.where);
      if (item.kind === 'branch') {
        const test = this.expression(item.test, env);
        if (typeof test !== 'boolean') fail(item.where, `if must be True or False, got ${describe(test)}`);
        this.runItems(call, test ? item.then : item.other, loops);
        this.skipBranch(call, test ? item.other : item.then, loops);
        continue;
      }
      if (item.kind === 'loop') {
        if (call.instance.members.has(item.variable) || call.instance.args.has(item.variable) || call.bound.has(item.variable)) fail(item.where, `name '${item.variable}' is already defined in ${def.name}`);
        const start = this.number(item.start, env);
        const stop = this.number(item.stop, env);
        const step = this.loopStep(item, env);
        if (!Number.isInteger(start.toy) || !Number.isInteger(stop.toy)) fail(item.where, `range(${item.start}, ${item.stop}) needs integers, got ${start.toy} and ${stop.toy}`);
        const span = combine([start, stop, step], rangeCount);
        const count = span.toy;
        if (count === 0) continue;
        const first: Num = start.derive === undefined ? { toy: start.toy, real: start.real } : { toy: start.toy, real: start.real, derive: start.derive };
        const run = () => {
          for (const k of rangeValues(start.toy, stop.toy, step.toy)) {
            call.values.set(item.variable, { ...first, toy: k });
            this.runItems(call, item.body, [...loops, { variable: item.variable, index: k }]);
          }
        };
        const base = this.g.copiesNow();
        for (const inner of innerLoops(item.body)) this.nest(call, item, inner, start, stop, step, base, env);
        const nest = this.nests.get(item);
        const real = this.dimSpec(span);
        if (nest !== undefined) this.g.realCopies(nest.drawn, nest.real, run, nest.base);
        else if (real === count) run();
        else this.g.realCopies(count, real, run);
        continue;
      }
      const { name, where } = item;
      const inLoop = loops.length > 0;
      const defined = call.values.has(name);
      if ((!inLoop && defined) || call.bound.has(name) || call.instance.members.has(name) || call.instance.args.has(name)) fail(where, `name '${name}' is already defined in ${def.name}`);
      if (item.target !== undefined && !defined) fail(where, `'${name}' must be defined before '${name}[${item.target}]' writes into it`);
      if (inLoop && defined && loops.some((loop) => loop.variable === name)) fail(where, `'${name}' is the loop variable`);
      const suffix = loops.map((loop) => `[${loop.index}]`).join('');
      const alias = this.resultAlias(call, name, env);
      const tex = loops.reduce((acc, loop) => addMark(acc, `_{${loop.index}}`), alias.tex);
      const make: Make = { name, path: `${alias.path}${suffix}`, suffix, tex, anon: [] };
      const before = this.g.nodes.length;
      const calls = call.children.length;
      const value = item.target === undefined ? this.expression(item.text, env, make) : this.write(item.target, item.text, name, env, make);
      const inner = call.children.slice(calls);
      const nested = new Set(inner.flatMap((c) => c.created));
      call.lines.push({ nodes: this.g.nodes.slice(before).filter((n) => !nested.has(n)), calls: inner });
      if (value === null && item.target === undefined && !inLoop) {
        call.values.set(name, null);
        continue;
      }
      const numeric = (isNum(value) || typeof value === 'boolean') && item.target === undefined;
      if (numeric && this.g.nodes.length !== before) fail(where, `'${name}' is ${describe(value)}, but computing it created tensors`);
      if (!numeric && !(value instanceof Node) && !(value instanceof Result)) fail(where, `'${name}' must be a tensor, a number, a bool, a module result or None, got ${describe(value)}`);
      const previous = call.values.get(name);
      call.values.set(name, value);
      if (inLoop) {
        this.iterationsOf(call, name, loops, previous).values.set(loops.map((loop) => loop.index).join(','), value);
      }
      if (numeric) continue;
      if (!call.results.includes(name)) call.results.push(name);
      call.resultValues.push({ name, value, where });
      if (value instanceof Node) {
        if (make.anon.length > 0) {
          call.anon.set(value, make.anon);
          this.anon.set(value, [...(this.anon.get(value) ?? []), ...make.anon]);
        }
        for (const node of make.anon) this.nodes.set(node.name, node);
        this.nodes.set(join(call.prefix, `${name}${suffix}`), value);
        this.nodes.set(value.name, value);
      }
    }
  }

  private write(target: string, text: unknown, name: string, env: Env, make: Make): Value {
    const index = this.parse(`${name}[${target}]`, env);
    if (index.k !== 'index') return this.fail(env, `'${name}[${target}]' is not an index`);
    const pos = index.pos;
    const ellipsis = index.items.findIndex((item) => item.k === 'ellipsis');
    if (index.items.filter((item) => item.k === 'ellipsis').length > 1) this.fail(env, `'${name}[${target}]' has more than one '...'`);
    const full = (item: IndexItem) => item.k === 'slice' && item.start === undefined && item.stop === undefined;
    const picks = index.items
      .map((item, d) => ({ item, dim: ellipsis >= 0 && d > ellipsis ? d - index.items.length : d }))
      .filter(({ item }) => item.k !== 'ellipsis' && !full(item));
    const mode: Mode = { k: 'graph', make };
    if (picks.length === 0) {
      if (index.items.length !== 1 || ellipsis !== 0) this.fail(env, `'${name}[${target}]' writes the whole tensor; write '${name}[...]' to update it in place, or give the result a new name`);
      const whole = this.evaluate({ k: 'name', name, pos }, env, mode, false);
      return this.apply('slice_scatter', [whole, this.evaluate(this.parse(text, env), env, mode, false), num(0), null, null], new Map(), env, mode, true, pos);
    }
    const scalar = (ast: Ast | undefined): Value => (ast === undefined ? null : this.evaluate(ast, env, mode, false));
    const part = (base: Value, item: IndexItem, dim: number): Value => {
      const items: IndexItem[] = dim >= 0 ? [...Array.from({ length: dim }, (): IndexItem => ({ k: 'slice' })), item] : [{ k: 'ellipsis' }, item, ...Array.from({ length: -dim - 1 }, (): IndexItem => ({ k: 'slice' }))];
      const local = { ...env, locals: new Map([...env.locals, [' base', base]]) };
      return this.evaluate({ k: 'index', obj: { k: 'name', name: ' base', pos }, items, pos }, local, mode, false);
    };
    const into = (base: () => Value, rest: { item: IndexItem; dim: number }[], top: boolean): Value => {
      if (rest.length === 0) return this.evaluate(this.parse(text, env), env, mode, false);
      const [{ item, dim }, ...others] = rest;
      const whole = base();
      if (item.k === 'at') {
        if (item.value.k === 'name' && item.value.name === 'None') this.fail(env, `'${name}[${target}]' cannot write through None`);
        const shifted = others.map((o) => ({ ...o, dim: dim >= 0 && o.dim > dim ? o.dim - 1 : o.dim }));
        const src = into(() => part(whole, item, dim), shifted, false);
        return this.apply('select_scatter', [whole, src, num(dim), scalar(item.value)], new Map(), env, mode, top, pos);
      }
      if (item.k !== 'slice') return this.fail(env, `'${name}[${target}]' has an entry that cannot be written`);
      const src = into(() => part(whole, item, dim), others, false);
      return this.apply('slice_scatter', [whole, src, num(dim), scalar(item.start), scalar(item.stop)], new Map(), env, mode, top, pos);
    };
    return into(() => this.evaluate({ k: 'name', name, pos }, env, mode, false), picks, true);
  }

  private scanAliases(call: Call, items: { name: string; text: unknown; where: Where; target?: string }[]): void {
    const results = new Set<string>();
    for (const { name, text, where, target } of items) {
      if (typeof text !== 'string' || target !== undefined) continue;
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
    const none = new Set<string>();
    for (const name of names) {
      const at = child(outputsWhere, declared, name);
      const value = call.values.get(name);
      if (value === null && isRecord(declared[name]) && (declared[name] as Record<string, unknown>).optional === true) {
        none.add(name);
        continue;
      }
      if (!(value instanceof Node)) fail(at, value === undefined ? `output '${name}' of ${def.name} is not computed in forward` : `output '${name}' of ${def.name} must be a tensor, got ${describe(value)}`);
      this.bindField(call, name, value, declared[name], { ...this.callEnv(call), where: at }, (message) => fail(at, `output '${name}' of ${def.name}: ${message}`), false);
      call.outputs.set(name, value);
    }
    if (names.length === 1) return call.outputs.get(names[0]) ?? null;
    return new Result(call, call.outputs, none);
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

export function argSpec(spec: unknown, at: Where): { type: string; fallback: unknown; options?: string[] } {
  if (typeof spec === 'string') {
    if (!ARG_TYPES.has(spec)) fail(at, `arg type must be one of ${[...ARG_TYPES].join(', ')}`);
    return { type: spec, fallback: undefined };
  }
  const field = record(spec, at, 'an arg');
  checkKeys(field, new Set(['type', 'default', 'options']), at);
  if (typeof field.type !== 'string' || !ARG_TYPES.has(field.type)) fail(child(at, field, 'type'), `arg type must be one of ${[...ARG_TYPES].join(', ')}`);
  if (field.options !== undefined) {
    const options = field.options;
    if (field.type !== 'str' || !Array.isArray(options) || options.length < 2 || !options.every((o) => typeof o === 'string')) fail(child(at, field, 'options'), 'options lists the strings a str arg can take, at least two');
    return { type: field.type, fallback: field.default, options: options as string[] };
  }
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
    case 'str':
      if (typeof value !== 'string') wrong('a string');
      return value;
    case 'list':
      if (!Array.isArray(value)) wrong('a list');
      return value;
    case 'dtype':
      if (!isDtype(value)) wrong('a dtype');
      return value as DtypeValue;
    default:
      if (!isModuleRef(value)) wrong('a module');
      return value;
  }
}
