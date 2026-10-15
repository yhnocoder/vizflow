import { DTYPES, Node, type NodeCopies, ProgramBuilder, type RealSize, type Size, type TileRegion, isIntegerDtype } from '../../autograd';
import { type Nested, Tensor } from '../../tensor';
import type { Ast, IndexItem } from '../expr';
import type { ModuleDef } from '../load';
import { type Where, child, fail } from '../source';
import { addMark, defaultTex, elemOf } from '../tex';
import { Expressions, describe, isDtype, isModuleRef } from './expressions';
import { type Alias, type Arrangement, Call, type DtypeValue, Eager, type Env, type GridDim, GridRun, Instance, InstanceList, type Iterations, ListCall, type Make, type Member, type Mode, type Num, Result, type Value } from './types';
import { templateExpressions, vizReferences } from './template';
import { checkKeys, combine, fieldSpec, isNum, isRecord, join, num, rangeCount, rangeValues, realOf, record } from './util';

const ARG_TYPES: ReadonlySet<string> = new Set(['dim', 'int', 'float', 'bool', 'str', 'list', 'dtype', 'module']);
const INITS: ReadonlySet<string> = new Set(['ones', 'zeros', 'normal', 'uniform']);
const MODULE_KEYS: ReadonlySet<string> = new Set(['module', 'import', 'args', 'inputs', 'outputs', 'init', 'forward', 'tex', 'formula', 'fold', 'layout', 'steps', 'unshown', 'kernel', 'grid', 'shared', 'save', 'backward', 'backward_steps']);
const ARRANGEMENTS: ReadonlySet<string> = new Set(['expand', 'stack']);
const BRANCH_ITEM_KEYS: ReadonlySet<string> = new Set(['if', 'then', 'else']);
const LOOP_KEY = '\u0000loop:';
const CHILD_KEYS: ReadonlySet<string> = new Set(['use', 'args', 'count', 'index', 'present']);
const BRANCH_KEYS: ReadonlySet<string> = new Set(['if', 'then', 'else']);
const TENSOR_KEYS: ReadonlySet<string> = new Set(['shape', 'dtype', 'init', 'value', 'present']);
const FIELD_KEYS: ReadonlySet<string> = new Set(['type', 'shape', 'dtype', 'optional']);
const LIST_USE = /^list\[(.+)\]$/;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const WRITE = /^([A-Za-z_][A-Za-z0-9_]*)\[(.+)\](\s*\+=)?$/;
const LOOP_HEAD = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+range\((.+)\)\s*$/;

type ForwardItem =
  | { kind: 'assign'; name: string; target: string | undefined; accumulate: boolean; text: unknown; where: Where }
  | { kind: 'loop'; variable: string; start: string; stop: string; step: string; body: ForwardItem[]; where: Where }
  | { kind: 'branch'; test: unknown; then: ForwardItem[]; other: ForwardItem[]; where: Where };

interface Nest {
  base: NodeCopies;
  drawn: number;
  real: Size;
}

interface LoopLevel {
  variable: string;
  index: number;
  grid?: boolean;
  item?: ForwardItem;
}

interface GridWrite {
  accumulate: boolean;
  values: Node[];
  tiles: Node[];
  regions: TileRegion[];
  where: Where;
}

interface CopyStore {
  region: TileRegion;
  value: Node;
}

interface CopyRun {
  index: number[];
  expand: string;
  writes: Map<string, GridWrite>;
  stores: Map<string, CopyStore[]>;
  adds: Set<string>;
  views: Map<string, Node>;
  tiles: Map<string, number>;
}

function spans(region: TileRegion, shape: readonly number[]): [number, number][] {
  return region.map((entry, d) => (entry === null ? [0, shape[d]] : typeof entry === 'number' ? [entry, entry + 1] : entry));
}

function overlapping(a: TileRegion, b: TileRegion, shape: readonly number[]): boolean {
  const sb = spans(b, shape);
  return spans(a, shape).every(([start, stop], d) => Math.max(start, sb[d][0]) < Math.min(stop, sb[d][1]));
}

function sameRegion(a: TileRegion, b: TileRegion): boolean {
  return a.every((entry, d) => JSON.stringify(entry) === JSON.stringify(b[d]));
}

function boundingRegion(regions: TileRegion[], shape: readonly number[]): TileRegion {
  return shape.map((_, d) => {
    const entries = regions.map((region) => region[d]);
    if (entries.some((entry) => entry === null)) return null;
    if (entries.every((entry) => entry === entries[0] && typeof entry === 'number')) return entries[0];
    const all = regions.map((region) => spans(region, shape)[d]);
    return [Math.min(...all.map(([start]) => start)), Math.max(...all.map(([, stop]) => stop))];
  });
}

function regionWithin(region: TileRegion, outer: TileRegion): TileRegion {
  return region.flatMap((entry, d) => {
    const box = outer[d];
    if (typeof box === 'number') return [];
    if (box === null || entry === null) return [entry];
    return [typeof entry === 'number' ? entry - box[0] : [entry[0] - box[0], entry[1] - box[0]]];
  });
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
  private readonly loopIds = new Map<ForwardItem, number>();
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
    const dims = this.gridDims(call);
    const seen = new Set<string>();
    for (const item of items) {
      if (item.kind !== 'assign') continue;
      if (item.target !== undefined && dims === undefined) fail(item.where, `'${item.name}[...]' writes into a tensor, which forward does only inside a loop; outside a loop, give the result a new name`);
      if (item.target !== undefined) continue;
      if (seen.has(item.name)) fail(item.where, `name '${item.name}' is already defined in ${def.name}`);
      seen.add(item.name);
    }
    this.scanAliases(call, flatAssignments(items));
    if (dims === undefined) this.runItems(call, items, []);
    else this.runGrid(call, items, dims);
    call.finished = true;
    this.markShared(call);
  }

  private gridDims(call: Call): GridDim[] | undefined {
    const def = call.instance.def;
    const layoutWhere = child(def.where, def.decl, 'layout');
    const layout = record(def.decl.layout, layoutWhere, 'layout');
    const arrangementWhere = child(layoutWhere, layout, 'grid');
    const arrangements = record(layout.grid, arrangementWhere, 'layout.grid');
    if (def.decl.grid === undefined) {
      if (layout.grid !== undefined) fail(arrangementWhere, `layout.grid arranges the copies of a grid module; ${def.name} has no grid`);
      return undefined;
    }
    const where = child(def.where, def.decl, 'grid');
    const grid = record(def.decl.grid, where, 'grid');
    const names = Object.keys(grid);
    if (names.length === 0) fail(where, 'grid needs at least one dimension, such as grid: { m: triton.cdiv(T, block_m) }');
    for (const [name, value] of Object.entries(arrangements)) {
      const at = child(arrangementWhere, arrangements, name);
      if (!names.includes(name)) fail(at, `layout.grid names '${name}', which is not a dimension of the grid of ${def.name}; its dimensions are ${names.join(', ')}`);
      if (typeof value !== 'string' || !ARRANGEMENTS.has(value)) fail(at, `layout.grid '${name}' must be expand or stack`);
    }
    return names.map((name) => {
      const at = child(where, grid, name);
      if (!NAME.test(name)) fail(at, `'${name}' is not a valid name`);
      if (call.instance.args.has(name) || call.instance.members.has(name) || call.bound.has(name) || call.values.has(name)) fail(at, `name '${name}' is already defined in ${def.name}`);
      const count = this.number(grid[name], this.callEnv(call, at));
      if (!Number.isInteger(count.toy) || count.toy < 1) fail(at, `grid '${name}' must be a positive integer, got ${count.toy}`);
      return { name, count, arrangement: (arrangements[name] ?? 'stack') as Arrangement };
    });
  }

  private runGrid(call: Call, items: ForwardItem[], dims: GridDim[]): void {
    const def = call.instance.def;
    const indices = dims.reduce<number[][]>((acc, dim) => acc.flatMap((prefix) => Array.from({ length: dim.count.toy }, (_, i) => [...prefix, i])), [[]]);
    const run = new GridRun(
      dims,
      indices.map((index) => ({ index, locals: new Map<string, Value>(dims.map((dim, d) => [dim.name, { toy: index[d], real: 0 }])) })),
    );
    call.grid = run;
    const base = this.g.copiesNow();
    for (const inner of innerLoops(items)) this.gridNest(call, dims, indices, inner, base);
    const snapshot = new Map(call.values);
    const writes = new Map<string, GridWrite>();
    const reset = () => {
      call.values.clear();
      for (const [name, value] of snapshot) call.values.set(name, value);
    };
    const runCopies = () => {
      for (const copy of run.copies) {
        reset();
        for (const [name, value] of copy.locals) call.values.set(name, value);
        const before = this.g.nodes.length;
        const expand = dims.flatMap((dim, d) => (dim.arrangement === 'expand' ? [copy.index[d]] : [])).join(',');
        const copyRun: CopyRun = { index: copy.index, expand, writes, stores: new Map(), adds: new Set(), views: new Map(), tiles: new Map() };
        this.runItems(
          call,
          items,
          dims.map((dim, d) => ({ variable: dim.name, index: copy.index[d], grid: true })),
          copyRun,
        );
        for (const [name, stores] of copyRun.stores) {
          const write = writes.get(name) as GridWrite;
          for (const store of stores) {
            write.tiles.push(store.value);
            write.regions.push(store.region);
          }
        }
        for (const node of this.g.nodes.slice(before)) run.copyOf.set(node, copy.index);
      }
    };
    const total = combine(
      dims.map((dim) => dim.count),
      (...counts) => counts.reduce((a, b) => a * b, 1),
    );
    const real = this.dimSpec(total);
    if (real === total.toy) runCopies();
    else this.g.realCopies(total.toy, real, runCopies);
    reset();
    const outputsWhere = child(def.where, def.decl, 'outputs');
    const declared = this.fieldOf(def, 'outputs');
    for (const name of Object.keys(declared)) {
      if (!writes.has(name)) fail(child(outputsWhere, declared, name), `output '${name}' of ${def.name} is not an input that the grid writes; a grid module returns only the inputs it writes`);
    }
    const written: Node[] = [];
    for (const [name, write] of writes) {
      if (!(name in declared)) fail(write.where, `the grid writes input '${name}', so '${name}' must also be an output of ${def.name}, with the same shape`);
      const env = this.callEnv(call, write.where);
      const alias = this.resultAlias(call, name, env);
      let node: Node;
      try {
        node = this.g.scatter_tiles(call.inputs.get(name) as Node, write.tiles, write.regions, write.accumulate, { name: alias.path, tex: alias.tex, elem: elemOf(alias.tex) });
      } catch (error) {
        return fail(write.where, (error as Error).message);
      }
      call.values.set(name, node);
      if (!call.results.includes(name)) call.results.push(name);
      call.resultValues.push({ name, value: node, where: write.where });
      this.nodes.set(join(call.prefix, name), node);
      this.nodes.set(node.name, node);
      written.push(node);
      run.outputs.set(node, write.values);
    }
    call.lines.push({ nodes: written, calls: [] });
  }

  private gridNest(call: Call, dims: GridDim[], indices: number[][], inner: Extract<ForwardItem, { kind: 'loop' }>, base: NodeCopies): void {
    const env = this.callEnv(call, inner.where);
    const names = new Map<string, number>();
    for (const bound of [inner.start, inner.stop]) countNames(this.parse(bound, env), names);
    const mentioned = dims.flatMap((dim, d) => (names.has(dim.name) ? [d] : []));
    if (mentioned.length === 0) return;
    const step = this.loopStep(inner, env);
    const spanAt = (values: Num[]): Num => {
      dims.forEach((dim, d) => call.values.set(dim.name, values[d]));
      const from = this.number(inner.start, env);
      const to = this.number(inner.stop, env);
      dims.forEach((dim) => call.values.delete(dim.name));
      return combine([from, to, step], rangeCount);
    };
    let drawn = 0;
    for (const index of indices) drawn += spanAt(index.map((i) => ({ toy: i, real: 0 }))).toy;
    if (drawn === 0) return;
    const key = (d: number) => `${LOOP_KEY}${dims[d].name}`;
    const span = realOf(spanAt(dims.map((_, d): Num => (mentioned.includes(d) ? { toy: 0, real: 0, derive: (sizes) => sizes[key(d)] } : { toy: 0, real: 0 }))));
    const counts = dims.map((dim) => realOf(dim.count));
    const derive: RealSize = (sizes, shapes) => {
      let total = 0;
      const visit = (k: number, at: Record<string, number>): void => {
        if (k === mentioned.length) {
          total += span(at, shapes);
          return;
        }
        const d = mentioned[k];
        const count = counts[d](sizes, shapes);
        for (let i = 0; i < count; i++) visit(k + 1, { ...at, [key(d)]: i });
      };
      visit(0, { ...sizes });
      return dims.reduce((acc, _, d) => (mentioned.includes(d) ? acc : acc * counts[d](sizes, shapes)), total);
    };
    const real: Size = this.g.real === undefined ? drawn : { toy: drawn, real: derive(this.g.realSizes(), this.g.realShapes()), derive };
    this.nests.set(inner, { base, drawn, real });
  }

  private markShared(call: Call): void {
    const def = call.instance.def;
    if (def.decl.shared === undefined) return;
    const where = child(def.where, def.decl, 'shared');
    const shared = def.decl.shared;
    if (def.decl.kernel !== true) fail(where, `shared lists the tensors a kernel keeps in shared memory, so it needs kernel: true; ${def.name} is not a kernel`);
    if (!Array.isArray(shared) || !shared.every((name) => typeof name === 'string')) fail(where, 'shared must be a list of forward names');
    const outputs = this.fieldOf(def, 'outputs');
    (shared as string[]).forEach((name, i) => {
      const at = child(where, shared, i);
      if (name in outputs) fail(at, `'${name}' is an output of ${def.name}; outputs stay in device memory, so they cannot be shared`);
      const values = call.resultValues.filter((entry) => entry.name === name && entry.value instanceof Node);
      if (values.length === 0) fail(at, `'${name}' is not a tensor computed in the forward of ${def.name}`);
      for (const { value } of values) for (const node of [...(call.anon.get(value as Node) ?? []), value as Node]) call.shared.add(node);
    });
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
      return { kind: 'assign', name: written === null ? key : written[1], target: written === null ? undefined : written[2], accumulate: written?.[3] !== undefined, text: item[key], where: child(at, item, key) };
    });
  }

  private iterationsOf(call: Call, name: string, loops: { variable: string }[], before: Value | undefined): Iterations {
    const iterations = call.iterations.get(name) ?? { variables: [], values: new Map<string, Value>(), before };
    if (loops.length > iterations.variables.length) iterations.variables = loops.map((loop) => loop.variable);
    call.iterations.set(name, iterations);
    return iterations;
  }

  private skipBranch(call: Call, items: ForwardItem[], loops: LoopLevel[]): void {
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

  private copyReads(texts: unknown[], env: Env, copy: CopyRun, target?: string): Set<string> {
    const names = new Map<string, number>();
    for (const text of texts) countNames(this.parse(text, env), names);
    if (target !== undefined) names.set(target, (names.get(target) ?? 0) - 1);
    const read = new Set([...names].filter(([, count]) => count > 0).map(([name]) => name));
    for (const name of copy.adds) if (read.has(name)) fail(env.where, `'${name}' is read after this copy adds to it with +=; the copies add to ${name} at the same time, so its value is known only after the grid finishes`);
    return read;
  }

  private checkControlReads(text: unknown, env: Env, copy: CopyRun | undefined): void {
    if (copy === undefined) return;
    const read = this.copyReads([text], env, copy);
    for (const name of copy.stores.keys()) if (read.has(name)) fail(env.where, `'${name}' is read in an if test or a loop bound after this copy writes it; assign the part you need to a name first`);
  }

  private readCopy(call: Call, texts: unknown[], env: Env, copy: CopyRun, make: Make, target?: string): void {
    const read = this.copyReads(texts, env, copy, target);
    for (const [name, stores] of copy.stores) {
      if (!read.has(name) || copy.views.has(name)) continue;
      const tex = this.ownTex(call.instance, name, env);
      const view = this.g.scatter_tiles(call.inputs.get(name) as Node, stores.map((store) => store.value), stores.map((store) => store.region), false, { name: this.anonPath(make), tex, elem: elemOf(tex) });
      make.anon.push(view);
      copy.views.set(name, view);
      call.values.set(name, view);
    }
  }

  private loopId(item: ForwardItem): number {
    let id = this.loopIds.get(item);
    if (id === undefined) {
      id = this.loopIds.size;
      this.loopIds.set(item, id);
    }
    return id;
  }

  private runItems(call: Call, items: ForwardItem[], loops: LoopLevel[], copy?: CopyRun): void {
    const def = call.instance.def;
    for (const item of items) {
      const env = this.callEnv(call, item.where);
      if (item.kind === 'branch') {
        this.checkControlReads(item.test, env, copy);
        const test = this.expression(item.test, env);
        if (typeof test !== 'boolean') fail(item.where, `if must be True or False, got ${describe(test)}`);
        this.runItems(call, test ? item.then : item.other, loops, copy);
        this.skipBranch(call, test ? item.other : item.then, loops);
        continue;
      }
      if (item.kind === 'loop') {
        if (call.instance.members.has(item.variable) || call.instance.args.has(item.variable) || call.bound.has(item.variable) || loops.some((loop) => loop.grid && loop.variable === item.variable)) fail(item.where, `name '${item.variable}' is already defined in ${def.name}`);
        for (const bound of [item.start, item.stop, item.step]) this.checkControlReads(bound, env, copy);
        const start = this.number(item.start, env);
        const stop = this.number(item.stop, env);
        const step = this.loopStep(item, env);
        if (!Number.isInteger(start.toy) || !Number.isInteger(stop.toy)) fail(item.where, `range(${item.start}, ${item.stop}) needs integers, got ${start.toy} and ${stop.toy}`);
        const span = combine([start, stop, step], rangeCount);
        const count = span.toy;
        const rounds = call.rounds.get(item.variable) ?? { outer: loops.map((loop) => loop.variable), values: new Map<string, number[]>(), loops: new Set<object>() };
        rounds.loops.add(item);
        rounds.values.set(loops.map((loop) => loop.index).join(','), rangeValues(start.toy, stop.toy, step.toy));
        call.rounds.set(item.variable, rounds);
        if (count === 0) continue;
        const first: Num = start.derive === undefined ? { toy: start.toy, real: start.real } : { toy: start.toy, real: start.real, derive: start.derive };
        const run = () => {
          for (const k of rangeValues(start.toy, stop.toy, step.toy)) {
            call.values.set(item.variable, { ...first, toy: k });
            this.runItems(call, item.body, [...loops, { variable: item.variable, index: k, item }], copy);
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
      const recorded = loops.length > 0;
      const inLoop = loops.some((loop) => !loop.grid);
      const writesInput = copy !== undefined && item.target !== undefined;
      const defined = call.values.has(name);
      if ((!inLoop && defined && !writesInput) || call.bound.has(name) || call.instance.members.has(name) || call.instance.args.has(name)) fail(where, `name '${name}' is already defined in ${def.name}`);
      if (item.target !== undefined && !defined) fail(where, `'${name}' must be defined before '${name}[${item.target}]' writes into it`);
      if (recorded && defined && loops.some((loop) => loop.variable === name)) fail(where, `'${name}' is the loop variable`);
      const suffix = loops.map((loop) => `[${loop.index}]`).join('');
      const alias = this.resultAlias(call, name, env);
      const tex = loops.filter((loop) => !loop.grid).reduce((acc, loop) => addMark(acc, `_{${loop.index}}`), alias.tex);
      const tile = writesInput ? (copy.tiles.get(`${alias.path}${suffix}`) ?? 0) + 1 : 1;
      if (writesInput) copy.tiles.set(`${alias.path}${suffix}`, tile);
      const path = tile === 1 ? alias.path : `${alias.path}#${tile}`;
      const make: Make = { name, path: tile === 1 ? `${alias.path}${suffix}` : `${alias.path}${suffix}#${tile}`, suffix, tex, anon: [] };
      const before = this.g.nodes.length;
      const calls = call.children.length;
      if (copy !== undefined) this.readCopy(call, writesInput ? [item.text, `${name}[${item.target}]`] : [item.text], env, copy, make, writesInput ? name : undefined);
      const value = copy !== undefined && item.target !== undefined ? this.gridWrite(call, item, env, make, copy) : item.target === undefined ? this.expression(item.text, env, make) : this.write(item.target, item.text, item.accumulate, name, env, make);
      const inner = call.children.slice(calls);
      const nested = new Set(inner.flatMap((c) => c.created));
      const created = this.g.nodes.slice(before).filter((n) => !nested.has(n));
      call.lines.push(copy === undefined ? { nodes: created, calls: inner } : { nodes: created, calls: inner, copy: copy.index });
      if (recorded) this.slot(call, loops, copy, path, make.path, created);
      if (writesInput) {
        for (const node of created) this.nodes.set(node.name, node);
        if (value instanceof Node) {
          const iterations = this.iterationsOf(call, name, loops, undefined);
          const key = loops.map((loop) => loop.index).join(',');
          const earlier = iterations.values.get(key);
          const anon = [...(earlier instanceof Node ? [...(this.anon.get(earlier) ?? []), earlier] : []), ...make.anon];
          if (anon.length > 0) {
            call.anon.set(value, [...(call.anon.get(value) ?? []), ...anon]);
            this.anon.set(value, [...(this.anon.get(value) ?? []), ...anon]);
          }
          iterations.values.set(key, value);
        }
        continue;
      }
      if (value === null && item.target === undefined && !inLoop) {
        call.values.set(name, null);
        if (recorded) this.iterationsOf(call, name, loops, undefined).values.set(loops.map((loop) => loop.index).join(','), null);
        continue;
      }
      const numeric = (isNum(value) || typeof value === 'boolean') && item.target === undefined;
      if (numeric && this.g.nodes.length !== before) fail(where, `'${name}' is ${describe(value)}, but computing it created tensors`);
      if (!numeric && !(value instanceof Node) && !(value instanceof Result)) fail(where, `'${name}' must be a tensor, a number, a bool, a module result or None, got ${describe(value)}`);
      const previous = call.values.get(name);
      call.values.set(name, value);
      if (recorded) {
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

  private slot(call: Call, loops: LoopLevel[], copy: CopyRun | undefined, path: string, made: string, created: Node[]): void {
    const body = loops.flatMap((loop) => (loop.item === undefined ? [] : [this.loopId(loop.item)])).join('/');
    for (const node of created) {
      const key = `${body}|${copy?.expand ?? ''}|${node.name.startsWith(made) ? `${path}${node.name.slice(made.length)}` : node.name}`;
      const members = call.slots.get(key);
      if (members === undefined) call.slots.set(key, [node]);
      else members.push(node);
    }
  }

  private gridWrite(call: Call, item: Extract<ForwardItem, { kind: 'assign' }>, env: Env, make: Make, copy: CopyRun): Value {
    const { name, target, where } = item;
    const input = call.inputs.get(name);
    if (input === undefined) return fail(where, `'${name}[${target}]' writes into '${name}', but a grid module only writes its inputs`);
    const index = this.parse(`${name}[${target}]`, env);
    if (index.k !== 'index') return this.fail(env, `'${name}[${target}]' is not an index`);
    const region = this.tileRegion(index.items, input.shape, env, `${name}[${target}]`);
    const ast = this.parse(item.text, env);
    const value = ast.k === 'name' ? this.reference(item.text, env) : this.expression(item.text, env, make);
    if (!(value instanceof Node)) return fail(where, `'${name}[${target}]' writes ${describe(value)}; a grid module writes tensors`);
    const write = copy.writes.get(name) ?? { accumulate: item.accumulate, values: [], tiles: [], regions: [], where };
    if (write.accumulate !== item.accumulate) fail(where, `'${name}' is written with both = and +=; a grid module either writes each input or adds to it`);
    copy.writes.set(name, write);
    write.values.push(value);
    if (item.accumulate) {
      write.tiles.push(value);
      write.regions.push(region);
      copy.adds.add(name);
      return value;
    }
    copy.stores.set(name, this.storeInCopy(copy.stores.get(name) ?? [], { region, value }, input, make, this.resultAlias(call, name, env).tex));
    copy.views.delete(name);
    call.values.set(name, input);
    return value;
  }

  private storeInCopy(stores: CopyStore[], latest: CopyStore, input: Node, make: Make, tex: string): CopyStore[] {
    let region = latest.region;
    let rest = stores;
    const earlier: CopyStore[] = [];
    for (;;) {
      const hit = rest.filter((store) => overlapping(store.region, region, input.shape));
      if (hit.length === 0) break;
      earlier.push(...hit);
      rest = rest.filter((store) => !hit.includes(store));
      region = boundingRegion([region, ...hit.map((store) => store.region)], input.shape);
    }
    if (earlier.length === 0) return [...stores, latest];
    if (sameRegion(region, latest.region)) return [...rest, latest];
    const made = (node: Node) => {
      make.anon.push(node);
      return node;
    };
    const options = () => ({ name: this.anonPath(make), tex, elem: elemOf(tex) });
    const whole = earlier.find((store) => sameRegion(store.region, region));
    const base = whole?.value ?? made(this.g.scatter_tiles(this.regionOf(input, region, (node) => made(node(options()))), earlier.map((store) => store.value), earlier.map((store) => regionWithin(store.region, region)), false, options()));
    const value = made(this.g.scatter_tiles(base, [latest.value], [regionWithin(latest.region, region)], false, options()));
    return [...rest, { region, value }];
  }

  private regionOf(input: Node, region: TileRegion, make: (op: (options: { name: string; tex: string; elem: string }) => Node) => Node): Node {
    let node = input;
    for (let d = region.length - 1; d >= 0; d--) {
      const entry = region[d];
      const from = node;
      if (typeof entry === 'number') node = make((options) => this.g.select(from, d, entry, options));
      else if (entry !== null) node = make((options) => this.g.narrow(from, d, entry[0], entry[1] - entry[0], options));
    }
    return node;
  }

  private tileRegion(items: IndexItem[], shape: readonly number[], env: Env, text: string): TileRegion {
    const ellipsis = items.findIndex((item) => item.k === 'ellipsis');
    if (items.filter((item) => item.k === 'ellipsis').length > 1) this.fail(env, `'${text}' has more than one '...'`);
    const given = items.filter((item) => item.k !== 'ellipsis');
    if (given.length > shape.length) this.fail(env, `'${text}' has ${given.length} indices for a tensor with shape (${shape.join(', ')})`);
    const fill = Array.from({ length: shape.length - given.length }, (): IndexItem | undefined => undefined);
    const expanded = ellipsis < 0 ? [...given, ...fill] : [...items.slice(0, ellipsis), ...fill, ...items.slice(ellipsis + 1)];
    return expanded.map((item, d) => {
      const size = shape[d];
      if (item === undefined || item.k === 'ellipsis' || (item.k === 'slice' && item.start === undefined && item.stop === undefined && item.step === undefined)) return null;
      if (item.k === 'at') {
        if (item.value.k === 'none') return this.fail(env, `'${text}' cannot write through None`);
        const i = this.int(item.value, env, 0);
        if (i < -size || i >= size) this.fail(env, `index ${i} is out of range for dimension ${d} of size ${size} in '${text}'`);
        return i < 0 ? size + i : i;
      }
      if (item.step !== undefined && this.int(item.step, env, 0) !== 1) this.fail(env, `'${text}' writes a slice with a step; a grid module writes slices with step 1`);
      const bound = (ast: Ast | undefined, fallback: number) => {
        if (ast === undefined) return fallback;
        const v = this.int(ast, env, 0);
        return Math.min(size, Math.max(0, v < 0 ? size + v : v));
      };
      const start = bound(item.start, 0);
      return [start, Math.max(start, bound(item.stop, size))];
    });
  }

  private write(target: string, text: unknown, accumulate: boolean, name: string, env: Env, make: Make): Value {
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
    const given = this.parse(text, env);
    const whole: Ast = { k: 'name', name, pos };
    const value = (): Value => this.evaluate(accumulate ? { k: 'bin', op: '+', a: picks.length === 0 ? whole : index, b: given, pos } : given, env, mode, false);
    if (picks.length === 0) {
      if (index.items.length !== 1 || ellipsis !== 0) this.fail(env, `'${name}[${target}]' writes the whole tensor; write '${name}[...]' to update it in place, or give the result a new name`);
      const current = this.evaluate(whole, env, mode, false);
      return this.apply('slice_scatter', [current, value(), num(0), null, null], new Map(), env, mode, true, pos);
    }
    const scalar = (ast: Ast | undefined): Value => (ast === undefined ? null : this.evaluate(ast, env, mode, false));
    const part = (base: Value, item: IndexItem, dim: number): Value => {
      const items: IndexItem[] = dim >= 0 ? [...Array.from({ length: dim }, (): IndexItem => ({ k: 'slice' })), item] : [{ k: 'ellipsis' }, item, ...Array.from({ length: -dim - 1 }, (): IndexItem => ({ k: 'slice' }))];
      const local = { ...env, locals: new Map([...env.locals, [' base', base]]) };
      return this.evaluate({ k: 'index', obj: { k: 'name', name: ' base', pos }, items, pos }, local, mode, false);
    };
    const into = (base: () => Value, rest: { item: IndexItem; dim: number }[], top: boolean): Value => {
      if (rest.length === 0) return value();
      const [{ item, dim }, ...others] = rest;
      const current = base();
      if (item.k === 'at') {
        if (item.value.k === 'name' && item.value.name === 'None') this.fail(env, `'${name}[${target}]' cannot write through None`);
        const shifted = others.map((o) => ({ ...o, dim: dim >= 0 && o.dim > dim ? o.dim - 1 : o.dim }));
        const src = into(() => part(current, item, dim), shifted, false);
        return this.apply('select_scatter', [current, src, num(dim), scalar(item.value)], new Map(), env, mode, top, pos);
      }
      if (item.k !== 'slice') return this.fail(env, `'${name}[${target}]' has an entry that cannot be written`);
      const src = into(() => part(current, item, dim), others, false);
      return this.apply('slice_scatter', [current, src, num(dim), scalar(item.start), scalar(item.stop)], new Map(), env, mode, top, pos);
    };
    return into(() => this.evaluate(whole, env, mode, false), picks, true);
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
