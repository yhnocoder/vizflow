import { DTYPES, Node } from '../../autograd';
import type { LayoutEntry, LayoutTree } from '../../scene3d';
import { type Ast, ParseError, parseRange } from '../expr';
import type { ModuleDef } from '../load';
import { type Where, child, fail } from '../source';
import { type PreludeItem, define, marked, quantPrelude } from './declare';
import { compileInstance } from './instance';
import { registerFormula } from './formula';
import { FOR_KEYS, GROUP_KEYS, ITEM_KEYS } from './keys';
import { closeModule, openModule, titleOf } from './modules';
import type { Compiler } from './text';
import type { Env, Make, Scope, Slot } from './types';
import { checkKeys, isRecord, join } from './util';

interface Lanes {
  start: number;
  members: { name: string | undefined; created: Node[]; slot: Slot | undefined }[];
  parallel?: { kind: 'parallel'; branches: LayoutTree[] };
}

type Beside = { value: unknown; env: Env; where: Where };

function countInstances(c: Compiler, items: unknown, env: Env, counts: Map<ModuleDef, number>): void {
  if (!Array.isArray(items)) return;
  for (const item of items) {
    if (!isRecord(item)) continue;
    if ('for' in item) {
      const iterations = rangeOf(c, item.in, env);
      for (const v of iterations) countInstances(c, item.body, { ...env, loops: new Map(env.loops).set(String(item.for), v) }, counts);
      continue;
    }
    for (const value of Object.values(item)) {
      if (!isRecord(value)) continue;
      if (typeof value.use === 'string') {
        const def = env.scope.def.namespace.get(value.use);
        if (def !== undefined) counts.set(def, (counts.get(def) ?? 0) + 1);
      } else if (Array.isArray(value.body)) {
        countInstances(c, value.body, env, counts);
      }
    }
  }
}

function rangeOf(c: Compiler, text: unknown, env: Env): number[] {
  if (typeof text !== 'string') c.fail(env, "'in' must be a range like '1..H'");
  let from: Ast;
  let to: Ast;
  try {
    [from, to] = parseRange(text);
  } catch (error) {
    if (error instanceof ParseError) c.fail(env, `${error.message} in '${text}'`, error.offset);
    throw error;
  }
  const a = c.int(from, env, 0);
  const b = c.int(to, env, 0);
  return Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);
}

export function compileBody(c: Compiler, scope: Scope, decl: Record<string, unknown>, env: Env, entries: LayoutEntry[]): { slots: Slot[] } {
  const bodyWhere = child(env.where, decl, 'body');
  if (!Array.isArray(decl.body)) fail(bodyWhere, 'body must be a list');
  const prelude = quantPrelude(scope, decl, env);
  const counts = new Map<ModuleDef, number>();
  countInstances(c, [...prelude.map((p) => p.item), ...decl.body], env, counts);
  const ordinals = new Map<ModuleDef, number>();
  return { slots: compileItems(c, scope, decl.body, { ...env, where: bodyWhere }, entries, counts, ordinals, prelude) };
}

function compileItems(
  c: Compiler,
  scope: Scope,
  items: unknown[],
  env: Env,
  entries: LayoutEntry[],
  counts: Map<ModuleDef, number>,
  ordinals: Map<ModuleDef, number>,
  prelude: PreludeItem[] = [],
): Slot[] {
  const slots: Slot[] = [];
  const named = new Set<string>();
  let lanes: Lanes | undefined;
  const patches: (() => void)[] = [];
  c.pendingPatches.push(patches);
  const listed = [...prelude, ...items.map((item, index) => ({ item, where: child(env.where, items, index), after: undefined }))];
  listed.forEach(({ item, where: at, after }) => {
    if (!isRecord(item)) fail(at, 'a body item must be a mapping');
    const start = entries.length;
    const before = c.g.nodes.length;
    const slotCount = slots.length;
    const { name, beside } = compileItem(c, scope, item, { ...env, where: at }, entries, counts, ordinals, slots);
    const created = c.g.nodes.slice(before);
    const slot = slots.length > slotCount ? slots[slots.length - 1] : undefined;
    if (beside === undefined) {
      lanes = name === undefined ? undefined : { start, members: [{ name, created, slot }] };
    } else {
      const { target, group } = besideGroup(c, beside, lanes, named);
      checkLaneUse(group, created, name, target, beside.where);
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
    after?.();
  });
  c.pendingPatches.pop();
  for (const patch of patches) patch();
  return slots;
}

function besideGroup(c: Compiler, beside: Beside, lanes: Lanes | undefined, named: ReadonlySet<string>): { target: string; group: Lanes } {
  if (typeof beside.value !== 'string') fail(beside.where, 'beside must name an instance, a group or a for item');
  const target = c.concreteName(beside.value, { ...beside.env, where: beside.where }).name;
  if (lanes !== undefined && lanes.members.some((m) => m.name === target)) return { target, group: lanes };
  if (named.has(target)) {
    fail(beside.where, `beside '${target}': '${target}' is not the item right before this one; beside names the item right before it or an item beside that one`);
  }
  fail(beside.where, `beside '${target}': no instance, group or named for item '${target}' comes before this one in the same body`);
}

function checkLaneUse(lanes: Lanes, created: Node[], name: string | undefined, target: string, where: Where): void {
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

function compileItem(
  c: Compiler,
  scope: Scope,
  item: Record<string, unknown>,
  env: Env,
  entries: LayoutEntry[],
  counts: Map<ModuleDef, number>,
  ordinals: Map<ModuleDef, number>,
  slots: Slot[],
): { name: string | undefined; beside: Beside | undefined } {
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
    for (const v of rangeOf(c, item.in, { ...env, where: child(at, item, 'in') })) {
      const loopEnv: Env = { ...env, loops: new Map(env.loops).set(item.for, v), where: child(at, item, 'body') };
      const branch: LayoutTree = { kind: 'module', children: [] };
      const start = c.g.nodes.length;
      if (typeof item.brace === 'string') branch.title = c.ofText(item.brace, { ...loopEnv, where: child(at, item, 'brace') });
      const rec = branch.title === undefined ? undefined : openModule(c, join(scope.path, `${forName ?? item.for}[${v}]`), branch.title, undefined, {}, loopEnv);
      iterations.push(compileItems(c, scope, item.body, loopEnv, branch.children, counts, ordinals));
      branches.push(branch);
      if (rec !== undefined) closeModule(c, rec, c.g.nodes.slice(start));
    }
    const sequential = item.sequential ?? false;
    if (typeof sequential !== 'boolean') fail(child(at, item, 'sequential'), 'sequential must be true or false');
    if (sequential) entries.push(...branches);
    else entries.push({ kind: 'parallel', branches });
    slots.push({ kind: 'loop', iterations, sequential });
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
    slots.push(compileInstance(c, scope, name, braceTitle, indices, value, valueEnv, entries, counts, ordinals, before));
    return { name, beside: besideOf(value, valueWhere) };
  }
  if (isRecord(value) && 'body' in value) {
    checkKeys(value, GROUP_KEYS, valueWhere);
    if (!Array.isArray(value.body)) fail(child(valueWhere, value, 'body'), 'body must be a list');
    if (scope.local(name) !== undefined || scope.groups.has(name)) fail(valueWhere, `name '${name}' is already defined`);
    scope.groups.add(name);
    const group: LayoutTree = { kind: 'module', title: titleOf(c, value, braceTitle, valueEnv), children: [] };
    if (typeof value.subtitle === 'string') group.subtitle = value.subtitle;
    entries.push(group);
    const rec = openModule(c, join(scope.path, name), group.title as string, group.subtitle, value, valueEnv);
    const groupSlots = compileItems(c, scope, value.body, { ...valueEnv, where: child(valueWhere, value, 'body') }, group.children, counts, ordinals);
    registerFormula(c, scope.path, [{ decl: value, env: valueEnv }], rec);
    closeModule(c, rec, c.g.nodes.slice(before));
    slots.push({ kind: 'group', decl: value, where: valueWhere, loops: env.loops, slots: groupSlots });
    return { name, beside: besideOf(value, valueWhere) };
  }
  slots.push(compileNodeItem(c, scope, name, base, indices, value, valueEnv, entries, before));
  return { name: undefined, beside: undefined };
}

function itemDtype(c: Compiler, value: unknown, env: Env): string {
  if (typeof value !== 'string') fail(env.where, 'dtype must be a dtype name or the name of a tensor');
  if ((DTYPES as readonly string[]).includes(value)) return value;
  const node = c.expression(value, env, undefined);
  if (!(node instanceof Node)) fail(env.where, `dtype '${value}' is neither a dtype (${DTYPES.join(', ')}) nor a tensor`);
  return node.dtype;
}

function besideNode(c: Compiler, value: unknown, env: Env): Node {
  if (typeof value !== 'string') fail(env.where, 'beside on a node must name a tensor');
  const node = c.expression(value, env, undefined);
  if (!(node instanceof Node)) fail(env.where, `beside '${value}' is not a tensor`);
  return node;
}

function compileNodeItem(c: Compiler, scope: Scope, name: string, base: string, indices: number[], value: unknown, env: Env, entries: LayoutEntry[], before: number): Slot {
  let expr: unknown = value;
  let declared: string | undefined;
  let dtype: string | undefined;
  let exprWhere = env.where;
  let beside: Node | undefined;
  if (isRecord(value)) {
    checkKeys(value, ITEM_KEYS, env.where);
    if (!('op' in value)) fail(env.where, `'${name}' needs op, use or body`);
    expr = value.op;
    exprWhere = child(env.where, value, 'op');
    if (value.tex !== undefined) {
      if (typeof value.tex !== 'string') fail(child(env.where, value, 'tex'), 'tex must be a string');
      declared = value.tex;
    }
    if (value.dtype !== undefined) dtype = itemDtype(c, value.dtype, { ...env, where: child(env.where, value, 'dtype') });
    if (value.beside !== undefined) beside = besideNode(c, value.beside, { ...env, where: child(env.where, value, 'beside') });
  }
  if (scope.local(name) !== undefined || scope.groups.has(name)) fail(env.where, `name '${name}' is already defined in ${c.scopeLabel(scope)}`);
  const override = scope.texOverride.get(name) ?? scope.texOverride.get(base);
  let tex: string;
  if (override !== undefined) tex = override;
  else if (declared !== undefined) tex = marked(c.ofText(declared, env), scope, c.ofRefersToNode(declared, env));
  else if (scope.outputTex.has(name)) tex = scope.outputTex.get(name) as string;
  else tex = marked(c.loopTex(base, indices), scope, false);
  const make: Make = { path: join(scope.path, name), tex, anon: [], counter: 0, dtype };
  const result = c.expression(expr, { ...env, where: exprWhere }, make);
  if (!(result instanceof Node) || result.name !== make.path) fail(exprWhere, `'${name}' must compute a new node; its expression only refers to an existing value`);
  scope.anon.set(result, make.anon);
  define(c, scope, name, result, env);
  if (beside !== undefined) c.besides.set(result, beside);
  for (const node of make.anon) {
    entries.push({ kind: 'node', node });
    c.nodes.set(node.name, node);
  }
  entries.push({ kind: 'node', node: result });
  return { kind: 'node', name, node: result, created: c.g.nodes.slice(before) };
}
