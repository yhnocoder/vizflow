import { Node } from '../../autograd';
import type { LevelSpec } from '../../scene3d/levels';
import { type Where, child, fail } from '../source';
import { describe } from './expressions';
import type { PageCompiler } from './page';
import { type Call, Instance } from './types';
import { checkKeys, record } from './util';

const LAYOUT_KEYS: ReadonlySet<string> = new Set(['place', 'merge', 'grid']);
const PLACE_KEYS = ['with', 'left', 'right', 'before', 'after'] as const;

type Placement = (typeof PLACE_KEYS)[number];

export function layoutOf(c: PageCompiler, root: Call): LevelSpec {
  const spec: LevelSpec = { lines: [], sides: [], merges: [], slots: [], grids: [] };
  spec.lines = levelsOf(c, root, spec);
  return spec;
}

function levelsOf(c: PageCompiler, call: Call, spec: LevelSpec): Node[][] {
  const groups = [...call.slots.values()].filter((members) => members.length > 1);
  spec.slots.push(...groups);
  const firstOf = new Map(groups.flatMap((members) => members.map((node): [Node, Node] => [node, members[0]])));
  const hidden = new Set(groups.flatMap((members) => members.slice(1)));
  const grid = call.grid;
  if (grid !== undefined || call.instance.def.decl.kernel === true) {
    spec.grids.push({
      module: call.prefix,
      names: grid?.dims.map((dim) => dim.name) ?? [],
      counts: grid?.dims.map((dim) => dim.count.toy) ?? [],
      real: grid?.dims.map((dim) => c.dimSpec(dim.count)) ?? [],
      arrangement: grid?.dims.map((dim) => dim.arrangement) ?? [],
      copyOf: grid?.copyOf ?? new Map(),
      shared: call.shared,
      registers: new Set(call.created.filter((node) => node.kind === 'op' && c.g.onChip(node) && !call.shared.has(node))),
    });
  }
  if (c.inBackward(call)) return arranged(call, hidden, (inner) => levelsOf(c, inner, spec));
  const def = call.instance.def;
  const where = child(def.where, def.decl, 'layout');
  const layout = record(def.decl.layout, where, 'layout');
  checkKeys(layout, LAYOUT_KEYS, where);
  const mergeWhere = child(where, layout, 'merge');
  const mergeList = layout.merge ?? [];
  if (!Array.isArray(mergeList)) fail(mergeWhere, 'merge must be a list of [a, b] pairs of module calls');
  const pairs = mergeList.map((pair, i) => {
    const at = child(mergeWhere, mergeList, i);
    if (!Array.isArray(pair) || pair.length !== 2) fail(at, 'each merge entry is a pair of module calls, such as [q_norm, k_norm]');
    const calls = pair.map((ref, k) => callIn(c, call, ref, child(at, pair, k))) as [Call, Call];
    if (calls[0].instance.def !== calls[1].instance.def) fail(at, `merge needs two calls of the same module; ${calls[0].instance.name} is ${calls[0].instance.def.name} and ${calls[1].instance.name} is ${calls[1].instance.def.name}`);
    return calls;
  });
  const lineOf = (target: Call) => call.lines.findIndex((line) => line.calls.includes(target));
  const later = new Map<Call, [Call, Call]>();
  const deferred = new Set<Call>();
  for (const pair of pairs) {
    const [a, b] = lineOf(pair[0]) <= lineOf(pair[1]) ? pair : [pair[1], pair[0]];
    deferred.add(a);
    later.set(b, pair);
  }
  const levels = arranged(call, hidden, (inner) => {
    if (deferred.has(inner)) return [];
    const pair = later.get(inner);
    if (pair === undefined) return levelsOf(c, inner, spec);
    const left = levelsOf(c, pair[0], spec);
    const right = levelsOf(c, pair[1], spec);
    spec.merges.push({ left: left.flat(), right: right.flat() });
    return sideBySide([left, right]);
  });
  const placeWhere = child(where, layout, 'place');
  const place = record(layout.place, placeWhere, 'place');
  for (const [name, value] of Object.entries(place)) {
    const at = child(placeWhere, place, name);
    const named = nodeIn(c, call, name, at);
    const node = firstOf.get(named) ?? named;
    const placement = record(value, at, `place '${name}'`);
    checkKeys(placement, new Set(PLACE_KEYS), at);
    const keys = Object.keys(placement) as Placement[];
    if (keys.length !== 1) fail(at, `place '${name}' takes one of ${PLACE_KEYS.join(', ')}`);
    const key = keys[0];
    const placed = nodeIn(c, call, placement[key], child(at, placement, key));
    const target = firstOf.get(placed) ?? placed;
    const from = levels.findIndex((level) => level.includes(node));
    if (levels.findIndex((level) => level.includes(target)) < 0) fail(child(at, placement, key), `'${String(placement[key])}' is not computed in the forward of ${def.name}`);
    if (node === target) fail(at, `'${name}' cannot be placed relative to itself`);
    if (from < 0 && node.kind !== 'op') {
      if (key === 'before' || key === 'after') fail(at, `'${name}' is a ${node.kind}; place it on a level with with, left or right`);
      spec.sides.push({ node, side: key, of: target });
      continue;
    }
    if (from < 0) fail(at, `'${name}' is not computed in the forward of ${def.name}`);
    levels[from] = levels[from].filter((n) => n !== node);
    const to = levels.findIndex((level) => level.includes(target));
    if (key === 'before') levels.splice(to, 0, [node]);
    else if (key === 'after') levels.splice(to + 1, 0, [node]);
    else {
      levels[to].push(node);
      spec.sides.push({ node, side: key, of: target });
    }
  }
  return levels.filter((level) => level.length > 0);
}

function arranged(call: Call, hidden: ReadonlySet<Node>, inner: (call: Call) => Node[][]): Node[][] {
  const levels: Node[][] = [];
  const copies = new Map<string, Node[][]>();
  const flush = () => {
    levels.push(...sideBySide([...copies.values()]));
    copies.clear();
  };
  for (const line of call.lines) {
    const dims = call.grid?.dims ?? [];
    const copy = line.copy;
    const key = copy === undefined ? undefined : dims.flatMap((dim, d) => (dim.arrangement === 'expand' ? [copy[d]] : [])).join(',');
    if (key === undefined) flush();
    const out = key === undefined ? levels : (copies.get(key) ?? copies.set(key, []).get(key)!);
    for (const called of line.calls) out.push(...inner(called));
    const nodes = line.nodes.filter((node) => !hidden.has(node));
    if (nodes.length > 0) out.push(nodes);
  }
  flush();
  return levels;
}

function sideBySide(parts: Node[][][]): Node[][] {
  const length = Math.max(0, ...parts.map((part) => part.length));
  return Array.from({ length }, (_, i) => parts.flatMap((part) => part[i] ?? []));
}

function callIn(c: PageCompiler, call: Call, ref: unknown, where: Where): Call {
  const value = c.reference(ref, c.callEnv(call, where));
  if (!(value instanceof Instance)) fail(where, `'${String(ref)}' is ${describe(value)}; merge takes module calls`);
  const found = call.children.filter((inner) => inner.instance === value);
  if (found.length !== 1) fail(where, `'${String(ref)}' is called ${found.length} times in this forward; merge needs a module called once`);
  return found[0];
}

function nodeIn(c: PageCompiler, call: Call, ref: unknown, where: Where): Node {
  const value = c.reference(ref, c.callEnv(call, where));
  if (!(value instanceof Node)) fail(where, `'${String(ref)}' is ${describe(value)}; place takes tensors, weights and buffers`);
  return value;
}
