import { Node } from '../../autograd';
import type { LevelSpec } from '../../scene3d/levels';
import { type Where, child, fail } from '../source';
import { describe } from './expressions';
import type { PageCompiler } from './page';
import { type Call, Instance, Result } from './types';
import { checkKeys, record } from './util';

const LAYOUT_KEYS: ReadonlySet<string> = new Set(['place', 'merge', 'grid']);
const PLACE_KEYS = ['with', 'left', 'right', 'before', 'after'] as const;

type Placement = (typeof PLACE_KEYS)[number];

interface Slots {
  firstOf: ReadonlyMap<Node, Node>;
  hidden: ReadonlySet<Node>;
}

export function layoutOf(c: PageCompiler, root: Call): LevelSpec {
  const spec: LevelSpec = { lines: [], sides: [], merges: [], slots: slotGroups(root), grids: [] };
  const firstOf = new Map(spec.slots.flatMap((members) => members.map((node): [Node, Node] => [node, members[0]])));
  const hidden = new Set(spec.slots.flatMap((members) => members.slice(1)));
  spec.lines = levelsOf(c, root, spec, { firstOf, hidden });
  return spec;
}

function slotGroups(root: Call): Node[][] {
  const parent = new Map<Node, Node>();
  const find = (n: Node): Node => {
    const p = parent.get(n);
    if (p === undefined || p === n) return n;
    const top = find(p);
    parent.set(n, top);
    return top;
  };
  const union = (a: Node, b: Node) => {
    for (const n of [a, b]) if (!parent.has(n)) parent.set(n, n);
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(rb, ra);
  };
  const visit = (call: Call) => {
    for (const members of call.slots.values()) for (const m of members.slice(1)) union(members[0], m);
    for (const group of repeatedCalls(call)) {
      const first = new Map(group[0].created.flatMap((n) => relativeName(group[0], n)).map(([key, n]) => [key, n]));
      for (const inner of group.slice(1)) {
        for (const [key, n] of inner.created.flatMap((node) => relativeName(inner, node))) {
          const counterpart = first.get(key);
          if (counterpart !== undefined) union(counterpart, n);
        }
      }
    }
    new Set([...call.children, ...call.lines.flatMap((line) => line.calls)]).forEach(visit);
  };
  visit(root);
  const groups = new Map<Node, Node[]>();
  for (const n of parent.keys()) {
    const top = find(n);
    groups.set(top, [...(groups.get(top) ?? []), n]);
  }
  return [...groups.values()].filter((members) => members.length > 1).map((members) => [...new Set(members)].sort((a, b) => a.id - b.id));
}

function relativeName(call: Call, n: Node): [string, Node][] {
  if (n.name.startsWith(`${call.prefix}.`)) return [[`in:${n.name.slice(call.prefix.length)}`, n]];
  for (const [kind, values] of [['out', call.outputs], ['arg', call.inputs]] as const) {
    for (const [name, value] of values) {
      if (n === value) return [[`${kind}:${name}`, n]];
      const rest = n.name.slice(value.name.length);
      if (n.name.startsWith(value.name) && (rest.startsWith('/') || rest.startsWith('['))) return [[`${kind}:${name}${rest}`, n]];
    }
  }
  return [];
}

function repeatedCalls(call: Call): Call[][] {
  const expand = new Set(call.grid?.dims.filter((dim) => dim.arrangement === 'expand').map((dim) => dim.name) ?? []);
  const producer = new Map(call.children.flatMap((child) => [...child.outputs.values()].map((node): [Node, Call] => [node, child])));
  const groups: Call[][] = [];
  for (const { variables, values } of call.iterations.values()) {
    const byKey = new Map<string, Call[]>();
    for (const [rounds, value] of values) {
      const made = value instanceof Result ? value.call : value instanceof Node ? producer.get(value) : undefined;
      if (made === undefined) continue;
      const indices = rounds.split(',');
      const key = variables.flatMap((name, d) => (expand.has(name) ? [indices[d]] : [])).join(',');
      const found = byKey.get(key);
      if (found === undefined) byKey.set(key, [made]);
      else if (!found.includes(made) && found[0].instance === made.instance) found.push(made);
    }
    groups.push(...[...byKey.values()].filter((calls) => calls.length > 1));
  }
  return groups;
}

function levelsOf(c: PageCompiler, call: Call, spec: LevelSpec, slots: Slots): Node[][] {
  const { firstOf } = slots;
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
  if (c.inBackward(call)) return arranged(call, slots, (inner) => levelsOf(c, inner, spec, slots));
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
  const levels = arranged(call, slots, (inner) => {
    if (deferred.has(inner)) return [];
    const pair = later.get(inner);
    if (pair === undefined) return levelsOf(c, inner, spec, slots);
    const left = levelsOf(c, pair[0], spec, slots);
    const right = levelsOf(c, pair[1], spec, slots);
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

function arranged(call: Call, slots: Slots, inner: (call: Call) => Node[][]): Node[][] {
  const { firstOf, hidden } = slots;
  const levels: Node[][] = [];
  const copies = new Map<string, Node[][]>();
  const cursors = new Map<Node[][], number>();
  const flush = () => {
    levels.push(...sideBySide([...copies.values()]));
    copies.clear();
    cursors.delete(levels);
  };
  let sequence: string | undefined;
  for (const line of call.lines) {
    const dims = call.grid?.dims ?? [];
    const copy = line.copy;
    const key = copy === undefined ? undefined : dims.flatMap((dim, d) => (dim.arrangement === 'expand' ? [copy[d]] : [])).join(',');
    if (key === undefined) flush();
    const out = key === undefined ? levels : (copies.get(key) ?? copies.set(key, []).get(key)!);
    const copyKey = copy?.join(',');
    if (copyKey !== undefined && copyKey !== sequence) cursors.set(out, -1);
    sequence = copyKey;
    const cursor = () => cursors.get(out) ?? out.length - 1;
    const insert = (level: Node[]) => {
      const at = cursor() + 1;
      out.splice(at, 0, level);
      cursors.set(out, at);
    };
    const anchor = (nodes: readonly Node[]): number => {
      const leads = new Set(nodes.flatMap((n) => firstOf.get(n) ?? []));
      if (leads.size === 0) return -1;
      let found = -1;
      out.forEach((level, i) => {
        if (level.some((n) => leads.has(n))) found = i;
      });
      return found;
    };
    for (const called of line.calls) {
      const levelsOfCall = inner(called);
      if (levelsOfCall.length > 0) levelsOfCall.forEach(insert);
      const at = anchor(called.created);
      if (at >= 0) cursors.set(out, Math.max(levelsOfCall.length > 0 ? cursor() : at, at));
    }
    const nodes = line.nodes.filter((node) => !hidden.has(node));
    if (nodes.length > 0) insert(nodes);
    const at = anchor(line.nodes);
    if (at >= 0) cursors.set(out, nodes.length > 0 ? Math.max(cursor(), at) : at);
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
