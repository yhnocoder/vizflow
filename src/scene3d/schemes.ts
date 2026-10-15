import type { Node } from '../autograd';

export const LEVEL_SCHEMES = ['lines', 'asap', 'alap', 'nested'] as const;
export type LevelScheme = (typeof LEVEL_SCHEMES)[number];
export type PlaceKey = 'with' | 'left' | 'right' | 'before' | 'after';

export const LEVEL_WIDTH = 4;
export const ITERATION_MODES = ['together', 'overlap'] as const;
export type IterationMode = (typeof ITERATION_MODES)[number];

export interface SchemeOptions {
  iterations: IterationMode;
  width: number;
}

export const DEFAULT_SCHEME_OPTIONS: SchemeOptions = { iterations: 'together', width: LEVEL_WIDTH };

export interface Place {
  node: Node;
  key: PlaceKey;
  of: Node;
}

export interface SchemeGraph {
  nodes: Node[];
  preds(n: Node): Node[];
  constant(n: Node): boolean;
  scope(n: Node): string | null;
  parent(id: string): string | null;
  pack(n: Node): string | undefined;
  rides(n: Node): boolean;
  late(n: Node): boolean;
  iteration(n: Node): string | undefined;
  places: Place[];
}

type Constraint = { kind: 'equal' | 'above'; low: Node; high: Node };

function constraints(g: SchemeGraph, members: Set<Node>): Constraint[] {
  const out: Constraint[] = [];
  for (const { node, key, of } of g.places) {
    if (!members.has(node) || !members.has(of)) continue;
    if (key === 'after') out.push({ kind: 'above', low: of, high: node });
    else if (key === 'before') out.push({ kind: 'above', low: node, high: of });
    else out.push({ kind: 'equal', low: node, high: of });
  }
  return out;
}

function successors(g: SchemeGraph): Map<Node, Node[]> {
  const succ = new Map<Node, Node[]>(g.nodes.map((n) => [n, []]));
  for (const n of g.nodes) for (const p of g.preds(n)) succ.get(p)?.push(n);
  return succ;
}

function asap(g: SchemeGraph): Map<Node, number> {
  const level = new Map<Node, number>();
  const floor = new Map<Node, number>();
  const rules = constraints(g, new Set(g.nodes));
  for (let round = 0, changed = true; changed && round < g.nodes.length + 1; round++) {
    changed = false;
    for (const n of g.nodes) level.set(n, Math.max(floor.get(n) ?? 0, ...g.preds(n).map((p) => (level.get(p) ?? 0) + 1)));
    for (const { kind, low, high } of rules) {
      const want = kind === 'equal' ? Math.max(level.get(low)!, level.get(high)!) : level.get(low)! + 1;
      for (const n of kind === 'equal' ? [low, high] : [high]) {
        if (level.get(n)! < want) {
          floor.set(n, want);
          changed = true;
        }
      }
    }
  }
  return level;
}

function prepared(g: SchemeGraph): Map<Node, number> {
  const level = asap(g);
  const succ = successors(g);
  const pinned = new Set(constraints(g, new Set(g.nodes)).flatMap((r) => [r.low, r.high]));
  for (const n of [...g.nodes].reverse()) {
    const after = succ.get(n)!;
    if (!g.constant(n) || pinned.has(n) || after.length === 0) continue;
    level.set(n, Math.max(level.get(n)!, Math.min(...after.map((s) => level.get(s)!)) - 1));
  }
  return level;
}

function alap(g: SchemeGraph): Map<Node, number> {
  const early = asap(g);
  const succ = successors(g);
  const level = new Map<Node, number>();
  const ceiling = new Map<Node, number>();
  const rules = constraints(g, new Set(g.nodes));
  const reversed = [...g.nodes].reverse();
  for (let round = 0, changed = true; changed && round < g.nodes.length + 1; round++) {
    changed = false;
    for (const n of reversed) {
      const after = succ.get(n)!.map((s) => level.get(s)! - 1);
      level.set(n, Math.min(ceiling.get(n) ?? Infinity, after.length === 0 ? early.get(n)! : Math.min(...after)));
    }
    for (const { kind, low, high } of rules) {
      const want = kind === 'equal' ? Math.min(level.get(low)!, level.get(high)!) : level.get(high)! - 1;
      for (const n of kind === 'equal' ? [low, high] : [low]) {
        if (level.get(n)! > want) {
          ceiling.set(n, want);
          changed = true;
        }
      }
    }
  }
  const lowest = Math.min(0, ...level.values());
  for (const [n, l] of level) level.set(n, l - lowest);
  return level;
}

interface Item {
  nodes: Node[];
  height: number;
  width: number;
  local: Map<Node, number>;
  module: string | null;
  constant: boolean;
}

function nested(g: SchemeGraph, options: SchemeOptions): Map<Node, number> {
  const members = new Set(g.nodes);
    const chainOf = (n: Node): (string | null)[] => {
    const out: (string | null)[] = [];
    for (let m = g.scope(n); m !== null; m = g.parent(m)) out.unshift(m);
    const iteration = options.iterations === 'together' ? g.iteration(n) : undefined;
    if (iteration !== undefined) out.push(`${out[out.length - 1] ?? ''}#${iteration}`);
    return [null, ...out];
  };
  const chains = new Map(g.nodes.map((n) => [n, chainOf(n)]));
  const rules = constraints(g, members);

  const layout = (scope: string | null, depth: number, nodes: Node[]): Item => {
    const items: Item[] = [];
    const itemOf = new Map<Node, Item>();
    const children = new Map<string, Node[]>();
    for (const n of nodes) {
      const child = chains.get(n)![depth + 1] as string | undefined;
      if (child === undefined) {
        const item: Item = { nodes: [n], height: 1, width: 1, local: new Map([[n, 0]]), module: null, constant: g.constant(n) };
        items.push(item);
        itemOf.set(n, item);
        continue;
      }
      if (!children.has(child)) {
        children.set(child, []);
        items.push({ nodes: [], height: 0, width: 0, local: new Map(), module: child, constant: false });
      }
      children.get(child)!.push(n);
    }
    for (const item of items) {
      if (item.module === null) continue;
      const inner = layout(item.module, depth + 1, children.get(item.module)!);
      Object.assign(item, inner, { module: item.module });
      for (const n of item.nodes) itemOf.set(n, item);
    }
    const preds = new Map<Item, Set<Item>>(items.map((item) => [item, new Set()]));
    const succs = new Map<Item, Set<Item>>(items.map((item) => [item, new Set()]));
    const lead = new Map<Item, Map<Item, number>>(items.map((item) => [item, new Map()]));
    for (const item of items) {
      for (const n of item.nodes) {
        for (const p of g.preds(n)) {
          const from = itemOf.get(p);
          if (from === undefined || from === item) continue;
          preds.get(item)!.add(from);
          succs.get(from)!.add(item);
          const leads = lead.get(item)!;
          leads.set(from, Math.min(leads.get(from) ?? Infinity, item.local.get(n)!));
        }
      }
    }
    const order = topological(items, preds);
    const start = new Map<Item, number>();
    const floor = new Map<Item, number>();
    const own: { kind: 'equal' | 'above'; low: Item; high: Item }[] = rules
      .filter((r) => itemOf.get(r.low)?.module === null && itemOf.get(r.high)?.module === null)
      .map((r) => ({ kind: r.kind, low: itemOf.get(r.low)!, high: itemOf.get(r.high)! }));
    const placed = new Set(own.flatMap((r) => [r.low, r.high]));
    const reach = reachability(order, succs);
    const groups = roleGroups(g, items, placed, reach);
    for (const group of groups) for (const other of group.slice(1)) own.push({ kind: 'equal', low: group[0], high: other });
    for (let round = 0, changed = true; changed && round < items.length + 1; round++) {
      changed = false;
      for (const item of order) start.set(item, Math.max(floor.get(item) ?? 0, ...[...preds.get(item)!].map((p) => start.get(p)! + p.height - lead.get(item)!.get(p)!)));
      for (const { kind, low, high } of own) {
        const want = kind === 'equal' ? Math.max(start.get(low)!, start.get(high)!) : start.get(low)! + 1;
        for (const item of kind === 'equal' ? [low, high] : [high]) {
          if (start.get(item)! < want) {
            floor.set(item, want);
            changed = true;
          }
        }
      }
    }
    const pinned = new Set(own.flatMap((r) => [r.low, r.high]));
    const latest = (item: Item) => Math.min(...[...succs.get(item)!].map((s) => start.get(s)! + lead.get(s)!.get(item)! - item.height));
    for (const item of [...order].reverse()) {
      if (!item.nodes.every((n) => g.late(n)) || pinned.has(item) || succs.get(item)!.size === 0) continue;
      start.set(item, Math.max(start.get(item)!, latest(item)));
    }
    const height = () => Math.max(0, ...items.map((item) => start.get(item)! + item.height));
    const unitOf = new Map<Item, Item[]>(items.map((item) => [item, [item]]));
    for (const group of groups) {
      const unit = [...new Set(group.flatMap((item) => unitOf.get(item)!))];
      for (const item of unit) unitOf.set(item, unit);
    }
    const widthAt = (l: number) => {
      const active = items.filter((item) => start.get(item)! <= l && l < start.get(item)! + item.height);
      return [...new Set(active.map((item) => unitOf.get(item)!))].reduce((sum, unit) => sum + Math.max(...unit.filter((item) => active.includes(item)).map((item) => item.width)), 0);
    };
    for (let l = 0; l < height(); l++) {
      let excess = widthAt(l) - options.width;
      if (excess <= 0) continue;
      const movable = items
        .filter((item) => item.module === null && start.get(item) === l && !pinned.has(item) && succs.get(item)!.size > 0 && latest(item) > l)
        .sort((p, q) => latest(q) - latest(p) || Number(q.constant) - Number(p.constant) || q.nodes[0].id - p.nodes[0].id);
      for (const item of movable) {
        if (excess <= 0) break;
        start.set(item, l + 1);
        excess--;
      }
    }
    const local = new Map<Node, number>();
    for (const item of items) for (const n of item.nodes) local.set(n, start.get(item)! + item.local.get(n)!);
    const used = [...new Set(local.values())].sort((p, q) => p - q);
    const rank = new Map(used.map((l, i) => [l, i]));
    for (const [n, l] of local) local.set(n, rank.get(l)!);
    const counts = new Map<number, number>();
    for (const l of local.values()) counts.set(l, (counts.get(l) ?? 0) + 1);
    return { nodes, height: used.length, width: Math.max(1, ...counts.values()), local, module: scope, constant: nodes.every((n) => g.constant(n)) };
  };

  return layout(null, 0, g.nodes).local;
}

const INDEX = /\[\d+\]|#\d+\.\d+$/g;

function copyKey(item: Item): string {
  return item.module === null ? `node:${item.nodes[0].name.replace(INDEX, '[]')}` : `module:${item.module.replace(INDEX, '[]')}`;
}

function reachability(order: Item[], succs: Map<Item, Set<Item>>): Map<Item, Set<Item>> {
  const reach = new Map<Item, Set<Item>>();
  for (const item of [...order].reverse()) {
    const out = new Set<Item>();
    for (const s of succs.get(item)!) {
      out.add(s);
      for (const r of reach.get(s)!) out.add(r);
    }
    reach.set(item, out);
  }
  return reach;
}

function roleGroups(g: SchemeGraph, items: Item[], placed: Set<Item>, reach: Map<Item, Set<Item>>): Item[][] {
  const keyed = new Map<string, Item[]>();
  const add = (key: string, item: Item) => keyed.set(key, [...(keyed.get(key) ?? []), item]);
  for (const item of items) {
    if (placed.has(item) || item.constant) continue;
    add(copyKey(item), item);
    if (item.module !== null) continue;
    const n = item.nodes[0];
    const computed = g.preds(n).filter((p) => !g.constant(p));
    if (computed.length > 0) add(`siblings:${computed.map((p) => p.id).sort((a, b) => a - b).join()}`, item);
    const pack = g.pack(n);
    if (pack !== undefined) add(`pack:${pack}`, item);
  }
  const waits = communications(g);
  const signature = (item: Item) => [...new Set(item.nodes.flatMap((n) => [...waits(n)]))].sort().join('|');
  const split = [...keyed].flatMap(([key, group]) => {
    if (key.startsWith('pack:')) return [group];
    const bySignature = new Map<string, Item[]>();
    for (const item of group) bySignature.set(signature(item), [...(bySignature.get(signature(item)) ?? []), item]);
    return [...bySignature.values()];
  });
  return split.filter((group) => group.length > 1 && group.every((p) => group.every((q) => !reach.get(p)!.has(q))));
}

function communications(g: SchemeGraph): (n: Node) => Set<string> {
  const memo = new Map<Node, Set<string>>();
  const waits = (n: Node): Set<string> => {
    const known = memo.get(n);
    if (known !== undefined) return known;
    const out = new Set<string>();
    memo.set(n, out);
    if (g.pack(n) !== undefined) out.add(n.name.replace(INDEX, '[]'));
    for (const p of g.preds(n)) for (const c of waits(p)) out.add(c);
    return out;
  };
  return waits;
}

function topological(items: Item[], preds: Map<Item, Set<Item>>): Item[] {
  const out: Item[] = [];
  const seen = new Set<Item>();
  const visit = (item: Item) => {
    if (seen.has(item)) return;
    seen.add(item);
    for (const p of preds.get(item)!) visit(p);
    out.push(item);
  };
  for (const item of items) visit(item);
  return out;
}

function reaches(g: SchemeGraph, from: Node, to: Node): boolean {
  const seen = new Set<Node>();
  const visit = (n: Node): boolean => {
    if (n === from) return true;
    if (seen.has(n)) return false;
    seen.add(n);
    return g.preds(n).some(visit);
  };
  return visit(to);
}

function contract(g: SchemeGraph): SchemeGraph {
  const riders = new Set(g.nodes.filter((n) => g.rides(n)));
  const memo = new Map<Node, Node[]>();
  const preds = (n: Node): Node[] => {
    const known = memo.get(n);
    if (known !== undefined) return known;
    const out = new Set(g.preds(n).flatMap((p) => (riders.has(p) ? preds(p) : [p])));
    out.delete(n);
    memo.set(n, [...out]);
    return memo.get(n)!;
  };
  return { ...g, nodes: g.nodes.filter((n) => !riders.has(n)), preds };
}

export function schemeLevels(scheme: Exclude<LevelScheme, 'lines'>, g: SchemeGraph, options: SchemeOptions = DEFAULT_SCHEME_OPTIONS): Map<Node, number> {
  const kept = contract(g);
  const level = scheme === 'asap' ? prepared(kept) : scheme === 'alap' ? alap(kept) : nested(kept, options);
  for (const { kind, low, high } of constraints(kept, new Set(kept.nodes))) {
    if (kind === 'equal' ? level.get(low) === level.get(high) : level.get(low)! < level.get(high)!) continue;
    const place = g.places.find((p) => [p.node, p.of].includes(low) && [p.node, p.of].includes(high))!;
    const [n, input] = reaches(kept, low, high) ? [high, low] : reaches(kept, high, low) ? [low, high] : [undefined, undefined];
    if (n === undefined) throw new Error(`layout: place '${place.key}' of '${place.node.name}' conflicts with the other places of '${low.name}' and '${high.name}'`);
    throw new Error(`layout: place '${place.key}' of '${place.node.name}' puts '${n.name}' on or below its input '${input.name}'`);
  }
  const succ = successors(g);
  for (const n of [...g.nodes].reverse()) {
    if (level.has(n)) continue;
    const users = succ.get(n)!.map((u) => level.get(u)!);
    level.set(n, new Set(users).size === 1 ? users[0] : Math.min(...users) - 0.5);
  }
  return level;
}
