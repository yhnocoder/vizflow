import type { Graph, Node } from '../autograd';
import { type Positions, positionsOf } from '../explain';

export interface CacheDecl {
  nodes: Node[];
  members: Node[];
  note?: string;
}

export type CacheKind = 'rows' | 'window' | 'state';

export interface CacheReads {
  needed: Map<Node, Set<number>>;
  reads: Map<Node, Set<number>>;
  missing: Node[];
}

export interface CacheAnalysis {
  positions: Positions;
  cached: ReadonlySet<Node>;
  derived: ReadonlySet<Node>;
  horizons(node: Node): (number | undefined)[];
  computes(node: Node, t: number): boolean;
  first(node: Node): number | undefined;
  reads(t: number): CacheReads;
  kind(decl: CacheDecl): CacheKind;
  window(decl: CacheDecl, t: number): number[];
  stored(decl: CacheDecl, positions: number): number;
}

export function analyzeCache(g: Graph, length: number, caches: readonly CacheDecl[], scope: ReadonlySet<Node>, outputs: readonly Node[]): CacheAnalysis {
  const positions = positionsOf(g, length);
  const memo = new Map<Node, (number | undefined)[]>();
  const horizons = (node: Node) => {
    let h = memo.get(node);
    if (h === undefined) memo.set(node, (h = Array.from({ length: node.value.size }, (_, offset) => positions.horizon(node, offset))));
    return h;
  };
  const constant = (node: Node) => horizons(node).every((h) => h === undefined);
  const cached = new Set(caches.flatMap((c) => c.nodes));
  const derived = new Set<Node>();
  for (const node of g.nodes) {
    if (node.kind !== 'op' || cached.has(node)) continue;
    const from = (n: Node) => cached.has(n) || derived.has(n);
    if (node.inputs.some(from) && node.inputs.every((n) => from(n) || constant(n))) derived.add(node);
  }
  const readsMemo = new Map<number, CacheReads>();
  const reads = (t: number): CacheReads => {
    const known = readsMemo.get(t);
    if (known !== undefined) return known;
    const found = new Map<Node, Set<number>>();
    const missing = new Set<Node>();
    const visit = (node: Node, offset: number) => {
      if (!cached.has(node) && !derived.has(node)) {
        missing.add(node);
        return;
      }
      let set = found.get(node);
      if (set === undefined) found.set(node, (set = new Set()));
      if (set.has(offset)) return;
      set.add(offset);
      if (!derived.has(node)) return;
      for (const [input, at] of positions.deps(node, offset)) if (positions.horizon(input, at) !== undefined && input.kind !== 'buffer') visit(input, at);
    };
    const needed = new Map<Node, Set<number>>();
    const queue: [Node, number][] = [];
    const need = (node: Node, offset: number) => {
      let set = needed.get(node);
      if (set === undefined) needed.set(node, (set = new Set()));
      if (set.has(offset)) return;
      set.add(offset);
      queue.push([node, offset]);
    };
    for (const node of [...outputs, ...cached]) {
      if (!scope.has(node) || node.kind !== 'op') continue;
      horizons(node).forEach((h, offset) => h === t && need(node, offset));
    }
    while (queue.length > 0) {
      const [node, offset] = queue.pop() as [Node, number];
      for (const [input, at] of positions.deps(node, offset)) {
        const hi = positions.horizon(input, at);
        if (hi === undefined || input.kind === 'buffer') continue;
        if (hi === t && input.kind === 'op') need(input, at);
        else if (hi < t) visit(input, at);
      }
    }
    const result = { needed, reads: found, missing: [...missing] };
    readsMemo.set(t, result);
    return result;
  };
  const rowsRead = (decl: CacheDecl, t: number) => {
    const read = reads(t).reads;
    return [...new Set(decl.nodes.flatMap((n) => [...(read.get(n) ?? [])].map((offset) => horizons(n)[offset] as number)))].sort((a, b) => a - b);
  };
  return {
    positions,
    cached,
    derived,
    horizons,
    computes: (node, t) => (reads(t).needed.get(node)?.size ?? 0) > 0,
    first: (node) => {
      const defined = horizons(node).filter((h): h is number => h !== undefined);
      return defined.length === 0 ? undefined : Math.min(...defined);
    },
    reads,
    kind: (decl) => {
      if (decl.nodes.every((n) => new Set(horizons(n)).size === 1)) return 'state';
      return rowsRead(decl, length - 1).length < length - 1 ? 'window' : 'rows';
    },
    window: rowsRead,
    stored: (decl, n) => {
      const all = decl.nodes.flatMap((node) => horizons(node).filter((h): h is number => h !== undefined));
      if (decl.nodes.every((node) => new Set(horizons(node)).size === 1)) return all.filter((h) => h === n - 1).length;
      const span = rowsRead(decl, length - 1).length;
      const from = span < length - 1 ? n - span : 0;
      return all.filter((h) => h < n && h >= from).length;
    },
  };
}

export function checkCache(g: Graph, length: number, prompt: number, caches: readonly CacheDecl[], scope: ReadonlySet<Node>, outputs: readonly Node[]): string | undefined {
  const analysis = analyzeCache(g, length, caches, scope, outputs);
  for (let t = prompt; t < length; t++) {
    const { missing } = analysis.reads(t);
    if (missing.length > 0) {
      const names = missing.map((n) => `'${n.name}'`).join(', ');
      return `decoding position ${t + 1} reads earlier positions of ${names}, which no module caches and which cannot be recomputed from the cached tensors; add them, or the tensors they are computed from, to that module's cache:`;
    }
  }
  return undefined;
}
