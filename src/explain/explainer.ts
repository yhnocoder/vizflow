import type { Graph, Node } from '../autograd';
import { indexOf } from '../tensor';
import { type CellRef, type ElemRef, elemTex, isCell, matmulOperands, refKey } from './elem';
import { type FlowEdge, cellValue, contextOf, edgesOf } from './edges';
import { type EdgeCard, type NodeCard, edgeCardOf, nodeCardOf } from './cards';
import { explainDefOf } from './lookup';

export interface Explainer {
  edges(op?: Node): FlowEdge[];
  local(e: FlowEdge): number;
  upstream(e: FlowEdge): number;
  contribution(e: FlowEdge): number;
  zeroGrad(e: FlowEdge): boolean;
  cellValue(c: CellRef): number;
  outEdges(ref: ElemRef): FlowEdge[];
  nodeCard(node: Node, dir: 'forward' | 'backward', drawnFrom?: Node): NodeCard;
  edgeCard(e: FlowEdge): EdgeCard;
  elemTex(ref: ElemRef | CellRef): string;
  edgesVersion(): number;
}

interface Snapshot {
  version: number;
  edgesVersion: number;
  size: number;
  loss: Node | undefined;
  zero: Map<FlowEdge, boolean>;
  edges: FlowEdge[];
  byOp: Map<number, FlowEdge[]>;
  outgoing: Map<string, FlowEdge[]>;
  reachable: Set<string>;
}

class ExplainerImpl implements Explainer {
  private snapshot: Snapshot | undefined;

  constructor(private readonly g: Graph) {}

  edges(op?: Node): FlowEdge[] {
    const snapshot = this.current();
    if (op === undefined) return [...snapshot.edges];
    return [...(snapshot.byOp.get(op.id) ?? [])];
  }

  local(e: FlowEdge): number {
    if (isCell(e.from)) return 1;
    const ctx = contextOf(e.op);
    const out = isCell(e.to) ? matmulOperands(e.op, e.to.cell.index).out : e.to.index;
    return explainDefOf(e.op).local(ctx, out, e.arg, e.from.index);
  }

  upstream(e: FlowEdge): number {
    if (isCell(e.to)) {
      const grad = e.op.grad;
      return grad === null ? 0 : grad.get(matmulOperands(e.op, e.to.cell.index).out);
    }
    const grad = e.to.node.grad;
    return grad === null ? 0 : grad.get(e.to.index);
  }

  contribution(e: FlowEdge): number {
    return this.local(e) * this.upstream(e);
  }

  zeroGrad(e: FlowEdge): boolean {
    const snapshot = this.current();
    let zero = snapshot.zero.get(e);
    if (zero === undefined) {
      zero = !snapshot.reachable.has(refKey(e.to));
      snapshot.zero.set(e, zero);
    }
    return zero;
  }

  cellValue(c: CellRef): number {
    return cellValue(c.cell);
  }

  outEdges(ref: ElemRef): FlowEdge[] {
    return [...(this.current().outgoing.get(refKey(ref)) ?? [])];
  }

  nodeCard(node: Node, dir: 'forward' | 'backward', drawnFrom?: Node): NodeCard {
    return nodeCardOf(this.g, node, dir, (n) => this.reachesLoss(n), drawnFrom);
  }

  edgeCard(e: FlowEdge): EdgeCard {
    return edgeCardOf(this, e);
  }

  elemTex(ref: ElemRef | CellRef): string {
    return elemTex(ref);
  }

  private reachesLoss(node: Node): boolean {
    const reachable = this.current().reachable;
    for (let offset = 0; offset < node.value.size; offset++) {
      if (reachable.has(refKey({ node, index: indexOf(node.shape, offset) }))) return true;
    }
    return false;
  }

  edgesVersion(): number {
    return this.current().edgesVersion;
  }

  private current(): Snapshot {
    const loss = this.g.findLoss();
    const known = this.snapshot;
    if (known !== undefined && known.version === this.g.version && known.size === this.g.nodes.length && known.loss === loss) return known;
    this.snapshot = this.build(loss, known);
    return this.snapshot;
  }

  private build(loss: Node | undefined, previous: Snapshot | undefined): Snapshot {
    const reusable = previous !== undefined && previous.size === this.g.nodes.length;
    let changed = !reusable || previous.loss !== loss;
    const byOp = new Map<number, FlowEdge[]>();
    for (const node of this.g.nodes) {
      if (node.kind !== 'op') continue;
      const def = explainDefOf(node);
      const before = reusable ? previous.byOp.get(node.id) : undefined;
      let own: FlowEdge[];
      if (before !== undefined && def.selected === undefined) own = before;
      else {
        const fresh = edgesOf(node, def);
        own = before !== undefined && sameEdges(before, fresh) ? before : fresh;
        if (own !== before) changed = true;
      }
      byOp.set(node.id, own);
    }
    if (!changed) return { ...previous!, version: this.g.version };
    const edges: FlowEdge[] = [];
    const outgoing = new Map<string, FlowEdge[]>();
    for (const own of byOp.values()) {
      for (const e of own) {
        edges.push(e);
        const key = refKey(e.from);
        const list = outgoing.get(key);
        if (list === undefined) outgoing.set(key, [e]);
        else list.push(e);
      }
    }
    const edgesVersion = (previous?.edgesVersion ?? 0) + 1;
    return { version: this.g.version, edgesVersion, size: this.g.nodes.length, loss, zero: new Map(), edges, byOp, outgoing, reachable: this.reach(byOp, loss) };
  }

  private reach(byOp: Map<number, FlowEdge[]>, loss: Node | undefined): Set<string> {
    const reachable = new Set<string>();
    if (loss === undefined) return reachable;
    for (let offset = 0; offset < loss.value.size; offset++) {
      reachable.add(refKey({ node: loss, index: indexOf(loss.shape, offset) }));
    }
    for (let id = loss.id; id >= 0; id--) {
      const own = byOp.get(id);
      if (own === undefined) continue;
      const ordered = [...own.filter((e) => isCell(e.from)), ...own.filter((e) => !isCell(e.from))];
      for (const e of ordered) {
        if (reachable.has(refKey(e.to))) reachable.add(refKey(e.from));
      }
    }
    return reachable;
  }
}

export function explain(g: Graph): Explainer {
  return new ExplainerImpl(g);
}

function sameEdges(a: readonly FlowEdge[], b: readonly FlowEdge[]): boolean {
  return a.length === b.length && a.every((e, i) => e.id === b[i].id && e.arg === b[i].arg && refKey(e.from) === refKey(b[i].from) && refKey(e.to) === refKey(b[i].to) && e.constant === b[i].constant);
}
