import type { Graph, Node } from '../autograd';
import { indexOf } from '../tensor';
import { type CellRef, type ElemRef, elemTex, isCell, matmulOperands, refKey } from './elem';
import { type FlowEdge, cellValue, contextOf, edgesOf } from './edges';
import { type EdgeCard, type NodeCard, edgeCardOf, nodeCardOf } from './cards';
import { explainDefOf, lossOf } from './lookup';

export interface Explainer {
  edges(op?: Node): FlowEdge[];
  local(e: FlowEdge): number;
  upstream(e: FlowEdge): number;
  contribution(e: FlowEdge): number;
  zeroGrad(e: FlowEdge): boolean;
  cellValue(c: CellRef): number;
  outEdges(ref: ElemRef): FlowEdge[];
  nodeCard(node: Node, dir: 'forward' | 'backward'): NodeCard;
  edgeCard(e: FlowEdge): EdgeCard;
  elemTex(ref: ElemRef | CellRef): string;
}

interface Snapshot {
  key: string;
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
    return !this.current().reachable.has(refKey(e.to));
  }

  cellValue(c: CellRef): number {
    return cellValue(c.cell);
  }

  outEdges(ref: ElemRef): FlowEdge[] {
    return [...(this.current().outgoing.get(refKey(ref)) ?? [])];
  }

  nodeCard(node: Node, dir: 'forward' | 'backward'): NodeCard {
    return nodeCardOf(this.g, node, dir, (n) => this.reachesLoss(n));
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

  private current(): Snapshot {
    const loss = lossOf(this.g);
    const key = `${this.g.version}:${this.g.nodes.length}:${loss?.id ?? -1}`;
    if (this.snapshot?.key !== key) this.snapshot = this.build(key, loss);
    return this.snapshot;
  }

  private build(key: string, loss: Node | undefined): Snapshot {
    const edges: FlowEdge[] = [];
    const byOp = new Map<number, FlowEdge[]>();
    const outgoing = new Map<string, FlowEdge[]>();
    for (const node of this.g.nodes) {
      if (node.kind !== 'op') continue;
      const own = edgesOf(node, explainDefOf(node));
      byOp.set(node.id, own);
      for (const e of own) {
        edges.push(e);
        const key = refKey(e.from);
        const list = outgoing.get(key);
        if (list === undefined) outgoing.set(key, [e]);
        else list.push(e);
      }
    }
    return { key, edges, byOp, outgoing, reachable: this.reach(byOp, loss) };
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
