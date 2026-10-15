import { DTYPE_BYTES, type Graph, type Node, type OpCost, opDef, realSizeOf } from '../autograd';
import type { StepInfo } from '../steps';

export type MemoryKind = 'param' | 'input' | 'buffer' | 'saved' | 'temp' | 'param_grad' | 'grad';

export const MEMORY_KINDS: readonly MemoryKind[] = ['param', 'input', 'buffer', 'saved', 'temp', 'param_grad', 'grad'];

export interface Scale {
  batch: number;
  copies?: { drawn: number; real: number };
}

export interface MemoryEntry {
  node: Node;
  kind: MemoryKind;
  shape: readonly number[];
  bytes: number;
  scale: Scale;
}

export interface FlopEntry {
  node: Node;
  dir: 'forward' | 'backward';
  terms: number[][];
  flops: number;
  scale: Scale;
}

export interface StepCost {
  overview: boolean;
  memory: MemoryEntry[];
  bytes: number;
  ops: FlopEntry[];
  flops: number;
  forward: number;
  backward: number;
}

export interface CostTimeline {
  steps: StepCost[];
  peak: number;
  peakStep: number;
  forward: number;
  backward: number;
  uncounted: Node[];
}

export interface CostOptions {
  real?: boolean;
  dims?: Record<string, number>;
  batch?: number;
}

const LEAF_KINDS: Readonly<Record<string, MemoryKind>> = { param: 'param', input: 'input', label: 'input', buffer: 'buffer' };

const product = (factors: readonly number[]) => factors.reduce((a, b) => a * b, 1);

export function scaleFactor(scale: Scale): number {
  return scale.batch * (scale.copies === undefined ? 1 : scale.copies.real / scale.copies.drawn);
}

class CostModel {
  readonly nodes: readonly Node[];
  readonly loss: Node | undefined;
  readonly root: number[] = [];
  readonly consumers: Node[][];
  readonly holds: { from: number; until: number }[][];
  readonly gradNodes = new Set<number>();
  readonly gradFrom: number[][];
  readonly needsGrad: boolean[];
  readonly real: boolean;
  readonly batch: number;
  readonly shapes: readonly (readonly number[])[];
  readonly sizes: Readonly<Record<string, number>>;
  readonly batched: boolean[] = [];

  constructor(
    readonly g: Graph,
    options: CostOptions,
  ) {
    this.real = options.real === true && g.real !== undefined;
    this.batch = this.real ? (options.batch ?? 1) : 1;
    this.sizes = this.real ? g.realSizes(options.dims ?? {}) : g.dims;
    this.shapes = this.real ? g.realShapes(options.dims ?? {}) : g.nodes.map((n) => n.shape);
    this.nodes = g.nodes;
    this.loss = g.findLoss();
    const loss = this.loss;
    this.consumers = this.nodes.map(() => []);
    this.holds = this.nodes.map(() => []);
    this.needsGrad = this.nodes.map((n) => loss !== undefined && n.id <= loss.id && n.grad !== null);
    for (const n of this.nodes) {
      this.batched[n.id] = n.kind === 'input' || n.kind === 'label' || n.inputs.some((input) => this.batched[input.id]);
      this.root[n.id] = this.isView(n) || this.cost(n)?.writes === true ? this.root[n.inputs[0].id] : n.id;
      n.inputs.forEach((input) => this.consumers[input.id].push(n));
      for (const saved of this.saved(n)) this.holds[saved.id].push({ from: n.id, until: n.id });
    }
    this.gradFrom = this.nodes.map(() => []);
    for (const spec of g.functions) {
      const anchor = Math.max(...spec.outputs.map((n) => n.id));
      for (const n of spec.backward) if (n.op === 'saved') this.holds[n.inputs[0].id].push({ from: anchor, until: n.id });
      for (const { input, grad } of spec.gradInputs) {
        this.gradFrom[input.id].push(grad.id);
        this.gradNodes.add(grad.id);
      }
    }
    for (let id = this.nodes.length - 1; id >= 0; id--) {
      if (!this.needsGrad[id]) continue;
      for (const c of this.consumers[id]) {
        if (!this.needsGrad[c.id] || !c.inputs.some((input, k) => input.id === id && this.differentiable(c, k))) continue;
        this.gradFrom[id].push(...(this.isView(c) ? this.gradFrom[c.id] : [c.id]));
      }
    }
  }

  cost(n: Node): OpCost | undefined {
    return n.kind === 'op' ? opDef(n.op as string)?.cost : undefined;
  }

  isView(n: Node): boolean {
    return this.cost(n)?.view === true;
  }

  differentiable(c: Node, k: number): boolean {
    return opDef(c.op as string)?.differentiable?.[k] ?? true;
  }

  gradInputs(c: Node): number[] {
    if (!this.needsGrad[c.id]) return [];
    return c.inputs.flatMap((input, k) => (this.differentiable(c, k) && this.needsGrad[input.id] ? [k] : []));
  }

  saved(c: Node): Node[] {
    if (c.kind !== 'op') return [];
    const cost = this.cost(c);
    const picks = this.gradInputs(c).flatMap((k) => (cost === undefined ? c.inputs.map((_, i) => i) : (cost.saves[k] ?? [])));
    return [...new Set(picks.flatMap((pick) => (pick === 'out' ? [c] : c.inputs[pick] === undefined ? [] : [c.inputs[pick]])))];
  }

  shape(n: Node): readonly number[] {
    return this.shapes[n.id];
  }

  scale(n: Node): Scale {
    const scale: Scale = { batch: this.batched[n.id] ? this.batch : 1 };
    const { copies } = this.cost(n)?.writes === true ? this.nodes[this.root[n.id]] : n;
    const real = this.real ? product(copies.real.map((d) => realSizeOf(d, this.sizes, this.shapes) as number)) : copies.drawn;
    if (real !== copies.drawn) scale.copies = { drawn: copies.drawn, real };
    return scale;
  }

  entry(n: Node, kind: MemoryKind): MemoryEntry {
    const scale = this.scale(n);
    const shape = this.real && this.batched[n.id] ? [this.batch, ...this.shape(n)] : this.shape(n);
    return { node: n, kind, shape, bytes: product(this.shape(n)) * DTYPE_BYTES[n.dtype] * scaleFactor(scale), scale };
  }

  forwardMemory(computed: Uint8Array): MemoryEntry[] {
    const kinds = new Map<number, MemoryKind>();
    for (const n of this.nodes) {
      const leaf = LEAF_KINDS[n.kind];
      if (leaf !== undefined) {
        kinds.set(n.id, leaf);
        continue;
      }
      if (!computed[n.id] || !this.computes(n) || this.g.onChip(n)) continue;
      const root = this.root[n.id];
      const saved = this.holds[n.id].some((hold) => computed[hold.from]);
      const pending = this.consumers[n.id].length === 0 || this.consumers[n.id].some((c) => !computed[c.id]);
      if (saved) kinds.set(root, 'saved');
      else if (pending && !kinds.has(root)) kinds.set(root, 'temp');
    }
    return [...kinds].map(([id, kind]) => this.entry(this.nodes[id], kind));
  }

  backwardMemory(processed: Uint8Array): MemoryEntry[] {
    const kinds = new Map<number, MemoryKind>();
    const grads: MemoryEntry[] = [];
    for (const n of this.nodes) {
      const leaf = LEAF_KINDS[n.kind];
      if (leaf !== undefined) kinds.set(n.id, leaf);
      else if (n.phase === 'backward') this.backwardTemp(n, processed, kinds);
      else if (this.computes(n) && !this.g.onChip(n) && this.holds[n.id].some((hold) => !processed[hold.until])) kinds.set(this.root[n.id], 'saved');
      if (!this.needsGrad[n.id] || this.isView(n) || n === this.loss) continue;
      const arrived = this.gradFrom[n.id].some((c) => processed[c]);
      if (arrived && (n.kind !== 'op' || !processed[n.id])) grads.push(this.entry(n, n.kind === 'param' ? 'param_grad' : 'grad'));
    }
    return [...[...kinds].map(([id, kind]) => this.entry(this.nodes[id], kind)), ...grads];
  }

  private backwardTemp(n: Node, processed: Uint8Array, kinds: Map<number, MemoryKind>): void {
    const root = this.root[n.id];
    if (!processed[n.id] || this.g.onChip(n) || this.nodes[root].phase === 'forward' || this.gradNodes.has(root) || kinds.has(root)) return;
    if (this.consumers[n.id].some((c) => !processed[c.id])) kinds.set(root, 'temp');
  }

  closure(targets: readonly Node[], computed: Uint8Array, phase: Node['phase'] = 'forward'): Node[] {
    const fresh: Node[] = [];
    const visit = (n: Node) => {
      if (n.kind !== 'op' || n.phase !== phase || computed[n.id]) return;
      computed[n.id] = 1;
      n.inputs.forEach(visit);
      fresh.push(n);
    };
    targets.filter((n) => this.computes(n)).forEach(visit);
    return fresh.sort((a, b) => a.id - b.id);
  }

  forwardOps(ops: readonly Node[], uncounted: Set<Node>, dir: FlopEntry['dir'] = 'forward'): FlopEntry[] {
    const out: FlopEntry[] = [];
    for (const n of ops) {
      const cost = this.cost(n);
      if (cost === undefined) uncounted.add(n);
      const terms = cost === undefined ? [] : [cost.flops(n.inputs.map((i) => [...this.shape(i)]), [...this.shape(n)]).forward];
      this.push(out, n, dir, terms);
    }
    return out;
  }

  backwardOps(ops: readonly Node[], uncounted: Set<Node>): FlopEntry[] {
    const out: FlopEntry[] = [];
    for (const n of [...ops].sort((a, b) => b.id - a.id)) {
      const ks = this.gradInputs(n);
      if (ks.length === 0) continue;
      const cost = this.cost(n);
      if (cost === undefined) uncounted.add(n);
      const backward = cost?.flops(n.inputs.map((i) => [...this.shape(i)]), [...this.shape(n)]).backward ?? [];
      this.push(out, n, 'backward', ks.flatMap((k) => (backward[k] === undefined ? [] : [backward[k]])));
    }
    return out;
  }

  private push(out: FlopEntry[], node: Node, dir: FlopEntry['dir'], terms: number[][]): void {
    const counted = terms.filter((t) => product(t) > 0);
    const scale = this.scale(node);
    const flops = counted.reduce((sum, t) => sum + product(t), 0) * scaleFactor(scale);
    if (flops > 0) out.push({ node, dir, terms: counted, flops, scale });
  }

  computes(n: Node): boolean {
    return n.kind === 'op' && LEAF_KINDS[this.nodes[this.root[n.id]].kind] === undefined;
  }

  startOfGrad(n: Node): number {
    return n.kind === 'op' ? n.id + 1 : Math.min(...this.gradFrom[n.id]);
  }
}

export function costTimeline(g: Graph, list: readonly StepInfo[], options: CostOptions = {}): CostTimeline {
  const model = new CostModel(g, options);
  const uncounted = new Set<Node>();
  const computed = new Uint8Array(g.nodes.length);
  const processed = new Uint8Array(g.nodes.length);
  const batches: { id: string; ops: Node[] }[] = [];
  let forward = 0;
  let backward = 0;
  const backwardBatch = (info: StepInfo): Node[] => {
    const at = batches.findIndex((batch) => batch.id === info.forwardId);
    if (at >= 0) return batches.splice(at).flatMap((batch) => batch.ops.filter((n) => !processed[n.id]));
    const start = Math.min(...info.nodes.map((n) => model.startOfGrad(n)));
    return g.nodes.filter((n) => n.kind === 'op' && n.id >= start && !processed[n.id]);
  };
  const lastBackward = list.reduce((last, info, i) => (info.dir === 'backward' && info.kind !== 'overview' ? i : last), -1);
  const steps = list.map((info, i): StepCost => {
    if (info.kind === 'overview') return { overview: true, memory: [], bytes: 0, ops: [], flops: 0, forward: 0, backward: 0 };
    let memory: MemoryEntry[];
    let ops: FlopEntry[];
    if (info.dir === 'backward' && info.computes !== undefined) {
      const fresh = model.closure(info.computes, processed, 'backward');
      ops = model.forwardOps(fresh, uncounted, 'backward');
      memory = model.backwardMemory(processed);
    } else if (info.dir === 'forward') {
      const fresh = model.closure(info.nodes, computed);
      batches.push({ id: info.id, ops: fresh });
      ops = model.forwardOps(fresh, uncounted);
      memory = model.forwardMemory(computed);
    } else {
      const batch = i === lastBackward ? g.nodes.filter((n) => n.kind === 'op' && !processed[n.id]) : backwardBatch(info);
      batch.forEach((n) => (processed[n.id] = 1));
      ops = model.backwardOps(batch, uncounted);
      memory = model.backwardMemory(processed);
    }
    const flops = ops.reduce((sum, op) => sum + op.flops, 0);
    if (info.dir === 'forward') forward += flops;
    else backward += flops;
    return { overview: false, memory, bytes: memory.reduce((sum, m) => sum + m.bytes, 0), ops, flops, forward, backward };
  });
  let peakStep = -1;
  steps.forEach((step, i) => {
    if (!step.overview && (peakStep < 0 || step.bytes > steps[peakStep].bytes)) peakStep = i;
  });
  const peak = peakStep < 0 ? 0 : steps[peakStep].bytes;
  for (const step of steps) {
    if (!step.overview || peakStep < 0) continue;
    Object.assign(step, { memory: steps[peakStep].memory, bytes: peak, forward, backward });
  }
  return { steps, peak, peakStep, forward, backward, uncounted: [...uncounted] };
}
