import { Tensor, forEachIndex, normDims } from '../tensor';
import type { Graph } from './graph';
import type { Mesh } from './mesh';
import type { Node } from './node';
import { opDef } from './registry';

export type Placement = { kind: 'replicate' } | { kind: 'shard'; dim: number } | { kind: 'partial' } | { kind: 'local'; rank: number };

export interface GradPlacement {
  placement: Placement;
  reduce: boolean;
}

export interface Placements {
  readonly mesh: Mesh;
  of(node: Node): Placement;
  grad(node: Node): GradPlacement;
  owner(node: Node, index: readonly number[], dir?: 'forward' | 'backward'): number | undefined;
  ranks(node: Node): Tensor[] | undefined;
  gradRanks(node: Node): Tensor[] | undefined;
}

export class PlacementError extends Error {
  override name = 'PlacementError';
}

export const REPLICATE: Placement = Object.freeze({ kind: 'replicate' });
const PARTIAL: Placement = Object.freeze({ kind: 'partial' });

const LINEAR_UNARY = new Set(['identity', 'neg']);
const NONLINEAR_UNARY = new Set(['exp', 'log', 'rsqrt', 'sqrt', 'sigmoid', 'silu', 'softplus', 'relu', 'gelu', 'pow']);
const MOVEMENT = new Set(['transpose', 'view', 'select', 'narrow', 'cat', 'repeat_interleave', 'rotate_half']);
const SUMS = new Set(['sum', 'mean']);
const ROW_WISE = new Set(['softmax', 'log_softmax', 'logsumexp', 'max']);
const KEEP_DIM = new Set(['softmax', 'log_softmax']);

export function placementText(p: Placement): string {
  if (p.kind === 'replicate') return 'Replicate';
  if (p.kind === 'partial') return 'Partial';
  if (p.kind === 'shard') return `Shard(${p.dim})`;
  return `只在 rank ${p.rank} 上`;
}

function same(a: Placement, b: Placement): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'shard') return a.dim === (b as { dim: number }).dim;
  if (a.kind === 'local') return a.rank === (b as { rank: number }).rank;
  return true;
}

class PlacementsImpl implements Placements {
  private readonly forward = new Map<Node, Placement>();
  private readonly cache = new Map<string, Tensor[] | undefined>();
  private cacheVersion = -1;

  constructor(
    private readonly g: Graph,
    readonly mesh: Mesh,
    private readonly leaves: ReadonlyMap<Node, Placement>,
  ) {}

  of(node: Node): Placement {
    const known = this.forward.get(node);
    if (known !== undefined) return known;
    const placement = node.kind === 'op' ? this.rule(node, node.inputs.map((input) => this.of(input))) : (this.leaves.get(node) ?? REPLICATE);
    this.forward.set(node, placement);
    return placement;
  }

  grad(node: Node): GradPlacement {
    const forward = this.of(node);
    if (forward.kind === 'partial') return { placement: REPLICATE, reduce: false };
    if (forward.kind !== 'replicate') return { placement: forward, reduce: false };
    const reduce = this.shardedUses(node).length > 0;
    return { placement: reduce ? PARTIAL : REPLICATE, reduce };
  }

  owner(node: Node, index: readonly number[], dir: 'forward' | 'backward' = 'forward'): number | undefined {
    return this.ownerAt(node, dir === 'forward' ? this.of(node) : this.grad(node).placement, index);
  }

  private ownerAt(node: Node, p: Placement, index: readonly number[]): number | undefined {
    if (p.kind === 'local') return p.rank;
    if (p.kind === 'shard') return Math.floor(index[p.dim] / (node.shape[p.dim] / this.mesh.size));
    return undefined;
  }

  ranks(node: Node): Tensor[] | undefined {
    if (this.of(node).kind !== 'partial') return undefined;
    return this.cached(`f${node.id}`, () => {
      const def = opDef(node.op as string)!;
      return this.rankList((r) => def.forward(node.inputs.map((input) => this.localView(input, r)), node.attrs));
    });
  }

  gradRanks(node: Node): Tensor[] | undefined {
    if (!this.grad(node).reduce) return undefined;
    return this.cached(`b${node.id}`, () => {
      const parts = this.rankList(() => Tensor.zeros(node.shape));
      for (const use of this.shardedUses(node)) {
        const def = opDef(use.op as string)!;
        const gradOut = use.grad;
        if (gradOut === null) continue;
        for (let r = 0; r < this.mesh.size; r++) {
          const partial = this.of(use).kind === 'partial';
          const upstream = partial ? gradOut : this.masked(use, gradOut, this.of(use), r);
          const inputs = use.inputs.map((input) => (partial && this.of(input).kind === 'partial' ? this.ranks(input)![r] : input.value));
          const grads = def.backward(upstream, inputs, use.value, use.attrs);
          use.inputs.forEach((input, arg) => {
            const contribution = grads[arg];
            if (input !== node || contribution === null || contribution === undefined) return;
            for (let i = 0; i < contribution.size; i++) parts[r].data[i] += contribution.data[i];
          });
        }
      }
      return parts;
    });
  }

  private shardedUses(node: Node): Node[] {
    if (node.grad === null) return [];
    return this.g.nodes.filter((use) => {
      if (use.kind !== 'op' || use.op === 'all_reduce' || use.grad === null) return false;
      const flags = opDef(use.op as string)?.differentiable;
      if (!use.inputs.some((input, arg) => input === node && (flags === undefined || flags[arg]))) return false;
      return this.of(use).kind !== 'replicate';
    });
  }

  private rankList(make: (rank: number) => Tensor): Tensor[] {
    return Array.from({ length: this.mesh.size }, (_, r) => make(r));
  }

  private cached(key: string, make: () => Tensor[]): Tensor[] {
    if (this.cacheVersion !== this.g.version) {
      this.cache.clear();
      this.cacheVersion = this.g.version;
    }
    if (!this.cache.has(key)) this.cache.set(key, make());
    return this.cache.get(key)!;
  }

  private localView(node: Node, rank: number): Tensor {
    const p = this.of(node);
    if (p.kind === 'partial') return this.ranks(node)![rank];
    return this.masked(node, node.value, p, rank);
  }

  private masked(node: Node, values: Tensor, p: Placement, rank: number): Tensor {
    if (p.kind === 'replicate' || p.kind === 'partial') return values;
    const out = Tensor.zeros(values.shape);
    forEachIndex(values.shape, (index, offset) => {
      if (this.owner(node, index) === rank) out.data[offset] = values.data[offset];
    });
    return out;
  }

  private rule(node: Node, ins: Placement[]): Placement {
    const op = node.op as string;
    if (op === 'all_reduce') return this.allReduce(node, ins[0]);
    if (ins.every((p) => p.kind === 'replicate')) return REPLICATE;
    if (LINEAR_UNARY.has(op)) return ins[0];
    if (NONLINEAR_UNARY.has(op)) {
      if (ins[0].kind === 'partial') throw this.error(node, 'Partial 不能直接进入非线性运算，要先 all_reduce');
      return ins[0];
    }
    if (op === 'add' || op === 'sub' || op === 'mul' || op === 'div' || op === 'where') return this.elementwise(node, ins);
    if (MOVEMENT.has(op)) return this.movement(node, node.inputs, ins, (tensors) => opDef(op)!.forward(tensors, node.attrs));
    if (op === 'matmul') return this.matmul(node, ins[0], ins[1]);
    if (op === 'select_matmul') {
      const index = node.attrs.index as number;
      const head = this.movement(node, [node.inputs[0]], [ins[0]], ([t]) => opDef('select')!.forward([t], { dim: 1, index }));
      return this.matmul(node, head, ins[1], 2);
    }
    if (SUMS.has(op) || ROW_WISE.has(op)) return this.reduction(node, ins[0]);
    throw this.error(node, `运算 ${op} 没有放置规则，它的输入只能是 Replicate`);
  }

  private error(node: Node, message: string): PlacementError {
    return new PlacementError(`节点 '${node.name}'（${node.op}）：${message}`);
  }

  private allReduce(node: Node, input: Placement): Placement {
    this.checkGroup(node);
    if (input.kind !== 'partial') throw this.error(node, `all_reduce 的输入必须是 Partial，这里是 ${placementText(input)}`);
    return REPLICATE;
  }

  private checkGroup(node: Node): void {
    const group = node.attrs.group;
    if (group !== this.mesh.axis) throw this.error(node, `通信组 '${String(group)}' 不是 mesh 的轴，mesh 只有 '${this.mesh.axis}'`);
  }

  private elementwise(node: Node, ins: Placement[]): Placement {
    const op = node.op as string;
    const partial = ins.map((p) => p.kind === 'partial');
    if (partial.some(Boolean)) {
      if (ins.length === 1) {
        if (op === 'mul' || op === 'div') return PARTIAL;
        throw this.error(node, 'Partial 加减一个常数不合法：每张卡都会加一次');
      }
      if ((op === 'add' || op === 'sub') && partial.every(Boolean)) return PARTIAL;
      if (op === 'mul' && ins.some((p) => p.kind === 'replicate')) return PARTIAL;
      if (op === 'div' && partial[0] && ins[1].kind === 'replicate') return PARTIAL;
      if (op === 'add' || op === 'sub') throw this.error(node, 'Partial 与非 Partial 相加不合法，要先 all_reduce');
      throw this.error(node, `Partial 不能和 ${ins.map(placementText).join('、')} 这样组合`);
    }
    const out = node.shape.length;
    const shifted = ins.map((p, i): Placement => (p.kind === 'shard' ? { kind: 'shard', dim: p.dim + out - node.inputs[i].shape.length } : p));
    const placed = shifted.filter((p) => p.kind !== 'replicate');
    if (!placed.every((p) => same(p, placed[0]))) throw this.error(node, `输入的放置 ${ins.map(placementText).join('、')} 不一致`);
    return placed[0];
  }

  private ownerTensor(node: Node, p: Placement): Tensor {
    const t = Tensor.zeros(node.shape);
    forEachIndex(node.shape, (index, offset) => {
      t.data[offset] = (this.ownerAt(node, p, index) ?? -1) + 1;
    });
    return t;
  }

  private movement(node: Node, inputs: readonly Node[], ins: Placement[], forward: (tensors: Tensor[]) => Tensor): Placement {
    if (ins.some((p) => p.kind === 'partial')) {
      if (ins.every((p) => p.kind === 'partial')) return PARTIAL;
      throw this.error(node, 'Partial 不能和其他放置拼在一起');
    }
    const owners = forward(inputs.map((input, i) => this.ownerTensor(input, ins[i])));
    const values = Array.from(owners.data, (v) => Math.round(Math.abs(v)));
    if (values.every((v) => v === 0)) return REPLICATE;
    if (values.some((v) => v === 0)) throw this.error(node, 'Replicate 和切分后的张量不能拼在一起');
    if (values.every((v) => v === values[0])) return { kind: 'local', rank: values[0] - 1 };
    const shape = owners.shape;
    for (let dim = 0; dim < shape.length; dim++) {
      if (shape[dim] % this.mesh.size !== 0) continue;
      const chunk = shape[dim] / this.mesh.size;
      let fits = true;
      forEachIndex(shape, (index, offset) => {
        if (fits && values[offset] !== Math.floor(index[dim] / chunk) + 1) fits = false;
      });
      if (fits) return { kind: 'shard', dim };
    }
    throw this.error(node, `切分的边界没有落在结果 (${shape.join(', ')}) 的某一维的等分点上，例如按 head 切时 head 数要能被 ${this.mesh.size} 整除`);
  }

  private matmul(node: Node, a: Placement, b: Placement, aDims = node.inputs[0].shape.length): Placement {
    if (aDims !== 2 || node.inputs[1].shape.length !== 2) {
      if (a.kind === 'replicate' && b.kind === 'replicate') return REPLICATE;
      throw this.error(node, '只有二维矩阵乘支持切分');
    }
    if (a.kind === 'partial' || b.kind === 'partial') {
      if (a.kind === 'replicate' || b.kind === 'replicate') return PARTIAL;
      throw this.error(node, 'Partial 只能乘 Replicate');
    }
    if (a.kind === 'replicate' && b.kind === 'replicate') return REPLICATE;
    if (a.kind === 'local' || b.kind === 'local') {
      const ranks = [a, b].filter((p): p is { kind: 'local'; rank: number } => p.kind === 'local').map((p) => p.rank);
      if (ranks.every((r) => r === ranks[0]) && [a, b].every((p) => p.kind === 'local' || p.kind === 'replicate')) return { kind: 'local', rank: ranks[0] };
      throw this.error(node, `两个输入 ${placementText(a)}、${placementText(b)} 不在同一张卡上`);
    }
    if (a.kind === 'replicate' && b.kind === 'shard' && b.dim === 1) return { kind: 'shard', dim: 1 };
    if (a.kind === 'shard' && a.dim === 0 && b.kind === 'replicate') return { kind: 'shard', dim: 0 };
    if (a.kind === 'shard' && a.dim === 1 && b.kind === 'shard' && b.dim === 0) return PARTIAL;
    throw this.error(node, `矩阵乘不支持 ${placementText(a)} × ${placementText(b)}`);
  }

  private reduction(node: Node, input: Placement): Placement {
    const op = node.op as string;
    const ndim = node.inputs[0].shape.length;
    const attrs = node.attrs as { dim?: number | number[] | null; keepdim?: boolean };
    const dims = normDims(attrs.dim ?? null, ndim);
    const keepdim = KEEP_DIM.has(op) || attrs.keepdim === true;
    if (input.kind === 'partial') {
      if (SUMS.has(op)) return PARTIAL;
      throw this.error(node, 'Partial 不能直接进入非线性运算，要先 all_reduce');
    }
    if (input.kind === 'local') return input;
    if (input.kind !== 'shard') return REPLICATE;
    if (dims.includes(input.dim)) {
      if (SUMS.has(op)) return PARTIAL;
      throw this.error(node, `沿被切分的第 ${input.dim} 维做 ${op} 需要先通信`);
    }
    return { kind: 'shard', dim: keepdim ? input.dim : input.dim - dims.filter((d) => d < input.dim).length };
  }
}

export function placements(g: Graph, mesh: Mesh, leaves: ReadonlyMap<Node, Placement>): Placements {
  return new PlacementsImpl(g, mesh, leaves);
}
