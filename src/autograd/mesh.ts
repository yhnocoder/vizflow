import { type EpMode, type EpTopology, type Transfer, type Via, combinePlan, dispatchPlan } from '../sim/deepep';
import { DTYPE_BITS } from './dtype';
import type { Node } from './node';
import type { Placements } from './placement';

export interface MeshTopology {
  size: number;
  nodes: number;
}

export interface Mesh extends MeshTopology {
  axis: string;
  real: MeshTopology;
}

export interface Device {
  node: number;
  gpu: number;
}

export type CommPhase = 'reduce_scatter' | 'all_gather';

export interface CommTransfer {
  src: number;
  dst: number;
  bytes: number;
  link: Via;
  hop: number;
  phase?: CommPhase;
  plan?: Transfer;
}

export interface CommOptions {
  tokens?: number;
  mode?: EpMode;
  dir?: 'forward' | 'backward';
}

export function gpusPerNode(topology: MeshTopology): number {
  return topology.size / topology.nodes;
}

export function rankDevice(topology: MeshTopology, rank: number): Device {
  const perNode = gpusPerNode(topology);
  return { node: Math.floor(rank / perNode), gpu: rank % perNode };
}

export function linkKind(topology: MeshTopology, a: number, b: number): Via {
  if (a === b) return 'local';
  return rankDevice(topology, a).node === rankDevice(topology, b).node ? 'nvlink' : 'rdma';
}

export function epTopology(topology: MeshTopology, numExperts: number): EpTopology {
  return { numScaleoutRanks: topology.nodes, numScaleupRanks: gpusPerNode(topology), numExperts };
}

export function ringAllReduce(topology: MeshTopology, bytes: number): CommTransfer[] {
  const n = topology.size;
  const share = ((n - 1) * bytes) / n;
  const phases: CommPhase[] = ['reduce_scatter', 'all_gather'];
  return phases.flatMap((phase) =>
    Array.from({ length: n }, (_, src) => {
      const dst = (src + 1) % n;
      return { src, dst, bytes: share, link: linkKind(topology, src, dst), hop: 0, phase };
    }),
  );
}

export function ringSendRecv(topology: MeshTopology, bytes: number, dir: 'forward' | 'backward' = 'forward'): CommTransfer[] {
  const n = topology.size;
  return Array.from({ length: n }, (_, src) => {
    const dst = (src + (dir === 'forward' ? 1 : n - 1)) % n;
    return { src, dst, bytes, link: linkKind(topology, src, dst), hop: 0 };
  });
}

export function isParamLike(node: Node): boolean {
  return node.kind === 'param' || (node.op === 'transpose' && node.inputs[0].kind === 'param');
}

export function messageBytes(node: Node, tokens: number): number {
  const bytes = DTYPE_BITS[node.dtype] / 8;
  if (isParamLike(node) || node.realShape.length === 0) return node.realShape.reduce((a, b) => a * b, 1) * bytes;
  const width = node.realShape.slice(1).reduce((a, b) => a * b, 1);
  return tokens * width * bytes;
}

function topkIdx(placements: Placements, x: Node, sel: Node): number[][][] {
  const [T, E] = sel.shape;
  const perRank: number[][][] = Array.from({ length: placements.mesh.size }, () => []);
  for (let t = 0; t < T; t++) {
    const rank = placements.owner(x, [t, 0]);
    if (rank === undefined) throw new Error(`commTransfers: '${x.name}' must be split by token over the mesh`);
    const experts: number[] = [];
    for (let e = 0; e < E; e++) if (sel.value.data[t * E + e] === 1) experts.push(e);
    perRank[rank].push(experts);
  }
  return perRank;
}

function fromPlan(plan: Transfer[]): CommTransfer[] {
  return plan.map((t) => ({ src: t.srcRank, dst: t.dstRank, bytes: t.bytes, link: t.via, hop: t.hop === 'send' ? 0 : 1, plan: t }));
}

export function commTransfers(placements: Placements, node: Node, options: CommOptions = {}): CommTransfer[] {
  const mesh = placements.mesh;
  const mode = options.mode ?? 'hybrid';
  if (node.op === 'all_reduce') {
    if (options.tokens === undefined) throw new Error(`commTransfers: all_reduce '${node.name}' needs the number of tokens`);
    return ringAllReduce(mesh, messageBytes(node, options.tokens));
  }
  if (node.op === 'send_recv') {
    if (options.tokens === undefined) throw new Error(`commTransfers: send_recv '${node.name}' needs the number of tokens`);
    return ringSendRecv(mesh, messageBytes(node, options.tokens / mesh.real.size), options.dir);
  }
  if (node.op === 'dispatch') {
    const [x, sel] = node.inputs;
    const plan = dispatchPlan(epTopology(mesh, sel.shape[1]), topkIdx(placements, x, sel), { mode, hidden: x.realShape[1], dtype: node.attrs.fp8 === true ? 'fp8' : 'bf16' });
    return fromPlan(plan);
  }
  if (node.op === 'combine') {
    const sel = node.inputs[1];
    const source = placements.of(node);
    if (source.kind !== 'shard' || source.dim !== 0) throw new Error(`commTransfers: combine '${node.name}' must return tokens split over the mesh`);
    const plan = combinePlan(epTopology(mesh, sel.shape[1]), topkIdx(placements, node, sel), { mode, allowMultipleReduction: true, hidden: node.realShape[1] });
    return fromPlan(plan);
  }
  throw new Error(`commTransfers: '${node.name}' is not a communication op`);
}
