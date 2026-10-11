import { DTYPE_BITS, type Node, linkKind, messageBytes, placementText } from '../autograd';
import type { CardRow } from '../explain';
import type { StepInfo, StepNote } from '../steps';
import type { Tensor } from '../tensor';
import type { LinearRgb, ThemeColors } from '../theme';
import type { SceneController } from './controller';
import { DECODE_LATENCY_NOTE, MODE_TOKENS, MODE_TOKEN_TEXT, RING_NOTE, type ParallelMode, byteSize, gradReduces, stepMode } from './modes';

export const PARTIAL_LEGEND_TEXT = '细柱 = 各卡的部分值，虚线框 = 和';
export const ALL_REDUCE_NOTE = 'all_reduce：两张卡的部分值相加，结果回到每张卡';
export const NO_COMM_NOTE = 'all_reduce 的反向不通信';

export function parallelMode(c: SceneController): ParallelMode {
  const steps = c.steps;
  return stepMode(c.options.parallel?.infer === true, steps?.list[steps.current]);
}

export function rankColor(theme: ThemeColors, rank: number): LinearRgb {
  return rank % 2 === 0 ? theme.rank0 : theme.rank1;
}

export function elementRank(c: SceneController, node: Node, index: readonly number[]): number | undefined {
  const parallel = c.options.parallel;
  if (parallel === undefined) return undefined;
  if (c.state.dir === 'backward' && node.grad === null) return undefined;
  return parallel.placements.owner(node, index, c.state.dir);
}

export function partialParts(c: SceneController, node: Node): Tensor[] | undefined {
  const placements = c.options.parallel?.placements;
  if (placements === undefined) return undefined;
  if (c.state.dir === 'forward') return placements.ranks(node);
  return c.stepNodes.has(node) ? placements.gradRanks(node) : undefined;
}

function bytesText(bytes: number): string {
  const { value, unit } = byteSize(bytes);
  return `${value}\\,\\mathrm{${unit}}`;
}

function sizeRows(node: Node, mode: ParallelMode, T: number, S: number): CardRow[] {
  const bytes = DTYPE_BITS[node.dtype] / 8;
  if (node.kind === 'param') {
    const shape = node.realShape.join('\\times ');
    return [{ label: '消息大小', value: `参数的梯度，与 token 数无关：$S=${shape}\\times ${bytes}\\,\\mathrm{B}=${bytesText(S)}$` }];
  }
  const width = node.realShape.slice(1).reduce((a, b) => a * b, 1);
  return [
    { label: 'token 数', value: `$T=${T}$，${MODE_TOKEN_TEXT[mode]}` },
    { label: '消息大小', value: `$S=T\\times h\\times ${bytes}\\,\\mathrm{B}=${T}\\times ${width}\\times ${bytes}\\,\\mathrm{B}=${bytesText(S)}$` },
  ];
}

function commRows(c: SceneController, node: Node, title: string): CardRow[] {
  const { mesh } = c.options.parallel!.placements;
  const mode = parallelMode(c);
  const T = MODE_TOKENS[mode];
  const n = mesh.real.size;
  const S = messageBytes(node, T);
  const ring = (2 * (n - 1) * S) / n;
  return [
    { label: '通信', value: title },
    { label: '通信组', value: `${mesh.axis}，$n=${n}$ 张卡（这一页画 ${mesh.size} 张）` },
    { label: '链路', value: linkText(mesh.real) },
    ...sizeRows(node, mode, T, S),
    { label: '每卡发送', value: `扁平环形算法 $2\\frac{n-1}{n}S=${bytesText(ring)}$` },
    { label: '算法假设', value: RING_NOTE },
    ...(mode === 'decode' ? [{ label: '耗时', value: DECODE_LATENCY_NOTE }] : []),
  ];
}

function linkText(topology: { size: number; nodes: number }): string {
  if (topology.nodes === 1) return `NVLink，${topology.size} 张卡在同一节点`;
  const crossing = linkKind(topology, 0, topology.size - 1) === 'rdma';
  return `${topology.nodes} 个节点，每个节点 ${topology.size / topology.nodes} 张卡；节点内 NVLink${crossing ? '，节点之间 RDMA' : ''}`;
}

export function parallelRows(c: SceneController, node: Node): CardRow[] {
  const parallel = c.options.parallel;
  if (parallel === undefined) return [];
  const placements = parallel.placements;
  if (c.state.dir === 'forward') {
    const rows: CardRow[] = [{ label: '放置', value: placementText(placements.of(node)) }];
    if (node.op === 'all_reduce') rows.push(...commRows(c, node, '前向 all_reduce'));
    return rows;
  }
  if (node.grad === null) return [];
  const grad = placements.grad(node);
  const rows: CardRow[] = [{ label: '梯度放置', value: grad.reduce ? 'Partial，all_reduce 之后是 Replicate' : placementText(grad.placement) }];
  if (grad.reduce) rows.push(...commRows(c, node, '反向 all_reduce'));
  if (node.op === 'all_reduce') rows.push({ label: '通信', value: '反向不通信，梯度原样交给每张卡上的部分和' });
  return rows;
}

export function parallelNotes(c: SceneController, info: StepInfo | undefined): StepNote[] {
  const parallel = c.options.parallel;
  const steps = c.steps;
  if (parallel === undefined || info === undefined || steps === undefined || info.kind === 'overview') return [];
  const placements = parallel.placements;
  if (info.dir === 'forward') {
    return info.nodes.filter((node) => node.op === 'all_reduce').map((node) => ({ nodes: [node], text: ALL_REDUCE_NOTE }));
  }
  const notes: StepNote[] = gradReduces(placements, steps.list, info).map((node) => ({ nodes: [node], text: `$\\partial ${node.tex}$ 是 Partial，反向做 all_reduce` }));
  for (const node of info.nodes) {
    if (c.g.nodes.some((use) => use.op === 'all_reduce' && use.inputs[0] === node)) notes.push({ nodes: [node], text: NO_COMM_NOTE });
  }
  return notes;
}

export function parallelLegend(): string {
  return `<span><i class="vf-sw" style="background:var(--rank0)"></i>rank 0</span><span><i class="vf-sw" style="background:var(--rank1)"></i>rank 1</span><span>${PARTIAL_LEGEND_TEXT}</span>`;
}
