import { DTYPE_BITS, type Node, commTransfers, isParamLike, linkKind, messageBytes, placementText } from '../autograd';
import type { CardRow } from '../explain';
import type { StepInfo, StepNote } from '../steps';
import type { Tensor } from '../tensor';
import type { LinearRgb, ThemeColors } from '../theme';
import type { SceneController } from './controller';
import { DECODE_LATENCY_NOTE, MODE_TOKENS, MODE_TOKEN_TEXT, RING_NOTE, type ParallelMode, byteSize, gradReduces, stepMode, stepSends } from './modes';

export const PARTIAL_LEGEND_TEXT = '细柱 = 各卡的部分值，虚线框 = 和';
export const IDLE_LEGEND_TEXT = '灰色 = 这一轮没有卡计算';
export const IDLE_TEXT = '这一轮没有卡计算这个格子';
export const ALL_REDUCE_NOTE = 'all_reduce：两张卡的部分值相加，结果回到每张卡';
export const NO_COMM_NOTE = 'all_reduce 的反向不通信';
export const SEND_RECV_NOTE = 'send_recv：每张卡把手里的块发给下一张卡，同时从上一张卡收一块';
export const SEND_RECV_DECODE_TEXT = 'decode 每条序列每步只有 1 个新 token，实际部署不对 decode 做 CP';

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

export function elementIdle(c: SceneController, node: Node, index: readonly number[]): boolean {
  const parallel = c.options.parallel;
  if (parallel === undefined || (c.state.dir === 'backward' && node.grad === null)) return false;
  const placements = parallel.placements;
  const placement = c.state.dir === 'forward' ? placements.of(node) : placements.grad(node).placement;
  return placement.kind === 'block' && placements.owner(node, index, c.state.dir) === undefined;
}

export function partialParts(c: SceneController, node: Node): Tensor[] | undefined {
  const placements = c.options.parallel?.placements;
  if (placements === undefined) return undefined;
  if (c.state.dir === 'forward') return placements.ranks(node);
  return c.stepNodes.has(node) ? placements.gradRanks(node) : undefined;
}

export function bytesText(bytes: number): string {
  const { value, unit } = byteSize(bytes);
  return `${value}\\,\\mathrm{${unit}}`;
}

function sizeRows(node: Node, mode: ParallelMode, T: number, S: number): CardRow[] {
  const bytes = DTYPE_BITS[node.dtype] / 8;
  if (isParamLike(node)) {
    const shape = node.realShape.join('\\times ');
    return [{ label: '消息大小', value: `参数的梯度，与 token 数无关：$S=${shape}\\times ${bytes}\\,\\mathrm{B}=${bytesText(S)}$` }];
  }
  if (node.realShape.length === 0) return [{ label: '消息大小', value: `一个标量，与 token 数无关：$S=${bytesText(S)}$` }];
  const width = node.realShape.slice(1).reduce((a, b) => a * b, 1);
  return [
    { label: 'token 数', value: `$T=${T}$，${MODE_TOKEN_TEXT[mode]}` },
    { label: '消息大小', value: `$S=T\\times h\\times ${bytes}\\,\\mathrm{B}=${T}\\times ${width}\\times ${bytes}\\,\\mathrm{B}=${bytesText(S)}$` },
  ];
}

function groupRows(c: SceneController, title: string): CardRow[] {
  const { mesh } = c.options.parallel!.placements;
  return [
    { label: '通信', value: title },
    { label: '通信组', value: `${mesh.axis}，$n=${mesh.real.size}$ 张卡（这一页画 ${mesh.size} 张）` },
    { label: '链路', value: linkText(mesh.real) },
  ];
}

function sendRecvRows(c: SceneController, node: Node, title: string): CardRow[] {
  const placements = c.options.parallel!.placements;
  const mode = parallelMode(c);
  if (mode === 'decode') return [...groupRows(c, title), { label: '说明', value: SEND_RECV_DECODE_TEXT }];
  const T = MODE_TOKENS[mode];
  const n = placements.mesh.real.size;
  const width = node.realShape.slice(1).reduce((a, b) => a * b, 1);
  const bytes = DTYPE_BITS[node.dtype] / 8;
  const sent = commTransfers(placements, node, { tokens: T, dir: c.state.dir })[0].bytes;
  return [
    ...groupRows(c, title),
    { label: 'token 数', value: `$T=${T}$，${MODE_TOKEN_TEXT[mode]}；每张卡 $T/n=${T / n}$ 个` },
    { label: '每卡每轮发送', value: `$\\frac{T}{n}\\times h\\times ${bytes}\\,\\mathrm{B}=${T / n}\\times ${width}\\times ${bytes}\\,\\mathrm{B}=${bytesText(sent)}$` },
    { label: '轮数', value: `$n-1=${n - 1}$ 轮，每一轮的传输和这一轮的注意力计算同时进行` },
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
    ...groupRows(c, title),
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
    if (node.op === 'send_recv') rows.push(...sendRecvRows(c, node, '前向 send_recv：rank $r$ 发给 rank $r+1$'));
    return rows;
  }
  if (node.grad === null) return [];
  const grad = placements.grad(node);
  const rows: CardRow[] = [{ label: '梯度放置', value: grad.reduce ? 'Partial，all_reduce 之后是 Replicate' : placementText(grad.placement) }];
  if (grad.reduce) rows.push(...commRows(c, node, '反向 all_reduce'));
  if (node.op === 'all_reduce') rows.push({ label: '通信', value: '反向不通信，梯度原样交给每张卡上的部分和' });
  if (node.op === 'send_recv') rows.push(...sendRecvRows(c, node, '反向 send_recv：梯度沿环送回，rank $r$ 发给 rank $r-1$'));
  return rows;
}

export function parallelNotes(c: SceneController, info: StepInfo | undefined): StepNote[] {
  const parallel = c.options.parallel;
  const steps = c.steps;
  if (parallel === undefined || info === undefined || steps === undefined || info.kind === 'overview') return [];
  const placements = parallel.placements;
  const sends = stepSends(steps.list, info);
  if (info.dir === 'forward') {
    const notes: StepNote[] = info.nodes.filter((node) => node.op === 'all_reduce').map((node) => ({ nodes: [node], text: ALL_REDUCE_NOTE }));
    if (sends.length > 0) notes.push({ nodes: sends, text: SEND_RECV_NOTE });
    return notes;
  }
  const notes: StepNote[] = gradReduces(placements, steps.list, info).map((node) => ({ nodes: [node], text: `$\\partial ${node.tex}$ 是 Partial，反向做 all_reduce` }));
  if (sends.length > 0) {
    const grads = (nodes: Node[]) => nodes.map((node) => `$\\partial ${node.tex}$`).join('、');
    const inputs = sends.map((node) => node.inputs[0]);
    notes.push({ nodes: inputs, text: `send_recv 的反向：${grads(sends)} 沿环送回上一张卡，加到那张卡的 ${grads(inputs)} 上；整个环走完，等价于一次 reduce_scatter` });
  }
  for (const node of info.nodes) {
    if (c.g.nodes.some((use) => use.op === 'all_reduce' && use.inputs[0] === node)) notes.push({ nodes: [node], text: NO_COMM_NOTE });
  }
  return notes;
}

export const PLATFORM_PARTIAL_TEXT = '虚线框 = 这张卡的部分和';

export function platformLegend(): string {
  return `<span><i class="vf-sw" style="background:var(--nvlink)"></i>NVLink</span><span><i class="vf-sw" style="background:var(--rdma)"></i>RDMA</span><span>${PLATFORM_PARTIAL_TEXT}</span>`;
}

export function parallelLegend(idle: boolean): string {
  const idleText = idle ? `<span><i class="vf-sw" style="background:var(--line-strong)"></i>${IDLE_LEGEND_TEXT}</span>` : '';
  return `<span><i class="vf-sw" style="background:var(--rank0)"></i>rank 0</span><span><i class="vf-sw" style="background:var(--rank1)"></i>rank 1</span>${idleText}<span>${PARTIAL_LEGEND_TEXT}</span>`;
}
