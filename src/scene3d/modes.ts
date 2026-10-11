import type { Node, Placements } from '../autograd';
import type { StepInfo } from '../steps';

export type ParallelMode = 'train' | 'prefill' | 'decode';

export interface SceneParallel {
  placements: Placements;
  infer: boolean;
}

export const MODE_TOKENS: Readonly<Record<ParallelMode, number>> = { train: 4096, prefill: 4096, decode: 64 };

export const MODE_TOKEN_TEXT: Readonly<Record<ParallelMode, string>> = {
  train: '训练序列的长度',
  prefill: '提示的长度',
  decode: '同时 decode 的序列数，每条序列这一步只有 1 个新 token',
};

export const RING_NOTE = '按所有卡组成一个扁平的环计算；跨节点时 NCCL 实际可能选树形或分层（先节点内、再节点间）的算法，通信量会不同';

export function stepMode(infer: boolean, info: StepInfo | undefined): ParallelMode {
  if (!infer) return 'train';
  return info?.segment?.startsWith('decode') === true ? 'decode' : 'prefill';
}

export function byteSize(bytes: number): { value: string; unit: string } {
  const units = ['B', 'KiB', 'MiB', 'GiB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  const digits = Number.isInteger(value) ? 0 : 1;
  return { value: value.toFixed(digits), unit: units[unit] };
}

function lastReceiving(steps: readonly StepInfo[], node: Node): StepInfo | undefined {
  let found: StepInfo | undefined;
  for (const info of steps) if (info.dir === 'backward' && info.kind !== 'overview' && info.nodes.includes(node)) found = info;
  return found;
}

export function gradReduces(placements: Placements, steps: readonly StepInfo[], info: StepInfo): Node[] {
  if (info.dir !== 'backward' || info.kind === 'overview') return [];
  return info.nodes.filter((node) => placements.grad(node).reduce && lastReceiving(steps, node) === info);
}
