import type { Placements } from '../autograd';

export type ParallelMode = 'train' | 'prefill' | 'decode';

export interface SceneParallel {
  placements: Placements;
  infer: boolean;
}

export const MODE_TOKENS: Readonly<Record<ParallelMode, number>> = { train: 4096, prefill: 4096, decode: 64 };
