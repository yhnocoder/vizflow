import type { ServingRequest } from './types';

export const BLOCK_SIZE = 16;

export interface Sequence {
  id: string;
  shared: number;
  group: string;
}

export function sequenceOf(r: ServingRequest, index: number): Sequence {
  const group = r.group ?? '';
  return { id: r.id, shared: group === '' ? 0 : Math.min(r.shared ?? 0, r.prompt), group: group === '' ? `#${index}` : group };
}

export function tokenKey(s: Sequence, pos: number): string {
  return pos < s.shared ? `g:${s.group}:${pos}` : `r:${s.id}:${pos}`;
}

export function blockKey(s: Sequence, block: number): string {
  return (block + 1) * BLOCK_SIZE <= s.shared ? `g:${s.group}:b${block}` : `r:${s.id}:b${block}`;
}
