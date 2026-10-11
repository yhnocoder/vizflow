import type { ServingRequest } from './types';

export const BLOCK_SIZE = 16;

export function tokenIds(r: ServingRequest, index: number, length: number): string[] {
  const group = r.group ?? '';
  const shared = group === '' ? 0 : Math.min(r.shared ?? 0, r.prompt);
  return Array.from({ length }, (_, pos) => (pos < shared ? `g:${group}:${pos}` : `r:${index}:${pos}`));
}
