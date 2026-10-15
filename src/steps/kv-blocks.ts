import type { PagedRequestState, PagedScenario, PagedStep, Token } from '../sim';
import type { InferSpec } from './steps';

export const PAGE_REQUEST = 'A';

export interface KvBlocksRequest {
  id: string;
  prompt: string[];
  arrive: number;
}

export interface KvBlocksSpec {
  infer: InferSpec;
  blockSize: number;
  numBlocks: number;
  others: KvBlocksRequest[];
}

export interface KvSlotSource {
  request: string;
  position: number;
}

export function kvTokenText(token: Token | null): string {
  if (token === null) return '';
  return typeof token === 'string' && /^[A-Z]\d+$/.test(token) ? '·' : String(token);
}

export function pageTokens(infer: InferSpec): string[] {
  const vocab = infer.label.tokens ?? infer.input.tokens ?? [];
  const ids = [...Array.from(infer.input.value.data.subarray(0, 1)), ...Array.from(infer.label.value.data)].map(Math.round);
  return ids.map((id) => vocab[id] ?? String(id));
}

export function kvScenario(spec: KvBlocksSpec): PagedScenario {
  const { infer, blockSize, numBlocks, others } = spec;
  const tokens = pageTokens(infer);
  const requests = [{ id: PAGE_REQUEST, prompt: tokens.slice(0, infer.prompt), outputs: tokens.slice(infer.prompt), arrive: 0 }, ...others.map((o) => ({ id: o.id, prompt: o.prompt, outputs: [], arrive: o.arrive }))];
  return { requests, blockSize, numBlocks, steps: tokens.length - infer.prompt + 1 + others.length };
}

export function engineStepAt(steps: readonly PagedStep[], position: number): number {
  const at = steps.findIndex((s) => (s.requests.find((r) => r.id === PAGE_REQUEST)?.computed ?? 0) >= position + 1);
  return at < 0 ? steps.length - 1 : at;
}

export function slotOf(table: readonly number[], position: number, blockSize: number): { block: number; offset: number } {
  return { block: table[Math.floor(position / blockSize)], offset: position % blockSize };
}

export function slotSources(step: PagedStep, blockSize: number): (KvSlotSource | null)[][] {
  const sources = (step.blocks ?? []).map(() => Array.from({ length: blockSize }, (): KvSlotSource | null => null));
  for (const r of step.requests) {
    for (let position = 0; position < r.computed; position++) {
      const { block, offset } = slotOf(r.table, position, blockSize);
      sources[block][offset] ??= { request: r.id, position };
    }
  }
  return sources;
}

export function writtenPositions(r: PagedRequestState): [number, number] {
  return [r.computed - r.scheduled, r.computed];
}

export function touchedPositions(previous: PagedStep | undefined, r: PagedRequestState): [number, number] {
  const arrived = !(previous?.requests.some((q) => q.id === r.id) ?? false);
  return [arrived ? 0 : r.computed - r.scheduled, r.computed];
}
