import { chainBlockHashes, type Token } from './hash';
import { KvCacheManager, type KvRequest } from './kv-cache-manager';
import { RadixCacheSim, type RadixSnapshot } from './radix-cache';

export type PagedEngine = 'vllm' | 'sglang';

export interface PagedRequestSpec {
  id: string;
  prompt: readonly Token[];
  outputs: readonly Token[];
  arrive: number;
}

export interface PagedScenario {
  requests: readonly PagedRequestSpec[];
  blockSize: number;
  numBlocks: number;
  steps: number;
}

export interface PagedRequestState {
  id: string;
  tokens: Token[];
  computed: number;
  hit: number;
  scheduled: number;
  table: number[];
}

export interface VllmBlockState {
  id: number;
  refCnt: number;
  hashed: boolean;
  tokens: (Token | null)[];
  owners: string[];
}

export interface PagedStep {
  step: number;
  scheduled: { id: string; tokens: number; hit: number }[];
  requests: PagedRequestState[];
  blocks?: VllmBlockState[];
  free?: number[];
  slots?: { owner: string | null; token: Token | null; cached: boolean }[];
  tree?: RadixSnapshot;
}

interface Live {
  spec: PagedRequestSpec;
  tokens: Token[];
  computed: number;
  hit: number;
  admitted: boolean;
  scheduled: number;
}

function liveRequests(scenario: PagedScenario): Live[] {
  return scenario.requests.map((spec) => ({ spec, tokens: [...spec.prompt], computed: 0, hit: 0, admitted: false, scheduled: 0 }));
}

function sample(r: Live): void {
  const k = r.tokens.length - r.spec.prompt.length;
  r.tokens.push(r.spec.outputs[k] ?? `${r.spec.id}${k + 1}`);
}

function kvRequest(r: Live, blockSize: number): KvRequest {
  return {
    id: r.spec.id,
    numTokens: r.tokens.length,
    numPromptTokens: r.spec.prompt.length,
    numComputedTokens: r.computed,
    sharedPrefixBoundary: 0,
    blockHashes: () => chainBlockHashes(r.tokens, blockSize, undefined),
  };
}

export function simulateVllmPages(scenario: PagedScenario): PagedStep[] {
  const { blockSize, numBlocks } = scenario;
  const manager = new KvCacheManager({ numGpuBlocks: numBlocks, groups: [{ kind: 'full', blockSize }], maxModelLen: 1 << 20, maxInFlightTokens: 1 << 20, enablePrefixCaching: true }, () => {});
  const reqs = liveRequests(scenario);
  const out: PagedStep[] = [];
  for (let step = 0; step < scenario.steps; step++) {
    const scheduled: PagedStep['scheduled'] = [];
    for (const r of reqs) r.scheduled = 0;
    for (const r of reqs.filter((q) => q.admitted)) {
      if (manager.allocateSlots(kvRequest(r, blockSize), 1, null, false) === null) throw new Error(`no free block for ${r.spec.id} at step ${step}`);
      r.scheduled = 1;
      scheduled.push({ id: r.spec.id, tokens: 1, hit: 0 });
    }
    for (const r of reqs.filter((q) => !q.admitted && q.spec.arrive === step)) {
      const request = kvRequest(r, blockSize);
      const hit = manager.getComputedBlocks(request);
      const n = r.tokens.length - hit.numTokens;
      if (manager.allocateSlots(request, n, hit, true) === null) throw new Error(`no free block for ${r.spec.id} at step ${step}`);
      r.admitted = true;
      r.hit = hit.numTokens;
      r.computed = hit.numTokens;
      r.scheduled = n;
      scheduled.push({ id: r.spec.id, tokens: n, hit: hit.numTokens });
    }
    for (const r of reqs) {
      if (r.scheduled === 0) continue;
      r.computed += r.scheduled;
      if (r.computed === r.tokens.length) sample(r);
    }
    out.push(vllmSnapshot(step, scheduled, reqs, manager, blockSize));
  }
  return out;
}

function vllmSnapshot(step: number, scheduled: PagedStep['scheduled'], reqs: readonly Live[], manager: KvCacheManager, blockSize: number): PagedStep {
  const pool = manager.pool;
  const blocks: VllmBlockState[] = pool.blocks.map((b) => ({ id: b.id, refCnt: b.refCnt, hashed: b.primaryHash !== null, tokens: Array.from({ length: blockSize }, () => null), owners: [] }));
  const requests = reqs
    .filter((r) => r.admitted)
    .map((r) => {
      const table = manager.getBlocks(r.spec.id)[0].map((b) => b.id);
      table.forEach((id, k) => {
        const block = blocks[id];
        block.owners.push(r.spec.id);
        for (let j = 0; j < blockSize; j++) {
          const t = k * blockSize + j;
          if (t < r.computed && block.tokens[j] === null) block.tokens[j] = r.tokens[t];
        }
      });
      return { id: r.spec.id, tokens: [...r.tokens], computed: r.computed, hit: r.hit, scheduled: r.scheduled, table };
    });
  return { step, scheduled, requests, blocks, free: [...pool.freeQueue] };
}

export function simulateSglangPages(scenario: PagedScenario): PagedStep[] {
  const numSlots = scenario.numBlocks * scenario.blockSize;
  const cache = new RadixCacheSim(numSlots);
  const reqs = liveRequests(scenario);
  const out: PagedStep[] = [];
  const decoding = new Set<Live>();
  for (let step = 0; step < scenario.steps; step++) {
    const scheduled: PagedStep['scheduled'] = [];
    for (const r of reqs) r.scheduled = 0;
    const arriving = reqs.filter((q) => !q.admitted && q.spec.arrive <= step);
    if (arriving.length > 0) {
      for (const r of arriving) {
        const match = cache.matchPrefix(r.tokens.slice(0, r.tokens.length - 1));
        cache.admit(r.spec.id, r.tokens, r.spec.prompt.length, match);
        const n = r.tokens.length - match.prefixLength;
        cache.extend(r.spec.id, n);
        r.admitted = true;
        r.hit = match.prefixLength;
        r.computed = match.prefixLength + n;
        r.scheduled = n;
        scheduled.push({ id: r.spec.id, tokens: n, hit: match.prefixLength });
      }
      for (const r of arriving) cache.checkpoint(r.spec.id, Math.min(r.computed, r.spec.prompt.length));
    } else {
      for (const r of decoding) {
        cache.extend(r.spec.id, 1);
        r.computed += 1;
        r.scheduled = 1;
        scheduled.push({ id: r.spec.id, tokens: 1, hit: 0 });
      }
    }
    for (const r of reqs) {
      if (r.scheduled === 0) continue;
      sample(r);
      decoding.add(r);
    }
    out.push(sglangSnapshot(step, scheduled, reqs, cache));
  }
  return out;
}

function sglangSnapshot(step: number, scheduled: PagedStep['scheduled'], reqs: readonly Live[], cache: RadixCacheSim): PagedStep {
  const tree = cache.snapshot();
  const slots: NonNullable<PagedStep['slots']> = Array.from({ length: cache.numSlots + 1 }, () => ({ owner: null, token: null, cached: false }));
  for (const node of tree.nodes) node.slots.forEach((slot, k) => (slots[slot] = { owner: null, token: node.key[k], cached: true }));
  const requests = tree.requests.map((view) => {
    const r = reqs.find((q) => q.spec.id === view.id)!;
    view.slots.forEach((slot, t) => {
      if (t < r.computed && !slots[slot].cached) slots[slot] = { owner: view.id, token: r.tokens[t], cached: false };
    });
    return { id: view.id, tokens: [...r.tokens], computed: r.computed, hit: r.hit, scheduled: r.scheduled, table: [...view.slots] };
  });
  return { step, scheduled, requests, slots, tree };
}

export function simulatePages(engine: PagedEngine, scenario: PagedScenario): PagedStep[] {
  return engine === 'vllm' ? simulateVllmPages(scenario) : simulateSglangPages(scenario);
}
