import { describe, expect, it } from 'vitest';
import { chainBlockHashes, KvCacheManager, type CacheHit, type KvGroupSpec, type KvRequest, type Token, type VllmEvent } from '../../../src/sim/serve';

const range = (start: number, n: number) => Array.from({ length: n }, (_, i) => start + i);

interface Spec {
  id: string;
  prompt: Token[];
  maxTokens: number;
  arrival: number;
  cacheSalt?: string;
}

interface Config {
  numGpuBlocks: number;
  groups: KvGroupSpec[];
  maxNumBatchedTokens: number;
  maxModelLen: number;
  enablePrefixCaching?: boolean;
}

class TestRequest implements KvRequest {
  numComputedTokens = 0;
  sharedPrefixBoundary = 0;
  hit: number | null = null;
  readonly tokens: Token[];
  private hashes: string[] = [];

  constructor(readonly spec: Spec, private readonly hashBlockSize: number) {
    this.tokens = [...spec.prompt];
  }

  get id() {
    return this.spec.id;
  }

  get numTokens() {
    return this.tokens.length;
  }

  get numPromptTokens() {
    return this.spec.prompt.length;
  }

  blockHashes() {
    this.hashes = chainBlockHashes(this.tokens, this.hashBlockSize, this.spec.cacheSalt, this.hashes);
    return this.hashes;
  }
}

interface StepRecord {
  events: VllmEvent[];
  hits: Map<string, CacheHit>;
  tables: Map<string, number[][]>;
  freeQueue: number[];
}

function drive(config: Config, specs: Spec[]) {
  const events: VllmEvent[] = [];
  const kv = new KvCacheManager({
    numGpuBlocks: config.numGpuBlocks, groups: config.groups, maxModelLen: config.maxModelLen,
    maxInFlightTokens: config.maxNumBatchedTokens, enablePrefixCaching: config.enablePrefixCaching ?? true,
  }, (e) => events.push(e));
  const requests = specs.map((s) => new TestRequest(s, kv.hashBlockSize));
  const waiting: TestRequest[] = [];
  let running: TestRequest[] = [];
  const steps: StepRecord[] = [];
  for (let step = 0; requests.some((r) => r.numTokens - r.numPromptTokens < r.spec.maxTokens); step++) {
    events.length = 0;
    const hits = new Map<string, CacheHit>();
    for (const r of requests) if (r.spec.arrival === step) waiting.push(r);
    const scheduled: [TestRequest, number][] = [];
    let budget = config.maxNumBatchedTokens;
    for (const r of running) {
      const n = Math.min(r.numTokens - r.numComputedTokens, budget);
      if (kv.allocateSlots(r, n, null, false) === null) throw new Error(`no room for ${r.id}`);
      scheduled.push([r, n]);
      budget -= n;
    }
    while (budget > 0 && waiting.length > 0) {
      const r = waiting[0];
      const hit = kv.getComputedBlocks(r);
      r.sharedPrefixBoundary = hit.sharedPrefixBoundary;
      const n = Math.min(r.numTokens - hit.numTokens, budget);
      if (kv.allocateSlots(r, n, hit, true) === null) break;
      waiting.shift();
      running.push(r);
      r.numComputedTokens = hit.numTokens;
      r.hit = hit.numTokens;
      hits.set(r.id, hit);
      scheduled.push([r, n]);
      budget -= n;
    }
    for (const [r, n] of scheduled) r.numComputedTokens += n;
    for (const [r] of scheduled) {
      if (r.numComputedTokens < r.numTokens) continue;
      r.tokens.push(1_000_000 + r.numTokens);
      if (r.numTokens - r.numPromptTokens < r.spec.maxTokens) continue;
      kv.free(r.id, 'finish');
      running = running.filter((x) => x !== r);
    }
    const tables = new Map(requests.map((r) => [r.id, kv.getBlocks(r.id).map((bs) => bs.map((b) => b.id))]));
    steps.push({ events: [...events], hits, tables, freeQueue: [...kv.pool.freeQueue] });
  }
  return { kv, steps, requests };
}

function eventsOf<K extends VllmEvent['kind']>(s: StepRecord, kind: K): Extract<VllmEvent, { kind: K }>[] {
  return s.events.filter((e): e is Extract<VllmEvent, { kind: K }> => e.kind === kind);
}

const full4: Config = { numGpuBlocks: 16, groups: [{ kind: 'full', blockSize: 4 }], maxNumBatchedTokens: 64, maxModelLen: 128 };

describe('vLLM block pool and prefix cache (vllm f26d9f4)', () => {
  const { kv, steps } = drive(full4, [
    { id: 'A', prompt: range(1, 10), maxTokens: 3, arrival: 0 },
    { id: 'B', prompt: [...range(1, 8), 50, 51, 52, 53], maxTokens: 2, arrival: 1 },
  ]);

  it('block 0 is the null block and allocation starts at block 1 (block_pool.py:187)', () => {
    expect(kv.pool.blocks[0].isNull).toBe(true);
    expect(steps[0].tables.get('A')).toEqual([[1, 2, 3]]);
    expect(steps[0].freeQueue[0]).toBe(4);
  });

  it('only full blocks are hashed, chained over the prefix (kv_cache_utils.py:695, kv_cache_manager.py:612)', () => {
    expect(eventsOf(steps[0], 'hash').map((e) => [e.blockId, e.numTokens])).toEqual([[1, 4], [2, 8]]);
  });

  it('a shared prompt hits two blocks without touching the free queue (kv_cache_manager.py:295, block_pool.py:752)', () => {
    const hit = steps[1].hits.get('B')!;
    expect(hit.numTokens).toBe(8);
    expect(hit.blocks[0].map((b) => b.id)).toEqual([1, 2]);
    expect(steps[1].tables.get('B')).toEqual([[1, 2, 4]]);
    expect(eventsOf(steps[1], 'touch')).toEqual([]);
  });

  it('a block becomes hashed in the step that fills it: decode token 12 fills block 3', () => {
    expect(eventsOf(steps[2], 'hash').map((e) => e.blockId)).toEqual([3]);
  });

  it('blocks are freed in reverse, hashed ones to the tail and unhashed ones to the head (block_pool.py:774 free_blocks)', () => {
    expect(eventsOf(steps[2], 'free').map((f) => [f.req, f.blockIds, f.toHead, f.toTail])).toEqual([
      ['A', [3, 2, 1], [], [3]],
      ['B', [5, 4, 2, 1], [5], [4, 2, 1]],
    ]);
    expect(steps[2].freeQueue).toEqual([5, ...range(6, 10), 3, 4, 2, 1]);
  });
});

describe('reuse order and eviction on reallocation (vllm f26d9f4)', () => {
  const { steps } = drive({ ...full4, numGpuBlocks: 7 }, [
    { id: 'A', prompt: range(1, 8), maxTokens: 1, arrival: 0 },
    { id: 'C', prompt: range(30, 20), maxTokens: 1, arrival: 1 },
    { id: 'D', prompt: range(1, 8), maxTokens: 1, arrival: 2 },
  ]);

  it('allocation pops the head of the free queue and a reused block loses its hash (block_pool.py:666, :729)', () => {
    expect(steps[0].freeQueue).toEqual([3, 4, 5, 6, 2, 1]);
    expect(eventsOf(steps[1], 'allocate')[0].blockIds).toEqual([3, 4, 5, 6, 2]);
    expect(eventsOf(steps[1], 'evict').map((e) => [e.blockId, e.reason])).toEqual([[2, 'reuse']]);
  });

  it('the hit is capped at num_tokens - 1, so an 8-token prompt cached in 2 blocks hits 1 block (kv_cache_manager.py:295)', () => {
    expect(steps[2].hits.get('D')!.numTokens).toBe(4);
    expect(eventsOf(steps[2], 'touch')).toEqual([{ kind: 'touch', req: 'D', group: 0, blockIds: [1] }]);
    expect(eventsOf(steps[2], 'allocate')[0].blockIds).toEqual([2]);
    expect(eventsOf(steps[2], 'evict').map((e) => e.blockId)).toEqual([2]);
  });
});

describe('cache salt (vllm f26d9f4 kv_cache_utils.py generate_block_hash_extra_keys)', () => {
  it('the salt enters the first block hash, so equal prompts with different salts do not share blocks', () => {
    const { steps } = drive(full4, [
      { id: 'A', prompt: range(1, 9), maxTokens: 1, arrival: 0, cacheSalt: 'x' },
      { id: 'B', prompt: range(1, 9), maxTokens: 1, arrival: 1, cacheSalt: 'y' },
      { id: 'C', prompt: range(1, 9), maxTokens: 1, arrival: 2, cacheSalt: 'x' },
    ]);
    expect(steps[1].hits.get('B')!.numTokens).toBe(0);
    expect(steps[2].hits.get('C')!.numTokens).toBe(8);
  });
});

describe('sliding-window group (vllm f26d9f4 single_type_kv_cache_manager.py SlidingWindowManager)', () => {
  const { steps } = drive(
    { numGpuBlocks: 16, groups: [{ kind: 'sliding_window', blockSize: 2, window: 4 }], maxNumBatchedTokens: 64, maxModelLen: 64 },
    [
      { id: 'A', prompt: range(1, 6), maxTokens: 4, arrival: 0 },
      { id: 'B', prompt: range(1, 6), maxTokens: 2, arrival: 5 },
    ],
  );

  it('only the window tail before num_prompt_tokens - 1 is hashed (single_type_kv_cache_manager.py:1079 reachable_block_mask)', () => {
    expect(eventsOf(steps[0], 'hash').map((e) => e.blockId)).toEqual([1, 2]);
  });

  it('blocks wholly outside the window are freed and replaced by the null block (single_type_kv_cache_manager.py:687, :1160)', () => {
    expect(steps[1].tables.get('A')).toEqual([[0, 2, 3, 4]]);
    expect(eventsOf(steps[1], 'free')).toEqual([{ kind: 'free', req: 'A', group: 0, reason: 'out_of_window', blockIds: [1], toHead: [], toTail: [1] }]);
    expect(steps[2].tables.get('A')).toEqual([[0, 0, 3, 4]]);
  });

  it('a later identical prompt hits the cached window tail of cdiv(window - 1, block) contiguous blocks (single_type_kv_cache_manager.py:981)', () => {
    expect(steps[5].hits.get('B')!.numTokens).toBe(4);
    expect(eventsOf(steps[5], 'touch')[0].blockIds).toEqual([1, 2]);
  });
});

describe('hybrid full + sliding window (vllm f26d9f4 kv_cache_coordinator.py:816 HybridKVCacheCoordinator.find_longest_cache_hit)', () => {
  const { steps } = drive(
    { numGpuBlocks: 32, groups: [{ kind: 'full', blockSize: 2 }, { kind: 'sliding_window', blockSize: 2, window: 4 }], maxNumBatchedTokens: 64, maxModelLen: 64 },
    [
      { id: 'A', prompt: range(1, 8), maxTokens: 1, arrival: 0 },
      { id: 'B', prompt: [...range(1, 8), ...range(50, 4)], maxTokens: 1, arrival: 1 },
    ],
  );

  it('all groups share one pool and each request has one block table per group', () => {
    expect(eventsOf(steps[0], 'allocate').map((e) => [e.group, e.blockIds])).toEqual([[0, [1, 2, 3, 4]], [1, [5, 6, 7, 8]]]);
  });

  it('the hit is the length every group agrees on: full attention has 8 tokens, the window group only 6', () => {
    const hit = steps[1].hits.get('B')!;
    expect(hit.numTokens).toBe(6);
    expect(hit.blocks.map((bs) => bs.filter((b) => !b.isNull).map((b) => b.id))).toEqual([[1, 2, 3], [6, 7]]);
  });

  it('the longer full-attention hit is kept as a shared-prefix junction for window caching (kv_cache_manager.py get_computed_blocks)', () => {
    expect(eventsOf(steps[1], 'hash').filter((e) => e.group === 1).map((e) => e.numTokens)).toEqual([8, 10]);
  });
});

describe('compressed group, a model of the DeepSeek V4 compressed cache', () => {
  const config: Config = {
    numGpuBlocks: 64,
    groups: [{ kind: 'compressed', tokensPerBlock: 16, ratio: 4 }, { kind: 'sliding_window', blockSize: 4, window: 8 }],
    maxNumBatchedTokens: 64,
    maxModelLen: 128,
  };
  const { kv, steps } = drive(config, [
    { id: 'A', prompt: range(1, 22), maxTokens: 12, arrival: 0 },
    { id: 'B', prompt: [...range(1, 22), ...range(60, 8)], maxTokens: 1, arrival: 20 },
  ]);

  it('the scheduler block size is the LCM and the hash block size the GCD of the group block sizes (kv_cache_utils.py:804)', () => {
    expect(kv.schedulerBlockSize).toBe(16);
    expect(kv.hashBlockSize).toBe(4);
    expect(steps[0].tables.get('A')![0]).toEqual([1, 2]);
  });

  it('the compressed cache is hashed only up to the scheduler block boundary in a hybrid model (kv_cache_coordinator.py:799)', () => {
    expect(eventsOf(steps[0], 'hash').filter((e) => e.group === 0).map((e) => e.numTokens)).toEqual([16]);
  });

  it('when the prompt tail block becomes full it also gets the partial prompt-tail hash (single_type_kv_cache_manager.py:864)', () => {
    const hashes = eventsOf(steps[10], 'hash').filter((e) => e.group === 0);
    expect(hashes.map((e) => [e.blockId, e.numTokens, e.partial, e.secondary])).toEqual([[2, 32, false, false], [2, 20, true, true]]);
  });

  it('the prefix hit lands on a multiple of the 16-token scheduler block and needs the window blocks before it', () => {
    const hit = steps[20].hits.get('B')!;
    expect(hit.numTokens).toBe(16);
    expect(hit.blocks.map((bs) => bs.filter((b) => !b.isNull).map((b) => b.id))).toEqual([[1], [5, 6]]);
  });

  it('a compressed ratio that does not divide the hit alignment is rejected (kv_cache_utils.py:900)', () => {
    const make = (groups: KvGroupSpec[]) => () => new KvCacheManager({ numGpuBlocks: 64, groups, maxModelLen: 128, maxInFlightTokens: 64, enablePrefixCaching: true }, () => {});
    expect(make([{ kind: 'compressed', tokensPerBlock: 12, ratio: 12 }, { kind: 'full', blockSize: 8 }])).not.toThrow();
    expect(make([{ kind: 'compressed', tokensPerBlock: 16, ratio: 3 }])).toThrow();
  });
});

describe('prefix caching disabled (vllm f26d9f4 block_pool.py:774)', () => {
  it('nothing is hashed and freed blocks all go to the head', () => {
    const { steps } = drive({ ...full4, enablePrefixCaching: false }, [{ id: 'A', prompt: range(1, 8), maxTokens: 1, arrival: 0 }]);
    expect(eventsOf(steps[0], 'hash')).toEqual([]);
    expect(steps[0].freeQueue.slice(0, 2)).toEqual([2, 1]);
  });
});
