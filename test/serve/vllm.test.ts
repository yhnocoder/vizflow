import { describe, expect, test } from 'vitest'
import { VllmEngine, simulateVllm, type VllmConfig, type VllmEvent, type VllmSnapshot } from '../../src/serve'

const range = (start: number, n: number) => Array.from({ length: n }, (_, i) => start + i)

const full4: VllmConfig = { numGpuBlocks: 16, groups: [{ kind: 'full', blockSize: 4 }], maxNumBatchedTokens: 64, maxNumSeqs: 4, maxModelLen: 128 }

function req(id: string, snap: VllmSnapshot) {
  const r = snap.requests.find((x) => x.id === id)
  if (!r) throw new Error(`no request ${id}`)
  return r
}

function block(id: number, snap: VllmSnapshot) {
  return snap.blocks[id]!
}

function eventsOf<K extends VllmEvent['kind']>(snap: VllmSnapshot, kind: K): Extract<VllmEvent, { kind: K }>[] {
  return snap.events.filter((e): e is Extract<VllmEvent, { kind: K }> => e.kind === kind)
}

describe('vLLM block pool and prefix cache (vllm f26d9f4)', () => {
  const snaps = simulateVllm(full4, [
    { id: 'A', prompt: range(1, 10), outputs: [100, 101, 102], maxTokens: 3, arrival: 0 },
    { id: 'B', prompt: [...range(1, 8), 50, 51, 52, 53], outputs: [200, 201], maxTokens: 2, arrival: 1 },
  ])

  test('block 0 is the null block and allocation starts at block 1 (block_pool.py:187 null_block = free_block_queue.popleft())', () => {
    expect(snaps[0]!.blocks[0]!.isNull).toBe(true)
    expect(req('A', snaps[0]!).blockTables).toEqual([[1, 2, 3]])
    expect(snaps[0]!.freeQueue[0]).toBe(4)
  })

  test('only full blocks are hashed, chained over the prefix (kv_cache_utils.py:695 hash_block_tokens, kv_cache_manager.py:612 cache up to num_tokens)', () => {
    const hashed = eventsOf(snaps[0]!, 'hash').map((e) => [e.blockId, e.numTokens])
    expect(hashed).toEqual([
      [1, 4],
      [2, 8],
    ])
    expect(block(3, snaps[0]!).hash).toBeNull()
  })

  test('shared system prompt: second request hits two blocks and their refCnt becomes 2 (kv_cache_manager.py:295 get_computed_blocks, block_pool.py:752 touch)', () => {
    const s = snaps[1]!
    expect(req('B', s).prefixHitTokens).toBe(8)
    expect(req('B', s).blockTables).toEqual([[1, 2, 4]])
    expect(eventsOf(s, 'hit')).toEqual([{ kind: 'hit', req: 'B', group: 0, blockIds: [1, 2] }])
    for (const id of [1, 2]) {
      expect(block(id, s).refCnt).toBe(2)
      expect(block(id, s).refs.map((r) => r.req)).toEqual(['A', 'B'])
    }
    expect(eventsOf(s, 'touch')).toEqual([])
  })

  test('a block becomes hashed in the step that fills it: decode token 12 fills block 3 (kv_cache_manager.py:612)', () => {
    expect(eventsOf(snaps[2]!, 'hash').map((e) => e.blockId)).toEqual([3])
  })

  test('finish order decides free queue order: blocks freed in reverse, hashed to the tail, unhashed prepended (block_pool.py:774 free_blocks)', () => {
    const frees = eventsOf(snaps[2]!, 'free')
    expect(frees.map((f) => [f.req, f.blockIds, f.toHead, f.toTail])).toEqual([
      ['A', [3, 2, 1], [], [3]],
      ['B', [5, 4, 2, 1], [5], [4, 2, 1]],
    ])
    expect(snaps[2]!.freeQueue).toEqual([5, ...range(6, 10), 3, 4, 2, 1])
  })

  test('snapshots are frozen', () => {
    expect(Object.isFrozen(snaps[0]!)).toBe(true)
    expect(Object.isFrozen(snaps[0]!.blocks[1]!)).toBe(true)
  })
})

describe('reuse order and eviction on reallocation (vllm f26d9f4)', () => {
  const config: VllmConfig = { ...full4, numGpuBlocks: 7 }
  const snaps = simulateVllm(config, [
    { id: 'A', prompt: range(1, 8), outputs: [100], maxTokens: 1, arrival: 0 },
    { id: 'C', prompt: range(30, 20), outputs: [300], maxTokens: 1, arrival: 1 },
    { id: 'D', prompt: range(1, 8), outputs: [400], maxTokens: 1, arrival: 2 },
  ])

  test('allocation pops the head of the free queue; a popped block that still carries a hash loses it (block_pool.py:666 get_new_blocks, :729 _maybe_evict_cached_block)', () => {
    expect(snaps[0]!.freeQueue).toEqual([3, 4, 5, 6, 2, 1])
    const s = snaps[1]!
    expect(eventsOf(s, 'allocate')[0]!.blockIds).toEqual([3, 4, 5, 6, 2])
    expect(eventsOf(s, 'evict').map((e) => [e.blockId, e.reason])).toEqual([[2, 'reuse']])
  })

  test('hit is capped at num_tokens - 1, so an 8-token prompt cached in 2 blocks hits 1 block (kv_cache_manager.py:295)', () => {
    const s = snaps[2]!
    expect(req('D', s).prefixHitTokens).toBe(4)
    expect(eventsOf(s, 'touch')).toEqual([{ kind: 'touch', req: 'D', group: 0, blockIds: [1] }])
    expect(eventsOf(s, 'allocate')[0]!.blockIds).toEqual([2])
    expect(eventsOf(s, 'evict').map((e) => e.blockId)).toEqual([2])
  })
})

describe('preemption under a tiny pool (vllm f26d9f4 scheduler.py)', () => {
  const snaps = simulateVllm({ ...full4, numGpuBlocks: 5 }, [
    { id: 'A', prompt: range(1, 6), outputs: range(100, 6), maxTokens: 6, arrival: 0 },
    { id: 'B', prompt: range(20, 6), outputs: range(200, 6), maxTokens: 6, arrival: 0 },
  ])

  test('allocate_slots failure preempts running[-1], frees its blocks, resets num_computed_tokens and prepends it to waiting (scheduler.py:802, :1604, :1624)', () => {
    const s = snaps[3]!
    expect(eventsOf(s, 'preempt')).toEqual([{ kind: 'preempt', req: 'B', for: 'A' }])
    expect(eventsOf(s, 'free')[0]).toMatchObject({ req: 'B', reason: 'preempt', blockIds: [4, 3], toTail: [4, 3] })
    expect(req('B', s)).toMatchObject({ status: 'preempted', numComputedTokens: 0, numPreemptions: 1 })
    expect(s.waiting).toEqual(['B'])
    expect(req('A', s).blockTables).toEqual([[1, 2, 4]])
  })

  test('no waiting request is admitted in a step that preempted (scheduler.py:900)', () => {
    expect(eventsOf(snaps[3]!, 'admit')).toEqual([])
  })

  test('admission needs the whole sequence to fit, so B waits while A runs (scheduler_reserve_full_isl, scheduler.py:1270)', () => {
    expect(req('B', snaps[4]!).status).toBe('preempted')
    expect(req('A', snaps[5]!).status).toBe('finished')
  })

  test('resumed request recomputes prompt and outputs with a prefix hit on its surviving block (scheduler.py:1129)', () => {
    const s = snaps[6]!
    expect(eventsOf(s, 'admit')).toEqual([{ kind: 'admit', req: 'B', resumed: true, hitTokens: 4, numNewTokens: 5 }])
    expect(eventsOf(s, 'touch')[0]!.blockIds).toEqual([3])
    expect(req('B', s).blockTables).toEqual([[3, 4, 2]])
    expect(req('B', snaps[snaps.length - 1]!).numOutputTokens).toBe(6)
  })
})

describe('chunked prefill (vllm f26d9f4 scheduler.py)', () => {
  const snaps = simulateVllm({ ...full4, maxNumBatchedTokens: 4 }, [
    { id: 'A', prompt: range(1, 10), outputs: [100, 101], maxTokens: 2, arrival: 0 },
  ])

  test('a 10-token prompt is prefilled in chunks of the token budget and samples only after the last chunk', () => {
    expect(snaps.map((s) => [req('A', s).numComputedTokens, req('A', s).numOutputTokens])).toEqual([
      [4, 0],
      [8, 0],
      [10, 1],
      [11, 2],
    ])
  })

  test('each chunk hashes the block it fills and allocates only the blocks it needs', () => {
    expect(snaps.map((s) => eventsOf(s, 'hash').map((e) => e.blockId))).toEqual([[1], [2], [], []])
    expect(snaps.map((s) => eventsOf(s, 'allocate').flatMap((e) => e.blockIds))).toEqual([[1], [2], [3], []])
  })

  test('without chunked prefill a prompt longer than the budget is never admitted', () => {
    expect(() =>
      simulateVllm({ ...full4, maxNumBatchedTokens: 4, enableChunkedPrefill: false }, [
        { id: 'A', prompt: range(1, 10), outputs: [1], maxTokens: 1, arrival: 0 },
      ]),
    ).toThrow(/nothing could be scheduled/)
  })
})

describe('sliding-window group (vllm f26d9f4 single_type_kv_cache_manager.py SlidingWindowManager)', () => {
  const snaps = simulateVllm(
    { numGpuBlocks: 16, groups: [{ kind: 'sliding_window', blockSize: 2, window: 4 }], maxNumBatchedTokens: 64, maxNumSeqs: 4, maxModelLen: 64 },
    [
      { id: 'A', prompt: range(1, 6), outputs: range(100, 4), maxTokens: 4, arrival: 0 },
      { id: 'B', prompt: range(1, 6), outputs: range(200, 2), maxTokens: 2, arrival: 5 },
    ],
  )

  test('with prefix_cache_retention_interval=0 only the window tail before num_prompt_tokens-1 is hashed (cache.py:159, single_type_kv_cache_manager.py:1079 reachable_block_mask, kv_cache_coordinator.py:65 replay_boundaries)', () => {
    expect(eventsOf(snaps[0]!, 'hash').map((e) => e.blockId)).toEqual([1, 2])
    expect(block(3, snaps[0]!).hash).toBeNull()
  })

  test('blocks wholly outside the window are freed and replaced by the null block (single_type_kv_cache_manager.py:687 remove_skipped_blocks, :1160 get_num_skipped_tokens)', () => {
    expect(req('A', snaps[1]!).blockTables).toEqual([[0, 2, 3, 4]])
    expect(eventsOf(snaps[1]!, 'free')).toEqual([
      { kind: 'free', req: 'A', group: 0, reason: 'out_of_window', blockIds: [1], toHead: [], toTail: [1] },
    ])
    expect(req('A', snaps[2]!).blockTables).toEqual([[0, 0, 3, 4]])
  })

  test('a later identical prompt hits the cached window tail of cdiv(window-1, block) contiguous blocks (single_type_kv_cache_manager.py:981)', () => {
    const s = snaps[5]!
    expect(req('B', s).prefixHitTokens).toBe(4)
    expect(eventsOf(s, 'touch')[0]!.blockIds).toEqual([1, 2])
  })
})

describe('hybrid full + sliding window (vllm f26d9f4 kv_cache_coordinator.py:816 HybridKVCacheCoordinator.find_longest_cache_hit)', () => {
  const snaps = simulateVllm(
    {
      numGpuBlocks: 32,
      groups: [
        { kind: 'full', blockSize: 2 },
        { kind: 'sliding_window', blockSize: 2, window: 4 },
      ],
      maxNumBatchedTokens: 64,
      maxNumSeqs: 4,
      maxModelLen: 64,
    },
    [
      { id: 'A', prompt: range(1, 8), outputs: [100], maxTokens: 1, arrival: 0 },
      { id: 'B', prompt: [...range(1, 8), ...range(50, 4)], outputs: [200], maxTokens: 1, arrival: 1 },
    ],
  )

  test('all groups share one pool and each request has one block table per group', () => {
    expect(eventsOf(snaps[0]!, 'allocate').map((e) => [e.group, e.blockIds])).toEqual([
      [0, [1, 2, 3, 4]],
      [1, [5, 6, 7, 8]],
    ])
  })

  test('the hit is the length every group agrees on: full attention has 8 tokens, the window group only 6', () => {
    const s = snaps[1]!
    expect(req('B', s).prefixHitTokens).toBe(6)
    expect(eventsOf(s, 'hit').map((e) => [e.group, e.blockIds])).toEqual([
      [0, [1, 2, 3]],
      [1, [6, 7]],
    ])
  })

  test('the longer full-attention hit is kept as a shared-prefix junction for window caching (kv_cache_manager.py get_computed_blocks shared_prefix_boundary)', () => {
    const swaHashes = eventsOf(snaps[1]!, 'hash').filter((e) => e.group === 1)
    expect(swaHashes.map((e) => e.numTokens)).toEqual([8, 10])
  })
})

describe('compressed group: model of DeepSeek V4 compressed cache, layout to be confirmed (research/kv-cache/serving-v4.md section 1.1 and 1.5)', () => {
  const config: VllmConfig = {
    numGpuBlocks: 64,
    groups: [
      { kind: 'compressed', tokensPerBlock: 16, ratio: 4 },
      { kind: 'sliding_window', blockSize: 4, window: 8 },
    ],
    maxNumBatchedTokens: 64,
    maxNumSeqs: 4,
    maxModelLen: 128,
  }
  const snaps = simulateVllm(config, [
    { id: 'A', prompt: range(1, 22), outputs: range(100, 12), maxTokens: 12, arrival: 0 },
    { id: 'B', prompt: [...range(1, 22), ...range(60, 8)], outputs: [200], maxTokens: 1, arrival: 20 },
  ])

  test('scheduler block size is the LCM and hash block size the GCD of group block sizes (kv_cache_utils.py:804 resolve_kv_cache_block_sizes)', () => {
    expect(snaps[0]!.schedulerBlockSize).toBe(16)
    expect(snaps[0]!.hashBlockSize).toBe(4)
  })

  test('a compressed block covers tokensPerBlock source tokens and holds tokensPerBlock/ratio entries (kv_cache_interface.py:228 num_states)', () => {
    const s = snaps[0]!
    expect(req('A', s).blockTables[0]).toEqual([1, 2])
    expect([block(1, s).filled, block(1, s).capacity]).toEqual([4, 4])
    expect([block(2, s).filled, block(2, s).capacity]).toEqual([1, 4])
  })

  test('the compressed cache is hashed only up to the scheduler block boundary in a hybrid model (kv_cache_coordinator.py:799 get_num_cacheable_tokens)', () => {
    expect(eventsOf(snaps[0]!, 'hash').filter((e) => e.group === 0).map((e) => e.numTokens)).toEqual([16])
  })

  test('when the prompt tail block becomes full it also gets the partial prompt-tail hash (single_type_kv_cache_manager.py:864 _cache_partial_tail_block)', () => {
    const hashes = eventsOf(snaps[10]!, 'hash').filter((e) => e.group === 0)
    expect(hashes.map((e) => [e.blockId, e.numTokens, e.partial, e.secondary])).toEqual([
      [2, 32, false, false],
      [2, 20, true, true],
    ])
  })

  test('prefix hit lands on a multiple of the 16-token scheduler block and needs the window blocks before it', () => {
    const s = snaps[20]!
    expect(req('B', s).prefixHitTokens).toBe(16)
    expect(eventsOf(s, 'hit').map((e) => [e.group, e.blockIds])).toEqual([
      [0, [1]],
      [1, [5, 6]],
    ])
  })

  test('a compressed ratio that does not divide the hit alignment is rejected (kv_cache_utils.py:900)', () => {
    expect(
      () =>
        new VllmEngine(
          { ...config, groups: [{ kind: 'compressed', tokensPerBlock: 12, ratio: 12 }, { kind: 'full', blockSize: 8 }] },
          [],
        ),
    ).not.toThrow()
    expect(() => new VllmEngine({ ...config, groups: [{ kind: 'compressed', tokensPerBlock: 16, ratio: 3 }] }, [])).toThrow()
  })
})

describe('prefix caching disabled (vllm f26d9f4 block_pool.py:774)', () => {
  test('freed blocks all go to the head and nothing is hashed', () => {
    const snaps = simulateVllm({ ...full4, enablePrefixCaching: false }, [
      { id: 'A', prompt: range(1, 8), outputs: [1], maxTokens: 1, arrival: 0 },
    ])
    expect(eventsOf(snaps[0]!, 'hash')).toEqual([])
    expect(snaps[0]!.freeQueue.slice(0, 2)).toEqual([2, 1])
  })
})
