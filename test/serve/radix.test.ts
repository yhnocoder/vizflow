import { describe, expect, test } from 'vitest'
import { RadixCacheSim, type RadixSnapshot } from '../../src/serve'

function put(cache: RadixCacheSim, tokens: readonly number[]): number[] {
  const slots = cache.alloc(tokens.length)
  if (!slots) throw new Error('out of slots')
  cache.insert(tokens, slots)
  return slots
}

function node(snap: RadixSnapshot, key: readonly (number | string)[]) {
  const n = snap.nodes.find((x) => JSON.stringify(x.key) === JSON.stringify(key))
  if (!n) throw new Error(`no node with key ${JSON.stringify(key)}`)
  return n
}

describe('SGLang token-to-KV slot allocator (sglang d62b9c24 mem_cache/allocator/token.py)', () => {
  test('slots start at 1, alloc takes from the head of free_pages and free appends to the tail (token.py:42 clear, :54 alloc, :65 free)', () => {
    const cache = new RadixCacheSim(6)
    expect(cache.alloc(3)).toEqual([1, 2, 3])
    cache.free([2, 1])
    expect(cache.snapshot().freeSlots).toEqual([4, 5, 6, 2, 1])
    expect(cache.alloc(4)).toEqual([4, 5, 6, 2])
    expect(cache.alloc(2)).toBeNull()
  })

  test('with need_sort (PD disaggregation) freed slots wait in release_pages and are merged sorted when alloc runs short (token.py:54, base.py:207 merge_and_sort_free)', () => {
    const cache = new RadixCacheSim(4, true)
    cache.alloc(4)
    cache.free([3, 1])
    expect(cache.snapshot().releasedSlots).toEqual([3, 1])
    expect(cache.alloc(1)).toEqual([1])
    expect(cache.snapshot().freeSlots).toEqual([3])
  })
})

describe('SGLang radix cache, page_size 1 (sglang d62b9c24 mem_cache/radix_cache.py)', () => {
  test('match_prefix splits a node when the match ends inside its key (radix_cache.py:651 _match_prefix_helper, :677 _split_node)', () => {
    const cache = new RadixCacheSim(16)
    put(cache, [1, 2, 3, 4])
    const m = cache.matchPrefix([1, 2, 9])
    expect(m.prefixLength).toBe(2)
    expect(m.slots).toEqual([1, 2])
    const s = cache.snapshot()
    const head = node(s, [1, 2])
    const tail = node(s, [3, 4])
    expect(head.id).toBe(m.lastNode)
    expect(head.slots).toEqual([1, 2])
    expect(tail.parent).toBe(head.id)
    expect(tail.slots).toEqual([3, 4])
    expect(s.events.filter((e) => e.kind === 'split')).toEqual([{ kind: 'split', node: tail.id, newNode: head.id, at: 2 }])
  })

  test('evict frees least recently accessed leaves first (LRUStrategy, radix_cache.py:552 evict)', () => {
    const cache = new RadixCacheSim(16)
    put(cache, [1, 2, 3])
    put(cache, [4, 5])
    put(cache, [6, 7, 8, 9])
    cache.matchPrefix([4, 5])
    cache.snapshot()
    expect(cache.evict(4)).toBe(7)
    const s = cache.snapshot()
    expect(s.events.filter((e) => e.kind === 'evict').map((e) => (e.kind === 'evict' ? e.slots : []))).toEqual([
      [1, 2, 3],
      [6, 7, 8, 9],
    ])
    expect(s.nodes.map((n) => n.key)).toEqual([[], [4, 5]])
  })

  test('a parent becomes an eviction candidate once it is a childless unlocked leaf (radix_cache.py:572)', () => {
    const cache = new RadixCacheSim(16)
    put(cache, [1, 2, 3])
    put(cache, [1, 2, 4])
    cache.snapshot()
    expect(cache.evict(3)).toBe(4)
    const evicted = cache.snapshot().events.flatMap((e) => (e.kind === 'evict' ? [e.slots] : []))
    expect(evicted).toEqual([[3], [6], [1, 2]])
  })

  test('lock_ref protects the locked path from eviction; only the unlocked suffix is evicted (radix_cache.py:582 inc_lock_ref, :790 _update_leaf_status)', () => {
    const cache = new RadixCacheSim(16)
    put(cache, [1, 2, 3, 4, 5])
    const m = cache.matchPrefix([1, 2, 3])
    cache.incLockRef(m.lastNode)
    let s = cache.snapshot()
    expect(s.protectedSize).toBe(3)
    expect(s.evictableSize).toBe(2)
    expect(cache.evict(100)).toBe(2)
    s = cache.snapshot()
    expect(node(s, [1, 2, 3]).lockRef).toBe(1)
    expect(s.nodes.length).toBe(2)
    cache.decLockRef(m.lastNode)
    expect(cache.evict(100)).toBe(3)
    expect(cache.snapshot().freeSlots).toEqual([...Array.from({ length: 11 }, (_, i) => i + 6), 4, 5, 1, 2, 3])
  })

  test('logical clock: a match refreshes lastAccess along the matched path', () => {
    const cache = new RadixCacheSim(8)
    put(cache, [1, 2])
    const before = node(cache.snapshot(), [1, 2]).lastAccess
    cache.matchPrefix([1, 2])
    expect(node(cache.snapshot(), [1, 2]).lastAccess).toBeGreaterThan(before)
  })
})

describe('SGLang request flow over the radix cache (sglang d62b9c24 mem_cache/common.py:378 release_kv_cache, radix_cache.py:510 checkpoint)', () => {
  test('finished request is inserted, split at the prompt boundary, and unlocked; the next request reuses the prompt slots', () => {
    const cache = new RadixCacheSim(8)
    cache.admit('r1', [1, 2, 3])
    cache.extend('r1', [7])
    cache.release('r1')
    let s = cache.snapshot()
    expect(node(s, [1, 2, 3]).slots).toEqual([1, 2, 3])
    expect(node(s, [7]).slots).toEqual([4])
    expect(s.protectedSize).toBe(0)
    const m = cache.admit('r2', [1, 2, 3, 8])
    expect(m.prefixLength).toBe(3)
    s = cache.snapshot()
    expect(s.requests[0]!.slots).toEqual([1, 2, 3, 5])
    expect(node(s, [1, 2, 3]).lockRef).toBe(1)
  })

  test('two requests that computed the same prompt: the second release frees its duplicate slots (radix_cache.py:450 _insert_cache free_segment)', () => {
    const cache = new RadixCacheSim(6)
    cache.admit('a', [1, 2])
    cache.admit('b', [1, 2])
    cache.release('a')
    cache.snapshot()
    cache.release('b')
    const s = cache.snapshot()
    expect(s.events.filter((e) => e.kind === 'free')).toEqual([{ kind: 'free', req: 'b', slots: [3, 4] }])
    expect(s.freeSlots).toEqual([5, 6, 3, 4])
  })

  test('admission evicts only the shortfall when free slots run out (allocator/base.py:135 evict_to_free_tokens)', () => {
    const cache = new RadixCacheSim(4)
    cache.admit('a', [1, 2])
    cache.release('a')
    cache.admit('b', [5, 6])
    cache.release('b')
    cache.snapshot()
    cache.admit('c', [9, 9])
    const s = cache.snapshot()
    expect(s.events.filter((e) => e.kind === 'evict').map((e) => (e.kind === 'evict' ? e.slots : []))).toEqual([[1, 2]])
    expect(s.requests[0]!.slots).toEqual([1, 2])
  })
})
