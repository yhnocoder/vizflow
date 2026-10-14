import { describe, expect, it } from 'vitest';
import { RadixCacheSim, type RadixSnapshot } from '../../../src/sim/serve';

function put(cache: RadixCacheSim, tokens: number[]): number[] {
  const slots = cache.alloc(tokens.length);
  if (!slots) throw new Error('out of slots');
  cache.insert(tokens, slots);
  return slots;
}

function node(snap: RadixSnapshot, key: (number | string)[]) {
  const n = snap.nodes.find((x) => JSON.stringify(x.key) === JSON.stringify(key));
  if (!n) throw new Error(`no node with key ${JSON.stringify(key)}`);
  return n;
}

function start(cache: RadixCacheSim, id: string, tokens: number[], promptLength: number, computed: number) {
  const match = cache.matchPrefix(tokens.slice(0, promptLength));
  cache.admit(id, tokens, promptLength, match);
  cache.extend(id, computed - match.prefixLength);
  return match;
}

const evicted = (s: RadixSnapshot) => s.events.flatMap((e) => (e.kind === 'evict' ? [e.slots] : []));

describe('SGLang token-to-KV slot allocator (sglang d62b9c24 mem_cache/allocator/token.py)', () => {
  it('slots start at 1, alloc takes from the head of free_pages and free appends to the tail (token.py:42, :54, :65)', () => {
    const cache = new RadixCacheSim(6);
    expect(cache.alloc(3)).toEqual([1, 2, 3]);
    cache.free([2, 1]);
    expect(cache.snapshot().freeSlots).toEqual([4, 5, 6, 2, 1]);
    expect(cache.alloc(4)).toEqual([4, 5, 6, 2]);
    expect(cache.alloc(2)).toBeNull();
  });

  it('with need_sort freed slots wait in release_pages and are merged sorted when alloc runs short (token.py:54, base.py:207)', () => {
    const cache = new RadixCacheSim(4, true);
    cache.alloc(4);
    cache.free([3, 1]);
    expect(cache.snapshot().releasedSlots).toEqual([3, 1]);
    expect(cache.alloc(1)).toEqual([1]);
    expect(cache.snapshot().freeSlots).toEqual([3]);
  });
});

describe('SGLang radix cache, page_size 1 (sglang d62b9c24 mem_cache/radix_cache.py)', () => {
  it('match_prefix splits a node when the match ends inside its key (radix_cache.py:651, :677)', () => {
    const cache = new RadixCacheSim(16);
    put(cache, [1, 2, 3, 4]);
    const m = cache.matchPrefix([1, 2, 9]);
    expect(m.prefixLength).toBe(2);
    expect(m.slots).toEqual([1, 2]);
    const s = cache.snapshot();
    const head = node(s, [1, 2]);
    const tail = node(s, [3, 4]);
    expect(head.id).toBe(m.lastNode);
    expect(tail.parent).toBe(head.id);
    expect(tail.slots).toEqual([3, 4]);
    expect(s.events.filter((e) => e.kind === 'split')).toEqual([{ kind: 'split', node: tail.id, newNode: head.id, at: 2 }]);
  });

  it('cachedPrefixLength reads the cached prefix without splitting or refreshing nodes', () => {
    const cache = new RadixCacheSim(16);
    put(cache, [1, 2, 3, 4]);
    const before = cache.snapshot();
    expect(cache.cachedPrefixLength([1, 2, 9])).toBe(2);
    expect(cache.cachedPrefixLength([1, 2, 3, 4, 5])).toBe(4);
    const after = cache.snapshot();
    expect(after.nodes).toEqual(before.nodes);
    expect(after.events).toEqual([]);
  });

  it('evict frees least recently accessed leaves first, whole leaves at a time (radix_cache.py:552 evict)', () => {
    const cache = new RadixCacheSim(16);
    put(cache, [1, 2, 3]);
    put(cache, [4, 5]);
    put(cache, [6, 7, 8, 9]);
    cache.matchPrefix([4, 5]);
    cache.snapshot();
    expect(cache.evict(4)).toBe(7);
    const s = cache.snapshot();
    expect(evicted(s)).toEqual([[1, 2, 3], [6, 7, 8, 9]]);
    expect(s.nodes.map((n) => n.key)).toEqual([[], [4, 5]]);
  });

  it('a parent becomes an eviction candidate once it is a childless unlocked leaf (radix_cache.py:572)', () => {
    const cache = new RadixCacheSim(16);
    put(cache, [1, 2, 3]);
    put(cache, [1, 2, 4]);
    cache.snapshot();
    expect(cache.evict(3)).toBe(4);
    expect(evicted(cache.snapshot())).toEqual([[3], [6], [1, 2]]);
  });

  it('lock_ref protects the locked path; only the unlocked suffix is evicted (radix_cache.py:582, :790)', () => {
    const cache = new RadixCacheSim(16);
    put(cache, [1, 2, 3, 4, 5]);
    const m = cache.matchPrefix([1, 2, 3]);
    cache.incLockRef(m.lastNode);
    expect([cache.protectedSize(), cache.evictableSize()]).toEqual([3, 2]);
    expect(cache.evict(100)).toBe(2);
    const s = cache.snapshot();
    expect(node(s, [1, 2, 3]).lockRef).toBe(1);
    expect(s.nodes.length).toBe(2);
    cache.decLockRef(m.lastNode);
    expect(cache.evict(100)).toBe(3);
    expect(cache.snapshot().freeSlots).toEqual([...Array.from({ length: 11 }, (_, i) => i + 6), 4, 5, 1, 2, 3]);
  });

  it('a match refreshes lastAccess along the matched path', () => {
    const cache = new RadixCacheSim(8);
    put(cache, [1, 2]);
    const before = node(cache.snapshot(), [1, 2]).lastAccess;
    cache.matchPrefix([1, 2]);
    expect(node(cache.snapshot(), [1, 2]).lastAccess).toBeGreaterThan(before);
  });
});

describe('SGLang request flow over the radix cache (sglang d62b9c24 mem_cache/common.py:378 release_kv_cache, radix_cache.py:510 checkpoint)', () => {
  it('a finished request is inserted, split at the prompt boundary and unlocked; the next request reuses the prompt slots', () => {
    const cache = new RadixCacheSim(8);
    start(cache, 'r1', [1, 2, 3, 7], 3, 4);
    cache.release('r1', 4, true);
    let s = cache.snapshot();
    expect(node(s, [1, 2, 3]).slots).toEqual([1, 2, 3]);
    expect(node(s, [7]).slots).toEqual([4]);
    expect(s.protectedSize).toBe(0);
    const m = start(cache, 'r2', [1, 2, 3, 8], 4, 4);
    expect(m.prefixLength).toBe(3);
    s = cache.snapshot();
    expect(s.requests[0].slots).toEqual([1, 2, 3, 5]);
    expect(node(s, [1, 2, 3]).lockRef).toBe(1);
  });

  it('a checkpoint of a chunk moves the lock to the new last node and keeps the request slots (radix_cache.py:510)', () => {
    const cache = new RadixCacheSim(8);
    start(cache, 'r', [1, 2, 3, 4, 5], 5, 3);
    cache.checkpoint('r', 3);
    const s = cache.snapshot();
    expect(node(s, [1, 2, 3]).lockRef).toBe(1);
    expect(s.requests[0]).toMatchObject({ slots: [1, 2, 3], cacheProtectedLen: 3 });
    expect([s.protectedSize, s.evictableSize]).toEqual([3, 0]);
  });

  it('two requests that computed the same prompt: the second release frees its duplicate slots (radix_cache.py:450 _insert_cache)', () => {
    const cache = new RadixCacheSim(6);
    start(cache, 'a', [1, 2], 2, 2);
    start(cache, 'b', [1, 2], 2, 2);
    cache.release('a', 2, true);
    cache.snapshot();
    cache.release('b', 2, true);
    const s = cache.snapshot();
    expect(s.events.filter((e) => e.kind === 'free')).toEqual([{ kind: 'free', req: 'b', slots: [3, 4] }]);
    expect(s.freeSlots).toEqual([5, 6, 3, 4]);
  });

  it('a retracted request frees what it owns and unlocks without inserting (release_kv_cache with checkpoint=False)', () => {
    const cache = new RadixCacheSim(8);
    put(cache, [1, 2]);
    start(cache, 'r', [1, 2, 3, 4], 4, 4);
    cache.release('r', 0, false);
    const s = cache.snapshot();
    expect(s.nodes.map((n) => n.key)).toEqual([[], [1, 2]]);
    expect([s.protectedSize, s.evictableSize]).toEqual([0, 2]);
    expect(s.freeSlots).toEqual([5, 6, 7, 8, 3, 4]);
  });

  it('allocation evicts only the shortfall when free slots run out (allocator/base.py:135 evict_to_free_tokens)', () => {
    const cache = new RadixCacheSim(4);
    start(cache, 'a', [1, 2], 2, 2);
    cache.release('a', 2, true);
    start(cache, 'b', [5, 6], 2, 2);
    cache.release('b', 2, true);
    cache.snapshot();
    start(cache, 'c', [9, 9], 2, 2);
    const s = cache.snapshot();
    expect(evicted(s)).toEqual([[1, 2]]);
    expect(s.requests[0].slots).toEqual([1, 2]);
  });

  it('allocation that cannot be met even after eviction throws', () => {
    const cache = new RadixCacheSim(2);
    expect(() => start(cache, 'a', [1, 2, 3], 3, 3)).toThrow(/out of KV slots/);
  });
});
