import { describe, expect, it } from 'vitest';
import { BlockPool, type VllmEvent } from '../../../src/sim/serve';

const ctx = { req: 'r', group: 0 };

describe('vLLM block pool (vllm f26d9f4 v1/core/block_pool.py)', () => {
  it('block 0 is the null block and is never handed out or freed', () => {
    const pool = new BlockPool(4, true, 4, () => {});
    expect(pool.freeQueue).toEqual([1, 2, 3]);
    pool.freeBlocks([pool.nullBlock], ctx, 'finish');
    expect(pool.freeQueue).toEqual([1, 2, 3]);
    expect(() => pool.getNewBlocks(4, ctx)).toThrow();
  });

  it('a cached block with refCnt 0 stays findable until it is reused, and touch takes it out of the free queue (block_pool.py:752)', () => {
    const pool = new BlockPool(4, true, 4, () => {});
    const [a, b] = pool.getNewBlocks(2, ctx);
    pool.cacheFullBlocks([a, b], ['h1', 'h2'], 0, 2, 4, null, ctx);
    pool.freeBlocks([b, a], ctx, 'finish');
    expect(pool.freeQueue).toEqual([3, 2, 1]);
    expect(pool.getCachedBlock('h1', [0])!.map((x) => x.id)).toEqual([1]);
    pool.touch([a], ctx);
    expect(pool.freeQueue).toEqual([3, 2]);
    expect(a.refCnt).toBe(1);
  });

  it('two blocks may carry the same hash; the first one registered is returned (block_pool.py get_cached_block)', () => {
    const events: VllmEvent[] = [];
    const pool = new BlockPool(4, true, 4, (e) => events.push(e));
    const [a, b] = pool.getNewBlocks(2, ctx);
    pool.cacheFullBlocks([a], ['h'], 0, 1, 4, null, ctx);
    pool.cacheFullBlocks([b], ['h'], 0, 1, 4, null, ctx);
    expect(pool.getCachedBlock('h', [0])!.map((x) => x.id)).toEqual([1]);
    pool.freeBlocks([a], ctx, 'finish');
    pool.getNewBlocks(2, ctx);
    expect(pool.getCachedBlock('h', [0])!.map((x) => x.id)).toEqual([2]);
    expect(events.filter((e) => e.kind === 'evict').map((e) => e.kind === 'evict' && [e.blockId, e.reason])).toEqual([[1, 'reuse']]);
  });
});
