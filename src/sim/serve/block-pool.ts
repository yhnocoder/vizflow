export interface BlockHashKey {
  readonly hash: string;
  readonly group: number;
}

export interface Block {
  readonly id: number;
  readonly isNull: boolean;
  refCnt: number;
  primaryHash: BlockHashKey | null;
  primaryHashTokens: number | null;
  extraHashes: BlockHashKey[];
}

export type VllmEvent =
  | { readonly kind: 'touch'; readonly req: string; readonly group: number; readonly blockIds: readonly number[] }
  | { readonly kind: 'allocate'; readonly req: string; readonly group: number; readonly blockIds: readonly number[] }
  | { readonly kind: 'hash'; readonly req: string; readonly group: number; readonly blockId: number; readonly hash: string; readonly numTokens: number; readonly partial: boolean; readonly secondary: boolean }
  | { readonly kind: 'evict'; readonly req: string; readonly group: number; readonly blockId: number; readonly hash: string; readonly hashGroup: number; readonly reason: 'reuse' | 'promote' }
  | { readonly kind: 'free'; readonly req: string; readonly group: number; readonly reason: 'finish' | 'preempt' | 'out_of_window'; readonly blockIds: readonly number[]; readonly toHead: readonly number[]; readonly toTail: readonly number[] };

export interface PoolContext {
  readonly req: string;
  readonly group: number;
}

function hashKeyString(key: BlockHashKey): string {
  return `${key.hash}@${key.group}`;
}

function sameKey(a: BlockHashKey | null, b: BlockHashKey): boolean {
  return a !== null && a.hash === b.hash && a.group === b.group;
}

export class BlockPool {
  readonly blocks: Block[];
  freeQueue: number[];
  readonly nullBlock: Block;
  private readonly cached = new Map<string, number[]>();

  constructor(
    numBlocks: number,
    readonly enableCaching: boolean,
    readonly hashBlockSize: number,
    private readonly emit: (event: VllmEvent) => void,
  ) {
    if (!Number.isInteger(numBlocks) || numBlocks < 2) throw new Error('numGpuBlocks must be an integer >= 2 (block 0 is the null block)');
    this.blocks = Array.from({ length: numBlocks }, (_, id) => ({ id, isNull: id === 0, refCnt: 0, primaryHash: null, primaryHashTokens: null, extraHashes: [] }));
    this.freeQueue = this.blocks.slice(1).map((b) => b.id);
    this.nullBlock = this.blocks[0];
  }

  get numFreeBlocks(): number {
    return this.freeQueue.length;
  }

  getCachedBlock(hash: string, groups: readonly number[]): Block[] | null {
    const found: Block[] = [];
    for (const group of groups) {
      const ids = this.cached.get(hashKeyString({ hash, group }));
      if (!ids || ids.length === 0) return null;
      found.push(this.blocks[ids[0]]);
    }
    return found;
  }

  private contains(key: BlockHashKey, blockId: number): boolean {
    return this.cached.get(hashKeyString(key))?.includes(blockId) ?? false;
  }

  private insertHash(key: BlockHashKey, block: Block, numTokens: number, ctx: PoolContext, partial: boolean): void {
    if (sameKey(block.primaryHash, key)) return;
    if (this.contains(key, block.id)) return;
    const secondary = block.primaryHash !== null;
    if (secondary) block.extraHashes.push(key);
    else {
      block.primaryHash = key;
      block.primaryHashTokens = numTokens;
    }
    const k = hashKeyString(key);
    this.cached.set(k, [...(this.cached.get(k) ?? []), block.id]);
    this.emit({ kind: 'hash', req: ctx.req, group: ctx.group, blockId: block.id, hash: key.hash, numTokens, partial, secondary });
  }

  private removeHashes(block: Block, ctx: PoolContext, reason: 'reuse' | 'promote'): void {
    const keys = [...(block.primaryHash ? [block.primaryHash] : []), ...block.extraHashes];
    for (const key of keys) {
      const k = hashKeyString(key);
      const ids = this.cached.get(k);
      if (!ids || !ids.includes(block.id)) continue;
      const rest = ids.filter((id) => id !== block.id);
      if (rest.length > 0) this.cached.set(k, rest);
      else this.cached.delete(k);
      this.emit({ kind: 'evict', req: ctx.req, group: ctx.group, blockId: block.id, hash: key.hash, hashGroup: key.group, reason });
    }
    block.primaryHash = null;
    block.primaryHashTokens = null;
    block.extraHashes = [];
  }

  cacheFullBlocks(
    blocks: readonly Block[],
    groupHashes: readonly string[],
    numCachedBlocks: number,
    numFullBlocks: number,
    blockSize: number,
    mask: readonly boolean[] | null,
    ctx: PoolContext,
  ): void {
    for (let i = numCachedBlocks; i < numFullBlocks; i++) {
      const block = blocks[i];
      if (block.isNull || (mask !== null && !mask[i - numCachedBlocks])) continue;
      if (block.primaryHash !== null) this.removeHashes(block, ctx, 'promote');
      this.insertHash({ hash: groupHashes[i], group: ctx.group }, block, (i + 1) * blockSize, ctx, false);
    }
  }

  cachePartialBlock(block: Block, hash: string, numTokens: number, ctx: PoolContext): void {
    if (block.isNull) return;
    const key = { hash, group: ctx.group };
    const alreadyCached = sameKey(block.primaryHash, key) || this.contains(key, block.id);
    if (!alreadyCached && block.primaryHash !== null && block.primaryHashTokens !== null && block.primaryHashTokens < numTokens) this.removeHashes(block, ctx, 'promote');
    this.insertHash(key, block, numTokens, ctx, true);
  }

  getNewBlocks(n: number, ctx: PoolContext): Block[] {
    if (n > this.numFreeBlocks) throw new Error(`Cannot get ${n} free blocks from the pool`);
    const ids = this.freeQueue.slice(0, n);
    this.freeQueue = this.freeQueue.slice(n);
    const out = ids.map((id) => this.blocks[id]);
    for (const block of out) {
      if (this.enableCaching) this.removeHashes(block, ctx, 'reuse');
      block.refCnt += 1;
    }
    if (out.length > 0) this.emit({ kind: 'allocate', req: ctx.req, group: ctx.group, blockIds: ids });
    return out;
  }

  touch(blocks: readonly Block[], ctx: PoolContext): void {
    const revived: number[] = [];
    for (const block of blocks) {
      if (block.isNull) continue;
      if (block.refCnt === 0) revived.push(block.id);
      block.refCnt += 1;
    }
    if (revived.length === 0) return;
    const gone = new Set(revived);
    this.freeQueue = this.freeQueue.filter((id) => !gone.has(id));
    this.emit({ kind: 'touch', req: ctx.req, group: ctx.group, blockIds: revived });
  }

  freeBlocks(ordered: readonly Block[], ctx: PoolContext, reason: 'finish' | 'preempt' | 'out_of_window'): void {
    const toHead: number[] = [];
    const toTail: number[] = [];
    const real = ordered.filter((b) => !b.isNull);
    for (const block of real) {
      block.refCnt -= 1;
      if (block.refCnt > 0) continue;
      if (block.primaryHash === null || !this.enableCaching) toHead.push(block.id);
      else toTail.push(block.id);
    }
    this.freeQueue = [...toHead, ...this.freeQueue, ...toTail];
    if (real.length > 0) this.emit({ kind: 'free', req: ctx.req, group: ctx.group, reason, blockIds: real.map((b) => b.id), toHead, toTail });
  }
}
