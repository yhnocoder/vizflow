import { BlockPool, type Block, type VllmEvent } from './blockPool'

export type KvGroupSpec =
  | { readonly kind: 'full'; readonly blockSize: number }
  | { readonly kind: 'sliding_window'; readonly blockSize: number; readonly window: number }
  | { readonly kind: 'compressed'; readonly tokensPerBlock: number; readonly ratio: number }

export interface KvRequest {
  readonly id: string
  readonly numTokens: number
  readonly numPromptTokens: number
  readonly numComputedTokens: number
  readonly sharedPrefixBoundary: number
  blockHashes(): readonly string[]
}

export interface KvCacheConfig {
  readonly numGpuBlocks: number
  readonly groups: readonly KvGroupSpec[]
  readonly maxModelLen: number
  readonly maxInFlightTokens: number
  readonly enablePrefixCaching: boolean
}

export interface CacheHit {
  readonly blocks: readonly (readonly Block[])[]
  readonly numTokens: number
  readonly sharedPrefixBoundary: number
}

export function groupTokensPerBlock(spec: KvGroupSpec): number {
  return spec.kind === 'compressed' ? spec.tokensPerBlock : spec.blockSize
}

export function groupEntriesPerBlock(spec: KvGroupSpec): number {
  return spec.kind === 'compressed' ? spec.tokensPerBlock / spec.ratio : spec.blockSize
}

export function groupEntriesForTokens(spec: KvGroupSpec, tokens: number): number {
  return spec.kind === 'compressed' ? Math.floor(tokens / spec.ratio) : tokens
}

function isFullAttention(spec: KvGroupSpec): boolean {
  return spec.kind !== 'sliding_window'
}

function cdiv(a: number, b: number): number {
  return Math.ceil(a / b)
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b)
}

function lcm(a: number, b: number): number {
  return (a / gcd(a, b)) * b
}

function hashesAtBlockSize(hashes: readonly string[], hashBlockSize: number, blockSize: number): string[] {
  const scale = blockSize / hashBlockSize
  return Array.from({ length: Math.floor(hashes.length / scale) }, (_, i) => hashes[(i + 1) * scale - 1]!)
}

function swaContiguousBlocksForHit(window: number, blockSize: number): number {
  return cdiv(window - 1, blockSize)
}

function validateSpec(spec: KvGroupSpec): void {
  const size = groupTokensPerBlock(spec)
  if (!Number.isInteger(size) || size <= 0) throw new Error(`group block size must be a positive integer, got ${size}`)
  if (spec.kind === 'sliding_window' && !(Number.isInteger(spec.window) && spec.window > 0)) {
    throw new Error('sliding window must be a positive integer')
  }
  if (spec.kind === 'compressed' && !(Number.isInteger(spec.ratio) && spec.ratio > 0 && spec.tokensPerBlock % spec.ratio === 0)) {
    throw new Error('compressed group needs a positive integer ratio dividing tokensPerBlock')
  }
}

class GroupManager {
  readonly blockSize: number
  readonly reqToBlocks = new Map<string, Block[]>()
  readonly numCachedBlock = new Map<string, number>()
  cacheHitAlignmentTokens: number

  constructor(
    readonly spec: KvGroupSpec,
    readonly groupId: number,
    readonly pool: BlockPool,
    readonly maxAdmissionBlocksPerRequest: number | null,
    schedulerBlockSize: number,
  ) {
    this.blockSize = groupTokensPerBlock(spec)
    this.cacheHitAlignmentTokens = schedulerBlockSize
  }

  private ctx(req: string) {
    return { req, group: this.groupId }
  }

  numSkippedTokens(numComputedTokens: number): number {
    if (this.spec.kind !== 'sliding_window') return 0
    return Math.max(0, numComputedTokens - this.spec.window + 1)
  }

  getNumBlocksToAllocate(
    reqId: string,
    numTokens: number,
    newComputed: readonly Block[],
    totalComputedTokens: number,
    applyAdmissionCap: boolean,
  ): number {
    let numRequired = cdiv(numTokens, this.blockSize)
    if (applyAdmissionCap && this.maxAdmissionBlocksPerRequest !== null) {
      numRequired = Math.min(numRequired, this.maxAdmissionBlocksPerRequest)
    }
    const numReqBlocks = this.reqToBlocks.get(reqId)?.length ?? 0
    if (this.numCachedBlock.has(reqId)) return Math.max(numRequired - numReqBlocks, 0)
    const numSkippedBlocks = Math.floor(this.numSkippedTokens(totalComputedTokens) / this.blockSize)
    const numLocalComputedBlocks = newComputed.length + numReqBlocks
    const numNew = Math.max(numRequired - Math.max(numSkippedBlocks, numLocalComputedBlocks), 0)
    const skipNew = Math.max(0, numSkippedBlocks - numReqBlocks)
    const numEvictable = newComputed.slice(skipNew).filter((b) => b.refCnt === 0 && !b.isNull).length
    return numNew + numEvictable
  }

  addLocalComputedBlocks(reqId: string, newComputed: readonly Block[], numLocalComputedTokens: number): void {
    const numSkippedBlocks = Math.floor(this.numSkippedTokens(numLocalComputedTokens) / this.blockSize)
    const kept = numSkippedBlocks > 0 ? newComputed.slice(numSkippedBlocks) : newComputed
    if (this.pool.enableCaching) this.pool.touch(kept, this.ctx(reqId))
    const blocks = [...Array<Block>(numSkippedBlocks).fill(this.pool.nullBlock), ...kept]
    this.reqToBlocks.set(reqId, blocks)
    this.numCachedBlock.set(reqId, blocks.length)
  }

  allocateNewBlocks(reqId: string, numTokens: number): Block[] {
    const blocks = this.reqToBlocks.get(reqId) ?? []
    this.reqToBlocks.set(reqId, blocks)
    const numNew = cdiv(numTokens, this.blockSize) - blocks.length
    if (numNew <= 0) return []
    const fresh = this.pool.getNewBlocks(numNew, this.ctx(reqId))
    blocks.push(...fresh)
    return fresh
  }

  private reachableBlockMask(startBlock: number, endBlock: number, boundaries: readonly number[]): boolean[] | null {
    if (this.spec.kind !== 'sliding_window') return null
    const alignment = this.cacheHitAlignmentTokens
    if (alignment % this.blockSize !== 0) return null
    const need = swaContiguousBlocksForHit(this.spec.window, this.blockSize)
    const mask = Array<boolean>(endBlock - startBlock).fill(false)
    for (const boundary of boundaries) {
      const aligned = Math.floor(boundary / alignment) * alignment
      const end = aligned / this.blockSize
      for (let j = Math.max(startBlock, end - need); j < Math.min(endBlock, end); j++) mask[j - startBlock] = true
    }
    return mask
  }

  cacheBlocks(request: KvRequest, numTokens: number, replayBoundaries: readonly number[]): void {
    const numCached = this.numCachedBlock.get(request.id) ?? 0
    const numFull = Math.floor(numTokens / this.blockSize)
    if (numCached < numFull) {
      const boundaries = request.sharedPrefixBoundary ? [...replayBoundaries, request.sharedPrefixBoundary] : replayBoundaries
      const mask = this.reachableBlockMask(numCached, numFull, boundaries)
      const groupHashes = hashesAtBlockSize(request.blockHashes(), this.pool.hashBlockSize, this.blockSize)
      this.pool.cacheFullBlocks(this.reqToBlocks.get(request.id) ?? [], groupHashes, numCached, numFull, this.blockSize, mask, this.ctx(request.id))
      this.numCachedBlock.set(request.id, numFull)
    }
    if (isFullAttention(this.spec) && this.blockSize !== this.pool.hashBlockSize) this.cachePartialTailBlock(request, numTokens)
  }

  private cachePartialTailBlock(request: KvRequest, numTokens: number): void {
    const hashBlockSize = this.pool.hashBlockSize
    const boundary = Math.floor((request.numPromptTokens - 1) / hashBlockSize) * hashBlockSize
    if (boundary === 0 || boundary > numTokens) return
    if (boundary % this.blockSize === 0) return
    const blocks = this.reqToBlocks.get(request.id) ?? []
    const index = Math.floor(boundary / this.blockSize)
    if (index >= blocks.length) return
    const hash = request.blockHashes()[boundary / hashBlockSize - 1]!
    this.pool.cachePartialBlock(blocks[index]!, hash, boundary, this.ctx(request.id))
  }

  free(reqId: string, reason: 'finish' | 'preempt'): void {
    const blocks = this.reqToBlocks.get(reqId) ?? []
    this.reqToBlocks.delete(reqId)
    this.numCachedBlock.delete(reqId)
    this.pool.freeBlocks([...blocks].reverse(), this.ctx(reqId), reason)
  }

  removeSkippedBlocks(reqId: string, processedComputedTokens: number): void {
    const numSkippedTokens = this.numSkippedTokens(processedComputedTokens)
    if (numSkippedTokens <= 0) return
    const blocks = this.reqToBlocks.get(reqId)
    if (!blocks) return
    const last = Math.min(Math.floor(numSkippedTokens / this.blockSize), blocks.length)
    const freed: Block[] = []
    for (let i = last - 1; i >= 0; i--) {
      if (blocks[i]!.isNull) break
      freed.push(blocks[i]!)
      blocks[i] = this.pool.nullBlock
    }
    if (freed.length > 0) this.pool.freeBlocks(freed, this.ctx(reqId), 'out_of_window')
  }

  findLongestCacheHit(hashes: readonly string[], maxLength: number, groupIds: readonly number[], alignment: number): { blocks: Block[][]; length: number } {
    const view = hashesAtBlockSize(hashes, this.pool.hashBlockSize, this.blockSize)
    const bs = this.blockSize
    if (this.spec.kind !== 'sliding_window') {
      const computed: Block[][] = groupIds.map(() => [])
      const limit = Math.min(Math.floor(maxLength / bs), view.length)
      for (let i = 0; i < limit; i++) {
        const cached = this.pool.getCachedBlock(view[i]!, groupIds)
        if (!cached) break
        cached.forEach((b, g) => computed[g]!.push(b))
      }
      let length = computed[0]!.length * bs
      length -= length % alignment
      const numBlocks = cdiv(length, bs)
      return { blocks: computed.map((c) => c.slice(0, numBlocks)), length }
    }
    const need = swaContiguousBlocksForHit(this.spec.window, bs)
    const maxNumBlocks = Math.floor(maxLength / bs)
    let computed: Block[][] = groupIds.map(() => Array<Block>(maxNumBlocks).fill(this.pool.nullBlock))
    let contiguous = 0
    let matchFound = false
    for (let i = maxNumBlocks - 1; i >= 0; i--) {
      const cached = i < view.length ? this.pool.getCachedBlock(view[i]!, groupIds) : null
      if (cached) {
        if (contiguous === 0 && bs !== alignment && ((i + 1) * bs) % alignment !== 0) continue
        cached.forEach((b, g) => {
          computed[g]![i] = b
        })
        contiguous += 1
        if (contiguous >= need) {
          computed = computed.map((c) => c.slice(0, i + contiguous))
          matchFound = true
          break
        }
      } else {
        contiguous = 0
      }
    }
    if (!matchFound) {
      computed = computed.map((c) => c.slice(0, contiguous))
      while (bs !== alignment && (computed[0]!.length * bs) % alignment !== 0) computed = computed.map((c) => c.slice(0, -1))
    }
    return { blocks: computed, length: computed[0]!.length * bs }
  }
}

interface AttentionGroup {
  readonly spec: KvGroupSpec
  readonly groupIds: number[]
}

export class KvCacheManager {
  readonly pool: BlockPool
  readonly managers: readonly GroupManager[]
  readonly schedulerBlockSize: number
  readonly hashBlockSize: number
  private readonly attentionGroups: readonly AttentionGroup[]

  constructor(readonly config: KvCacheConfig, emit: (event: VllmEvent) => void) {
    if (config.groups.length === 0) throw new Error('at least one KV cache group is required')
    config.groups.forEach(validateSpec)
    const sizes = config.groups.map(groupTokensPerBlock)
    this.schedulerBlockSize = sizes.reduce(lcm)
    this.hashBlockSize = sizes.length === 1 || !config.enablePrefixCaching ? this.schedulerBlockSize : sizes.reduce(gcd)
    for (const spec of config.groups) {
      if (spec.kind === 'compressed' && this.schedulerBlockSize % spec.ratio !== 0) {
        throw new Error('prefix-cache boundaries must align with each compressed group ratio')
      }
    }
    this.pool = new BlockPool(config.numGpuBlocks, config.enablePrefixCaching, this.hashBlockSize, emit)
    this.managers = config.groups.map((spec, i) => {
      const cap =
        spec.kind === 'sliding_window'
          ? cdiv(Math.min(spec.window - 1 + config.maxInFlightTokens, config.maxModelLen), spec.blockSize) + 1
          : null
      return new GroupManager(spec, i, this.pool, cap, this.schedulerBlockSize)
    })
    const groups: AttentionGroup[] = []
    config.groups.forEach((spec, i) => {
      const same = groups.find((g) => JSON.stringify(g.spec) === JSON.stringify(spec))
      if (same) same.groupIds.push(i)
      else groups.push({ spec, groupIds: [i] })
    })
    this.attentionGroups = [...groups.filter((g) => isFullAttention(g.spec)), ...groups.filter((g) => !isFullAttention(g.spec))]
  }

  private get isHybrid(): boolean {
    return this.managers.length > 1
  }

  getBlocks(reqId: string): readonly (readonly Block[])[] {
    return this.managers.map((m) => m.reqToBlocks.get(reqId) ?? [])
  }

  getComputedBlocks(request: KvRequest): CacheHit {
    const empty = { blocks: this.managers.map(() => []), numTokens: 0, sharedPrefixBoundary: 0 }
    if (!this.config.enablePrefixCaching) return empty
    const maxLength = request.numTokens - 1
    const hashes = request.blockHashes()
    if (!this.isHybrid) {
      const hit = this.managers[0]!.findLongestCacheHit(hashes, maxLength, [0], this.managers[0]!.blockSize)
      return { blocks: hit.blocks, numTokens: hit.length, sharedPrefixBoundary: 0 }
    }
    return this.findHybridCacheHit(hashes, maxLength)
  }

  private findHybridCacheHit(hashes: readonly string[], maxLength: number): CacheHit {
    const n = this.managers.length
    let hitLength = maxLength
    let longest = 0
    const blocksByGroup: (Block[] | null)[] = Array(n).fill(null)
    const lengthByGroup: number[] = Array(n).fill(0)
    const isSimpleHybrid = this.attentionGroups.length === 2 && isFullAttention(this.attentionGroups[0]!.spec)
    for (;;) {
      let current = hitLength
      for (const group of this.attentionGroups) {
        const first = group.groupIds[0]!
        if (isFullAttention(group.spec) && blocksByGroup[first] !== null) {
          current = Math.min(current, lengthByGroup[first]!)
          continue
        }
        const hit = this.managers[first]!.findLongestCacheHit(hashes, current, group.groupIds, this.schedulerBlockSize)
        current = hit.length
        group.groupIds.forEach((gid, k) => {
          blocksByGroup[gid] = hit.blocks[k]!
          lengthByGroup[gid] = hit.length
        })
        longest = Math.max(longest, current)
      }
      if (current >= hitLength) break
      hitLength = current
      if (isSimpleHybrid) break
    }
    for (const group of this.attentionGroups) {
      if (!isFullAttention(group.spec)) continue
      const numBlocks = cdiv(hitLength, this.managers[group.groupIds[0]!]!.blockSize)
      for (const gid of group.groupIds) {
        const blocks = blocksByGroup[gid]
        if (blocks) blocksByGroup[gid] = blocks.slice(0, numBlocks)
      }
    }
    const uncached = longest - hitLength
    return {
      blocks: blocksByGroup.map((b) => b ?? []),
      numTokens: hitLength,
      sharedPrefixBoundary: uncached ? hitLength + uncached : 0,
    }
  }

  private numBlocksToAllocate(
    reqId: string,
    numTokens: number,
    newComputed: readonly (readonly Block[])[],
    totalComputedTokens: number,
    applyAdmissionCap: boolean,
  ): number {
    return this.managers.reduce(
      (sum, m, i) => sum + m.getNumBlocksToAllocate(reqId, numTokens, newComputed[i]!, totalComputedTokens, applyAdmissionCap),
      0,
    )
  }

  private cacheableTokens(numComputedTokens: number): number {
    return this.isHybrid ? Math.floor(numComputedTokens / this.schedulerBlockSize) * this.schedulerBlockSize : numComputedTokens
  }

  allocateSlots(
    request: KvRequest,
    numNewTokens: number,
    hit: CacheHit | null,
    fullSequenceMustFit: boolean,
  ): readonly (readonly Block[])[] | null {
    const newComputed = hit?.blocks ?? this.managers.map(() => [])
    const numNewComputedTokens = hit?.numTokens ?? 0
    const numLocalComputed = request.numComputedTokens + numNewComputedTokens
    const totalComputed = Math.min(numLocalComputed, this.config.maxModelLen)
    if (fullSequenceMustFit) {
      const fullTokens = Math.min(request.numTokens, this.config.maxModelLen)
      if (this.numBlocksToAllocate(request.id, fullTokens, newComputed, totalComputed, true) > this.pool.numFreeBlocks) return null
    }
    const numTokensMainModel = totalComputed + numNewTokens
    const numTokensNeedSlot = Math.min(numTokensMainModel, this.config.maxModelLen)
    for (const m of this.managers) m.removeSkippedBlocks(request.id, totalComputed)
    if (this.numBlocksToAllocate(request.id, numTokensNeedSlot, newComputed, numLocalComputed, false) > this.pool.numFreeBlocks) return null
    if (newComputed.some((b) => b.length > 0) && !this.managers.some((m) => m.numCachedBlock.has(request.id))) {
      this.managers.forEach((m, i) => m.addLocalComputedBlocks(request.id, newComputed[i]!, numLocalComputed))
    }
    const fresh = this.managers.map((m) => m.allocateNewBlocks(request.id, numTokensNeedSlot))
    if (this.config.enablePrefixCaching) {
      const numToCache = this.cacheableTokens(Math.min(totalComputed + numNewTokens, request.numTokens))
      for (const m of this.managers) m.cacheBlocks(request, numToCache, [request.numPromptTokens - 1])
    }
    return fresh
  }

  free(reqId: string, reason: 'finish' | 'preempt'): void {
    for (const m of this.managers) m.free(reqId, reason)
  }
}
