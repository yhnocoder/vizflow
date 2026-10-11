import type { Block, VllmEvent } from './blockPool'
import { chainBlockHashes, type Token } from './hash'
import {
  KvCacheManager,
  groupEntriesForTokens,
  groupEntriesPerBlock,
  groupTokensPerBlock,
  type CacheHit,
  type KvGroupSpec,
  type KvRequest,
} from './kvCacheManager'

export interface VllmConfig {
  readonly numGpuBlocks: number
  readonly groups: readonly KvGroupSpec[]
  readonly maxNumBatchedTokens: number
  readonly maxNumSeqs: number
  readonly maxModelLen: number
  readonly enableChunkedPrefill?: boolean
  readonly enablePrefixCaching?: boolean
}

export interface VllmRequestSpec {
  readonly id: string
  readonly prompt: readonly Token[]
  readonly outputs: readonly Token[]
  readonly maxTokens: number
  readonly arrival: number
  readonly cacheSalt?: string
}

export type VllmRequestStatus = 'pending' | 'waiting' | 'running' | 'preempted' | 'finished'

export interface VllmBlockRef {
  readonly req: string
  readonly group: number
  readonly index: number
}

export interface VllmBlockView {
  readonly id: number
  readonly isNull: boolean
  readonly refCnt: number
  readonly hash: string | null
  readonly hashGroup: number | null
  readonly hashTokens: number | null
  readonly extraHashes: readonly string[]
  readonly refs: readonly VllmBlockRef[]
  readonly group: number | null
  readonly filled: number
  readonly capacity: number | null
}

export interface VllmRequestView {
  readonly id: string
  readonly status: VllmRequestStatus
  readonly numPromptTokens: number
  readonly numTokens: number
  readonly numOutputTokens: number
  readonly numComputedTokens: number
  readonly blockTables: readonly (readonly number[])[]
  readonly prefixHitTokens: number | null
  readonly numPreemptions: number
}

export interface VllmSnapshot {
  readonly step: number
  readonly schedulerBlockSize: number
  readonly hashBlockSize: number
  readonly blocks: readonly VllmBlockView[]
  readonly freeQueue: readonly number[]
  readonly waiting: readonly string[]
  readonly running: readonly string[]
  readonly scheduled: readonly { readonly req: string; readonly numTokens: number }[]
  readonly requests: readonly VllmRequestView[]
  readonly events: readonly VllmEvent[]
}

class SimRequest implements KvRequest {
  status: VllmRequestStatus = 'pending'
  numComputedTokens = 0
  sharedPrefixBoundary = 0
  prefixHitTokens: number | null = null
  numPreemptions = 0
  readonly tokens: Token[]
  private hashes: string[] = []

  constructor(
    readonly spec: VllmRequestSpec,
    private readonly hashBlockSize: number,
  ) {
    this.tokens = [...spec.prompt]
  }

  get id(): string {
    return this.spec.id
  }

  get numTokens(): number {
    return this.tokens.length
  }

  get numPromptTokens(): number {
    return this.spec.prompt.length
  }

  get numOutputTokens(): number {
    return this.tokens.length - this.spec.prompt.length
  }

  blockHashes(): readonly string[] {
    this.hashes = chainBlockHashes(this.tokens, this.hashBlockSize, this.spec.cacheSalt, this.hashes)
    return this.hashes
  }
}

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value)
    for (const v of Object.values(value)) deepFreeze(v)
  }
  return value
}

export class VllmEngine {
  readonly kv: KvCacheManager
  private readonly requests: SimRequest[]
  private waiting: SimRequest[] = []
  private running: SimRequest[] = []
  private events: VllmEvent[] = []
  private stepIndex = 0
  private readonly enableChunkedPrefill: boolean

  constructor(
    readonly config: VllmConfig,
    specs: readonly VllmRequestSpec[],
  ) {
    const enablePrefixCaching = config.enablePrefixCaching ?? true
    this.enableChunkedPrefill = config.enableChunkedPrefill ?? true
    this.kv = new KvCacheManager(
      {
        numGpuBlocks: config.numGpuBlocks,
        groups: config.groups,
        maxModelLen: config.maxModelLen,
        maxInFlightTokens: config.maxNumBatchedTokens,
        enablePrefixCaching,
      },
      (e) => this.events.push(e),
    )
    const ids = new Set<string>()
    for (const spec of specs) {
      if (ids.has(spec.id)) throw new Error(`duplicate request id ${spec.id}`)
      ids.add(spec.id)
      if (spec.prompt.length === 0) throw new Error(`request ${spec.id} has an empty prompt`)
      if (spec.outputs.length < spec.maxTokens) throw new Error(`request ${spec.id} needs at least maxTokens output tokens`)
    }
    this.requests = specs.map((s) => new SimRequest(s, this.kv.hashBlockSize))
  }

  get done(): boolean {
    return this.requests.every((r) => r.status === 'finished')
  }

  private free(request: SimRequest, reason: 'finish' | 'preempt'): void {
    this.kv.free(request.id, reason)
  }

  private preempt(victim: SimRequest, forReq: string): void {
    this.free(victim, 'preempt')
    victim.status = 'preempted'
    victim.numComputedTokens = 0
    victim.numPreemptions += 1
    this.waiting.unshift(victim)
    this.events.push({ kind: 'preempt', req: victim.id, for: forReq })
  }

  private hitEvents(request: SimRequest, hit: CacheHit): VllmEvent[] {
    return hit.blocks.flatMap((blocks, group): VllmEvent[] => {
      const ids = blocks.filter((b) => !b.isNull).map((b) => b.id)
      return ids.length > 0 ? [{ kind: 'hit', req: request.id, group, blockIds: ids }] : []
    })
  }

  step(): VllmSnapshot {
    this.events = []
    for (const r of this.requests) {
      if (r.status === 'pending' && r.spec.arrival <= this.stepIndex) {
        r.status = 'waiting'
        this.waiting.push(r)
        this.events.push({ kind: 'arrive', req: r.id })
      }
    }
    const scheduled: { req: SimRequest; numTokens: number }[] = []
    let tokenBudget = this.config.maxNumBatchedTokens
    let preempted = false
    let index = 0
    while (index < this.running.length && tokenBudget > 0) {
      const request = this.running[index]!
      let numNew = Math.min(request.numTokens - request.numComputedTokens, tokenBudget)
      numNew = Math.min(numNew, this.config.maxModelLen - request.numComputedTokens - 1)
      if (numNew <= 0) {
        index += 1
        continue
      }
      let blocks: readonly (readonly Block[])[] | null = null
      for (;;) {
        blocks = this.kv.allocateSlots(request, numNew, null, false)
        if (blocks !== null) break
        const victim = this.running.pop()!
        this.preempt(victim, request.id)
        preempted = true
        if (victim === request) break
      }
      if (blocks === null) break
      scheduled.push({ req: request, numTokens: numNew })
      tokenBudget -= numNew
      index += 1
    }
    if (!preempted) {
      while (tokenBudget > 0 && this.waiting.length > 0 && this.running.length < this.config.maxNumSeqs) {
        const request = this.waiting[0]!
        const hit = this.kv.getComputedBlocks(request)
        request.sharedPrefixBoundary = hit.sharedPrefixBoundary
        const numComputed = hit.numTokens
        let numNew = request.numTokens - numComputed
        if (!this.enableChunkedPrefill && numNew > tokenBudget) break
        numNew = Math.min(numNew, tokenBudget)
        const mark = this.events.length
        const blocks = this.kv.allocateSlots(request, numNew, hit, true)
        if (blocks === null) break
        this.waiting.shift()
        this.running.push(request)
        const resumed = request.status === 'preempted'
        request.status = 'running'
        request.numComputedTokens = numComputed
        request.prefixHitTokens = numComputed
        this.events.splice(mark, 0, { kind: 'admit', req: request.id, resumed, hitTokens: numComputed, numNewTokens: numNew }, ...this.hitEvents(request, hit))
        scheduled.push({ req: request, numTokens: numNew })
        tokenBudget -= numNew
      }
    }
    for (const s of scheduled) {
      this.events.push({ kind: 'schedule', req: s.req.id, numTokens: s.numTokens, numComputedTokens: s.req.numComputedTokens })
      s.req.numComputedTokens += s.numTokens
    }
    const stopped = new Set<SimRequest>()
    for (const { req } of scheduled) {
      if (req.numComputedTokens < req.numTokens) continue
      req.tokens.push(req.spec.outputs[req.numOutputTokens]!)
      const reason = req.numOutputTokens >= req.spec.maxTokens ? 'length' : req.numTokens >= this.config.maxModelLen ? 'max_model_len' : null
      if (reason === null) continue
      req.status = 'finished'
      this.free(req, 'finish')
      this.events.push({ kind: 'finish', req: req.id, reason })
      stopped.add(req)
    }
    this.running = this.running.filter((r) => !stopped.has(r))
    const snapshot = this.snapshot(scheduled.map((s) => ({ req: s.req.id, numTokens: s.numTokens })))
    this.stepIndex += 1
    return snapshot
  }

  private snapshot(scheduled: { req: string; numTokens: number }[]): VllmSnapshot {
    const groups = this.config.groups
    const refs = new Map<number, VllmBlockRef[]>()
    const filled = new Map<number, number>()
    for (const r of this.requests) {
      this.kv.getBlocks(r.id).forEach((blocks, group) => {
        const spec = groups[group]!
        const tokensPerBlock = groupTokensPerBlock(spec)
        blocks.forEach((b, index) => {
          if (b.isNull) return
          refs.set(b.id, [...(refs.get(b.id) ?? []), { req: r.id, group, index }])
          const covered = Math.max(0, Math.min(r.numComputedTokens - index * tokensPerBlock, tokensPerBlock))
          filled.set(b.id, Math.max(filled.get(b.id) ?? 0, groupEntriesForTokens(spec, covered)))
        })
      })
    }
    const blocks = this.kv.pool.blocks.map((b): VllmBlockView => {
      const blockRefs = refs.get(b.id) ?? []
      const group = blockRefs[0]?.group ?? b.primaryHash?.group ?? null
      const spec = group === null ? null : groups[group]!
      let fill = filled.get(b.id) ?? 0
      if (spec && blockRefs.length === 0 && b.primaryHashTokens !== null) {
        const tokensPerBlock = groupTokensPerBlock(spec)
        fill = groupEntriesForTokens(spec, b.primaryHashTokens % tokensPerBlock || tokensPerBlock)
      }
      return {
        id: b.id,
        isNull: b.isNull,
        refCnt: b.isNull ? 0 : b.refCnt,
        hash: b.primaryHash?.hash ?? null,
        hashGroup: b.primaryHash?.group ?? null,
        hashTokens: b.primaryHashTokens,
        extraHashes: b.extraHashes.map((k) => k.hash),
        refs: blockRefs,
        group,
        filled: fill,
        capacity: spec ? groupEntriesPerBlock(spec) : null,
      }
    })
    return deepFreeze({
      step: this.stepIndex,
      schedulerBlockSize: this.kv.schedulerBlockSize,
      hashBlockSize: this.kv.hashBlockSize,
      blocks,
      freeQueue: [...this.kv.pool.freeQueue],
      waiting: this.waiting.map((r) => r.id),
      running: this.running.map((r) => r.id),
      scheduled,
      requests: this.requests.map((r) => ({
        id: r.id,
        status: r.status,
        numPromptTokens: r.numPromptTokens,
        numTokens: r.numTokens,
        numOutputTokens: r.numOutputTokens,
        numComputedTokens: r.numComputedTokens,
        blockTables: this.kv.getBlocks(r.id).map((bs) => bs.map((b) => b.id)),
        prefixHitTokens: r.prefixHitTokens,
        numPreemptions: r.numPreemptions,
      })),
      events: this.events,
    })
  }
}

export function simulateVllm(config: VllmConfig, requests: readonly VllmRequestSpec[]): VllmSnapshot[] {
  const engine = new VllmEngine(config, requests)
  const snapshots: VllmSnapshot[] = []
  const lastArrival = Math.max(0, ...requests.map((r) => r.arrival))
  while (!engine.done) {
    const snap = engine.step()
    snapshots.push(snap)
    if (snap.scheduled.length === 0 && snap.step >= lastArrival) {
      throw new Error(`step ${snap.step}: nothing could be scheduled and no request is still to arrive`)
    }
  }
  return snapshots
}
