import type { Token } from './hash'

export type RadixEvent =
  | { readonly kind: 'match'; readonly tokens: number; readonly matched: number; readonly lastNode: number }
  | { readonly kind: 'split'; readonly node: number; readonly newNode: number; readonly at: number }
  | { readonly kind: 'insert'; readonly node: number; readonly parent: number; readonly tokens: number }
  | { readonly kind: 'lock'; readonly node: number; readonly path: readonly number[] }
  | { readonly kind: 'unlock'; readonly node: number; readonly path: readonly number[] }
  | { readonly kind: 'evict'; readonly node: number; readonly slots: readonly number[] }
  | { readonly kind: 'alloc'; readonly req: string | null; readonly slots: readonly number[] }
  | { readonly kind: 'free'; readonly req: string | null; readonly slots: readonly number[] }

export interface RadixNodeView {
  readonly id: number
  readonly parent: number | null
  readonly key: readonly Token[]
  readonly slots: readonly number[]
  readonly lockRef: number
  readonly lastAccess: number
  readonly children: readonly number[]
  readonly evictable: boolean
}

export interface RadixRequestView {
  readonly id: string
  readonly tokens: readonly Token[]
  readonly slots: readonly number[]
  readonly cacheProtectedLen: number
  readonly lastNode: number
}

export interface RadixSnapshot {
  readonly clock: number
  readonly nodes: readonly RadixNodeView[]
  readonly freeSlots: readonly number[]
  readonly releasedSlots: readonly number[]
  readonly evictableSize: number
  readonly protectedSize: number
  readonly requests: readonly RadixRequestView[]
  readonly events: readonly RadixEvent[]
}

export interface RadixMatch {
  readonly prefixLength: number
  readonly slots: readonly number[]
  readonly lastNode: number
}

interface TreeNode {
  readonly id: number
  parent: TreeNode | null
  key: Token[]
  value: number[]
  children: Map<Token, TreeNode>
  lockRef: number
  lastAccess: number
}

interface ReqState {
  readonly id: string
  readonly promptLength: number
  tokens: Token[]
  slots: number[]
  cacheProtectedLen: number
  lock: TreeNode
}

function matchLength(a: readonly Token[], b: readonly Token[]): number {
  const n = Math.min(a.length, b.length)
  let i = 0
  while (i < n && a[i] === b[i]) i++
  return i
}

export class RadixCacheSim {
  private clock = 0
  private nextId = 0
  private readonly root: TreeNode
  private readonly evictableLeaves = new Set<TreeNode>()
  private evictableSize = 0
  private protectedSize = 0
  private freePages: number[]
  private releasePages: number[] = []
  private readonly reqs = new Map<string, ReqState>()
  private events: RadixEvent[] = []

  constructor(
    readonly numSlots: number,
    readonly needSort = false,
  ) {
    if (!Number.isInteger(numSlots) || numSlots < 1) throw new Error('numSlots must be a positive integer')
    this.freePages = Array.from({ length: numSlots }, (_, i) => i + 1)
    this.root = this.newNode(null, [], [])
    this.root.lockRef = 1
  }

  private tick(): number {
    this.clock += 1
    return this.clock
  }

  private newNode(parent: TreeNode | null, key: Token[], value: number[]): TreeNode {
    return { id: this.nextId++, parent, key, value, children: new Map(), lockRef: 0, lastAccess: this.tick() }
  }

  private findNode(id: number): TreeNode {
    const stack = [this.root]
    while (stack.length > 0) {
      const node = stack.pop()!
      if (node.id === id) return node
      stack.push(...node.children.values())
    }
    throw new Error(`no tree node ${id}`)
  }

  takeEvents(): RadixEvent[] {
    const out = this.events
    this.events = []
    return out
  }

  availableSize(): number {
    return this.freePages.length + this.releasePages.length
  }

  alloc(n: number, req: string | null = null): number[] | null {
    if (this.needSort && n > this.freePages.length && this.releasePages.length > 0) {
      this.freePages = [...this.freePages, ...this.releasePages].sort((a, b) => a - b)
      this.releasePages = []
    }
    if (n > this.freePages.length) return null
    const out = this.freePages.slice(0, n)
    this.freePages = this.freePages.slice(n)
    if (out.length > 0) this.events.push({ kind: 'alloc', req, slots: out })
    return out
  }

  free(slots: readonly number[], req: string | null = null): void {
    if (slots.length === 0) return
    if (this.needSort) this.releasePages = [...this.releasePages, ...slots]
    else this.freePages = [...this.freePages, ...slots]
    this.events.push({ kind: 'free', req, slots: [...slots] })
  }

  private splitNode(child: TreeNode, splitLen: number): TreeNode {
    const parent = child.parent!
    const newNode = this.newNode(parent, child.key.slice(0, splitLen), child.value.slice(0, splitLen))
    newNode.lockRef = child.lockRef
    newNode.children.set(child.key[splitLen]!, child)
    child.parent = newNode
    child.key = child.key.slice(splitLen)
    child.value = child.value.slice(splitLen)
    parent.children.set(newNode.key[0]!, newNode)
    this.events.push({ kind: 'split', node: child.id, newNode: newNode.id, at: splitLen })
    return newNode
  }

  private matchHelper(key: readonly Token[]): { value: number[]; node: TreeNode } {
    const accessTime = this.tick()
    let node = this.root
    node.lastAccess = accessTime
    const value: number[] = []
    let rest = key
    while (rest.length > 0 && node.children.has(rest[0]!)) {
      const child = node.children.get(rest[0]!)!
      child.lastAccess = accessTime
      const prefixLen = matchLength(child.key, rest)
      if (prefixLen < child.key.length) {
        const newNode = this.splitNode(child, prefixLen)
        value.push(...newNode.value)
        node = newNode
        break
      }
      value.push(...child.value)
      node = child
      rest = rest.slice(prefixLen)
    }
    return { value, node }
  }

  matchPrefix(tokens: readonly Token[]): RadixMatch {
    if (tokens.length === 0) return { prefixLength: 0, slots: [], lastNode: this.root.id }
    const { value, node } = this.matchHelper(tokens)
    this.events.push({ kind: 'match', tokens: tokens.length, matched: value.length, lastNode: node.id })
    return { prefixLength: value.length, slots: value, lastNode: node.id }
  }

  private insertHelper(key: readonly Token[], value: readonly number[]): { prefixLength: number; node: TreeNode } {
    const accessTime = this.tick()
    let node = this.root
    node.lastAccess = accessTime
    if (key.length === 0) return { prefixLength: 0, node }
    let rest = key
    let restValue = value
    let total = 0
    while (rest.length > 0 && node.children.has(rest[0]!)) {
      node = node.children.get(rest[0]!)!
      node.lastAccess = accessTime
      const prefixLen = matchLength(node.key, rest)
      total += prefixLen
      rest = rest.slice(prefixLen)
      restValue = restValue.slice(prefixLen)
      if (prefixLen < node.key.length) node = this.splitNode(node, prefixLen)
    }
    if (rest.length > 0) {
      const created = this.newNode(node, [...rest], [...restValue])
      node.children.set(rest[0]!, created)
      this.evictableSize += rest.length
      this.updateLeafStatus(node)
      this.updateLeafStatus(created)
      this.events.push({ kind: 'insert', node: created.id, parent: node.id, tokens: rest.length })
      node = created
    }
    return { prefixLength: total, node }
  }

  insert(tokens: readonly Token[], slots: readonly number[]): RadixMatch {
    if (slots.length < tokens.length) throw new Error('insert needs one slot per token')
    const { prefixLength, node } = this.insertHelper(tokens, slots.slice(0, tokens.length))
    return { prefixLength, slots: [], lastNode: node.id }
  }

  private updateLeafStatus(node: TreeNode): void {
    if (node.lockRef > 0) {
      this.evictableLeaves.delete(node)
      return
    }
    if (node.children.size > 0) {
      this.evictableLeaves.delete(node)
      return
    }
    this.evictableLeaves.add(node)
  }

  private pathIds(node: TreeNode): number[] {
    const ids: number[] = []
    for (let n: TreeNode | null = node; n && n !== this.root; n = n.parent) ids.push(n.id)
    return ids
  }

  incLockRef(nodeId: number): void {
    const start = this.findNode(nodeId)
    for (let node = start; node !== this.root; node = node.parent!) {
      if (node.lockRef === 0) {
        this.evictableSize -= node.key.length
        this.protectedSize += node.key.length
      }
      node.lockRef += 1
      this.updateLeafStatus(node)
    }
    this.events.push({ kind: 'lock', node: nodeId, path: this.pathIds(start) })
  }

  decLockRef(nodeId: number): void {
    const start = this.findNode(nodeId)
    for (let node = start; node !== this.root; node = node.parent!) {
      if (node.lockRef === 1) {
        this.evictableSize += node.key.length
        this.protectedSize -= node.key.length
      }
      node.lockRef -= 1
      this.updateLeafStatus(node)
    }
    this.events.push({ kind: 'unlock', node: nodeId, path: this.pathIds(start) })
  }

  evict(numTokens: number): number {
    const heap = [...this.evictableLeaves]
    const popOldest = (): TreeNode => {
      let best = 0
      for (let i = 1; i < heap.length; i++) {
        const a = heap[i]!
        const b = heap[best]!
        if (a.lastAccess < b.lastAccess || (a.lastAccess === b.lastAccess && a.id < b.id)) best = i
      }
      return heap.splice(best, 1)[0]!
    }
    let evicted = 0
    while (evicted < numTokens && heap.length > 0) {
      const x = popOldest()
      this.free(x.value)
      evicted += x.value.length
      const parent = x.parent!
      parent.children.delete(x.key[0]!)
      this.evictableSize -= x.key.length
      this.evictableLeaves.delete(x)
      this.updateLeafStatus(parent)
      this.events.push({ kind: 'evict', node: x.id, slots: x.value })
      if (parent.children.size === 0 && parent.lockRef === 0) heap.push(parent)
    }
    return evicted
  }

  private allocWithEviction(n: number, req: string): number[] {
    const shortfall = n - this.availableSize()
    if (shortfall > 0) this.evict(shortfall)
    const slots = this.alloc(n, req)
    if (slots === null) throw new Error(`out of KV slots: request ${req} needs ${n}, ${this.availableSize()} available`)
    return slots
  }

  admit(reqId: string, prompt: readonly Token[]): RadixMatch {
    if (this.reqs.has(reqId)) throw new Error(`request ${reqId} already admitted`)
    if (prompt.length === 0) throw new Error('empty prompt')
    const match = this.matchPrefix(prompt)
    const lock = this.findNode(match.lastNode)
    this.incLockRef(lock.id)
    const fresh = this.allocWithEviction(prompt.length - match.prefixLength, reqId)
    this.reqs.set(reqId, {
      id: reqId,
      promptLength: prompt.length,
      tokens: [...prompt],
      slots: [...match.slots, ...fresh],
      cacheProtectedLen: match.prefixLength,
      lock,
    })
    return match
  }

  extend(reqId: string, tokens: readonly Token[]): number[] {
    const req = this.request(reqId)
    const slots = this.allocWithEviction(tokens.length, reqId)
    req.tokens.push(...tokens)
    req.slots.push(...slots)
    return slots
  }

  private request(reqId: string): ReqState {
    const req = this.reqs.get(reqId)
    if (!req) throw new Error(`unknown request ${reqId}`)
    return req
  }

  checkpoint(reqId: string, upTo?: number): void {
    const req = this.request(reqId)
    const end = upTo ?? req.tokens.length
    const key = req.tokens.slice(0, end)
    const values = req.slots.slice(0, end)
    const result = this.insertHelper(key, values)
    if (req.promptLength > 0 && req.promptLength < key.length) {
      this.insertHelper(key.slice(0, req.promptLength), values.slice(0, req.promptLength))
    }
    this.free(req.slots.slice(req.cacheProtectedLen, result.prefixLength), reqId)
    const { value, node } = this.matchHelper(key)
    req.slots.splice(req.cacheProtectedLen, value.length - req.cacheProtectedLen, ...value.slice(req.cacheProtectedLen))
    req.cacheProtectedLen = value.length
    this.decLockRef(req.lock.id)
    this.incLockRef(node.id)
    req.lock = node
  }

  release(reqId: string, upTo?: number): void {
    const req = this.request(reqId)
    const owned = upTo ?? req.tokens.length
    this.checkpoint(reqId, owned)
    this.free(req.slots.slice(req.cacheProtectedLen, owned), reqId)
    this.decLockRef(req.lock.id)
    this.free(req.slots.slice(owned), reqId)
    this.reqs.delete(reqId)
  }

  snapshot(): RadixSnapshot {
    const nodes: RadixNodeView[] = []
    const stack = [this.root]
    while (stack.length > 0) {
      const node = stack.pop()!
      nodes.push(
        Object.freeze({
          id: node.id,
          parent: node.parent?.id ?? null,
          key: Object.freeze([...node.key]),
          slots: Object.freeze([...node.value]),
          lockRef: node.lockRef,
          lastAccess: node.lastAccess,
          children: Object.freeze([...node.children.values()].map((c) => c.id)),
          evictable: this.evictableLeaves.has(node),
        }),
      )
      stack.push(...[...node.children.values()].reverse())
    }
    const requests = [...this.reqs.values()].map((r) =>
      Object.freeze({
        id: r.id,
        tokens: Object.freeze([...r.tokens]),
        slots: Object.freeze([...r.slots]),
        cacheProtectedLen: r.cacheProtectedLen,
        lastNode: r.lock.id,
      }),
    )
    return Object.freeze({
      clock: this.clock,
      nodes: Object.freeze(nodes),
      freeSlots: Object.freeze([...this.freePages]),
      releasedSlots: Object.freeze([...this.releasePages]),
      evictableSize: this.evictableSize,
      protectedSize: this.protectedSize,
      requests: Object.freeze(requests),
      events: Object.freeze(this.takeEvents().map((e) => Object.freeze(e))),
    })
  }
}
