import { describe, expect, test } from 'vitest'
import type { RadixEvent, VllmEvent, VllmSnapshot } from '../../src/serve'
// @ts-expect-error untyped example module
import { radixScenario, v4Scenario, vllmScenario } from '../../examples/kv/paged-scenarios.js'

type RadixAction = { kind: string; req: string; prefixLength?: number; events: RadixEvent[] }
type RadixStep = { step: number; actions: RadixAction[] }

function eventsOf<K extends VllmEvent['kind']>(snap: VllmSnapshot, kind: K): Extract<VllmEvent, { kind: K }>[] {
  return snap.events.filter((e): e is Extract<VllmEvent, { kind: K }> => e.kind === kind)
}

function firstAdmit(snaps: VllmSnapshot[], req: string) {
  for (const snap of snaps) for (const e of eventsOf(snap, 'admit')) if (e.req === req) return { step: snap.step, event: e }
  throw new Error(`request ${req} never admitted`)
}

describe('vllm mode scenario', () => {
  const { snaps } = vllmScenario() as { snaps: VllmSnapshot[] }

  test('B diverges from A inside block 2, so only the two full system-prompt blocks are hit', () => {
    const { step, event } = firstAdmit(snaps, 'B')
    expect(step).toBe(1)
    expect(event.hitTokens).toBe(8)
    expect(eventsOf(snaps[1]!, 'hit')).toEqual([{ kind: 'hit', req: 'B', group: 0, blockIds: [1, 2] }])
  })

  test('A finishing sends its unhashed tail block to the head and its hashed block to the tail', () => {
    expect(eventsOf(snaps[3]!, 'free')).toEqual([
      { kind: 'free', req: 'A', group: 0, reason: 'finish', blockIds: [4, 3, 2, 1], toHead: [4], toTail: [3] },
    ])
  })

  test('C allocating many blocks evicts a hashed cached block', () => {
    expect(eventsOf(snaps[6]!, 'evict')).toEqual([{ kind: 'evict', req: 'C', group: 0, blockId: 3, hash: expect.any(String), hashGroup: 0, reason: 'reuse' }])
  })

  test('D revives the system-prompt blocks from the free queue', () => {
    expect(eventsOf(snaps[10]!, 'touch')).toEqual([{ kind: 'touch', req: 'D', group: 0, blockIds: [1, 2] }])
  })

  test('C runs out of blocks, running[-1] D is preempted and later resumes with a prefix hit', () => {
    expect(eventsOf(snaps[14]!, 'preempt')).toEqual([{ kind: 'preempt', req: 'D', for: 'C' }])
    expect(snaps[14]!.waiting).toEqual(['D'])
    const resumed = eventsOf(snaps[16]!, 'admit')
    expect(resumed).toEqual([{ kind: 'admit', req: 'D', resumed: true, hitTokens: 12, numNewTokens: 5 }])
  })

  test('all four requests finish within 20 steps', () => {
    expect(snaps.length).toBe(20)
  })
})

describe('radix mode scenario', () => {
  const { steps } = radixScenario() as { steps: RadixStep[] }
  const admitted = (req: string) => steps.flatMap((s) => s.actions).find((a) => a.kind === 'admit' && a.req === req)!

  test('RadixCache reuses B and D up to the divergence token, more than vLLM block hashing', () => {
    const { snaps } = vllmScenario() as { snaps: VllmSnapshot[] }
    for (const req of ['B', 'D']) {
      expect(admitted(req).prefixLength).toBe(10)
      expect(admitted(req).prefixLength).toBeGreaterThan(firstAdmit(snaps, req).event.hitTokens)
    }
  })

  test('B admission splits A prompt node at the divergence token', () => {
    const splits = admitted('B').events.filter((e) => e.kind === 'split')
    expect(splits).toEqual([expect.objectContaining({ kind: 'split', at: 10 })])
  })

  test('memory pressure evicts unlocked leaves and every request finishes', () => {
    expect(steps.flatMap((s) => s.actions.flatMap((a) => a.events)).some((e) => e.kind === 'evict')).toBe(true)
    expect(steps.flatMap((s) => s.actions).filter((a) => a.kind === 'release').map((a) => a.req).sort()).toEqual(['A', 'B', 'C', 'D'])
  })
})

describe('v4 mode scenario', () => {
  const { snaps } = v4Scenario() as { snaps: VllmSnapshot[] }

  test('scheduler block size is the LCM and hash block size the GCD of group block sizes', () => {
    expect(snaps[0]!.schedulerBlockSize).toBe(16)
    expect(snaps[0]!.hashBlockSize).toBe(4)
  })

  test('sliding-window blocks are freed out_of_window during decode', () => {
    const outOfWindow = snaps.flatMap((s) => eventsOf(s, 'free').filter((e) => e.reason === 'out_of_window').map(() => s.step))
    expect(outOfWindow).toEqual([1, 2, 6, 10, 14, 15])
  })

  test('B shares 22 tokens with A but the hit is aligned down to 16', () => {
    const { step, event } = firstAdmit(snaps, 'B')
    expect(step).toBe(13)
    expect(event.hitTokens).toBe(16)
    expect(eventsOf(snaps[13]!, 'hit').map((e) => [e.group, e.blockIds])).toEqual([
      [0, [1]],
      [1, [3]],
      [2, [7, 8]],
    ])
  })

  test('compressed blocks hold tokensPerBlock / ratio entries', () => {
    const block = (id: number) => snaps[0]!.blocks[id]!
    expect([block(1).filled, block(1).capacity]).toEqual([4, 4])
    expect([block(3).filled, block(3).capacity]).toEqual([2, 2])
  })
})
