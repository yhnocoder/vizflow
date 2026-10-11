import { padTo, vllmCaptureSizes } from './graphs';
import { chainBlockHashes, type Token } from './hash';
import { KvCacheManager, type KvRequest } from './kv-cache-manager';
import { BLOCK_SIZE, tokenIds } from './tokens';
import type { BatchRow, EngineEvent, EngineRun, EngineStep, RequestStatus, ServingConfig, ServingRequest } from './types';

class Req implements KvRequest {
  readonly id: string;
  readonly arrive: number;
  readonly prompt: number;
  readonly out: number;
  readonly tokens: Token[];
  outputs = 0;
  placeholders = 0;
  computed = 0;
  hit = 0;
  status: RequestStatus = 'pending';
  firstToken: number | null = null;
  finished: number | null = null;
  prefillTokens = 0;
  decodeTokens = 0;
  evictions = 0;
  readonly sharedPrefixBoundary = 0;
  private hashes: string[] = [];

  constructor(r: ServingRequest, index: number) {
    this.id = r.id;
    this.arrive = r.arrive;
    this.prompt = r.prompt;
    this.out = Math.max(1, r.out);
    this.tokens = tokenIds(r, index, this.prompt + this.out);
  }

  get numTokens(): number {
    return this.prompt + this.outputs;
  }

  get numPromptTokens(): number {
    return this.prompt;
  }

  get numComputedTokens(): number {
    return this.computed;
  }

  blockHashes(): readonly string[] {
    this.hashes = chainBlockHashes(this.tokens.slice(0, this.numTokens), BLOCK_SIZE, undefined, this.hashes);
    return this.hashes;
  }
}

interface Batch {
  step: number;
  rows: BatchRow[];
}

export function simulateVllm(config: ServingConfig): EngineRun {
  const maxSeqs = config.maxSeqs ?? 256;
  const blockCount = Math.max(1, Math.floor(config.kvTokens / BLOCK_SIZE));
  const kv = new KvCacheManager({
    numGpuBlocks: blockCount + 1, groups: [{ kind: 'full', blockSize: BLOCK_SIZE }],
    maxModelLen: Infinity, maxInFlightTokens: config.budget, enablePrefixCaching: true,
  }, () => {});
  const capture = vllmCaptureSizes(maxSeqs, config.budget);
  const reqs = config.requests.map((r, i) => new Req(r, i));
  const byId = new Map(reqs.map((r) => [r.id, r]));
  const waiting: Req[] = [];
  const running: Req[] = [];
  const steps: EngineStep[] = [];
  let inflight: Batch | null = null;

  const total = (r: Req) => r.prompt + r.outputs + r.placeholders;

  function preempt(victim: Req, events: EngineEvent[]) {
    running.splice(running.indexOf(victim), 1);
    kv.free(victim.id, 'preempt');
    victim.computed = 0;
    victim.placeholders = 0;
    victim.hit = 0;
    victim.status = 'preempted';
    victim.evictions += 1;
    waiting.unshift(victim);
    events.push({ kind: 'preempt', ids: [victim.id] });
  }

  function schedule(events: EngineEvent[]): BatchRow[] {
    let budget = config.budget;
    let preempted = false;
    const rows: BatchRow[] = [];
    const rowOf = new Map<Req, BatchRow>();
    for (let i = 0; i < running.length && budget > 0; i++) {
      const r = running[i];
      if (r.outputs + r.placeholders >= r.out) continue;
      let n = Math.min(total(r) - r.computed, budget);
      if (n <= 0) continue;
      while (kv.allocateSlots(r, n, null, false) === null) {
        const victim = running[running.length - 1];
        preempt(victim, events);
        preempted = true;
        if (victim === r) {
          n = 0;
          break;
        }
        const scheduled = rowOf.get(victim);
        if (scheduled !== undefined) {
          budget += scheduled.n;
          rows.splice(rows.indexOf(scheduled), 1);
          rowOf.delete(victim);
        }
      }
      if (n === 0) break;
      const row: BatchRow = { id: r.id, from: r.computed, n, duplicate: [], sampled: false };
      rows.push(row);
      rowOf.set(r, row);
      budget -= n;
    }
    while (!preempted && waiting.length > 0 && budget > 0 && running.length < maxSeqs) {
      const r = waiting[0];
      const hit = kv.getComputedBlocks(r);
      const n = Math.min(total(r) - hit.numTokens, budget);
      if (kv.allocateSlots(r, n, hit, true) === null) {
        events.push({ kind: 'blocked', ids: [r.id] });
        break;
      }
      const hitTokens = hit.numTokens;
      r.computed = hitTokens;
      r.hit = hitTokens;
      waiting.shift();
      running.push(r);
      if (r.status === 'preempted') events.push({ kind: 'resume', ids: [r.id], hit: hitTokens });
      r.status = 'running';
      const row: BatchRow = { id: r.id, from: hitTokens, n, duplicate: [], sampled: false };
      rows.push(row);
      rowOf.set(r, row);
      budget -= n;
    }
    for (const row of rows) {
      const r = byId.get(row.id)!;
      const end = total(r);
      if (r.computed >= r.prompt) r.decodeTokens += row.n;
      else r.prefillTokens += row.n;
      r.computed += row.n;
      row.sampled = r.computed === end;
      if (row.sampled && config.overlap) r.placeholders += 1;
    }
    return rows;
  }

  function processOutputs(batch: Batch, step: number, events: EngineEvent[]) {
    const got: string[] = [];
    const done: string[] = [];
    for (const row of batch.rows) {
      const r = byId.get(row.id)!;
      if (!row.sampled || r.status === 'finished') continue;
      r.outputs += 1;
      if (r.placeholders > 0) r.placeholders -= 1;
      r.firstToken ??= batch.step;
      kv.cacheBlocks(r, r.computed - r.placeholders);
      got.push(r.id);
      if (r.outputs >= r.out) {
        const at = running.indexOf(r);
        if (at >= 0) running.splice(at, 1);
        kv.free(r.id, 'finish');
        r.status = 'finished';
        r.finished = step;
        done.push(r.id);
      }
    }
    events.push({ kind: 'process', ids: got, batch: batch.step });
    if (done.length > 0) events.push({ kind: 'finish', ids: done });
  }

  const maxSteps = config.maxSteps ?? 4000;
  for (let step = 1; step <= maxSteps; step++) {
    const events: EngineEvent[] = [];
    for (const r of reqs) {
      if (r.arrive === step) {
        r.status = 'waiting';
        waiting.push(r);
      }
    }
    const rows = schedule(events);
    const batch: Batch = { step, rows };
    const ordered = [...rows].sort((a, b) => Number(a.n !== 1) - Number(b.n !== 1) || a.n - b.n);
    const tokens = rows.reduce((s, r) => s + r.n, 0);
    const uniform = rows.length > 0 && rows.every((r) => r.n === 1);
    const size = tokens > 0 ? padTo(capture, tokens) : null;
    const views = reqs.map((r) => ({
      id: r.id, status: r.status, prompt: r.prompt, outputs: r.outputs, inflight: r.placeholders,
      computed: r.computed, hit: r.hit,
    }));
    const kvUsed = (blockCount - kv.pool.numFreeBlocks) * BLOCK_SIZE;
    if (rows.length > 0) events.unshift({ kind: 'launch', ids: rows.map((r) => r.id), batch: step });
    if (config.overlap) {
      if (inflight !== null) processOutputs(inflight, step, events);
      inflight = rows.length > 0 ? batch : null;
    } else if (rows.length > 0) processOutputs(batch, step, events);
    steps.push({
      step, rows: ordered, tokens,
      mode: tokens === 0 ? 'idle' : uniform ? 'decode' : rows.every((r) => byId.get(r.id)!.computed - r.n < byId.get(r.id)!.prompt) ? 'prefill' : 'mixed',
      graph: tokens === 0 ? 'none' : size === null ? 'eager' : uniform ? 'full' : 'piecewise',
      padded: size === null ? 0 : size - tokens,
      requests: views, events, kvUsed,
    });
    if (inflight === null && reqs.every((r) => r.status === 'finished')) break;
  }
  return {
    engine: 'vllm',
    kvCapacity: blockCount * BLOCK_SIZE,
    steps,
    requests: reqs.map((r) => ({
      id: r.id, firstToken: r.firstToken, finished: r.finished, prefillTokens: r.prefillTokens,
      decodeTokens: r.decodeTokens, duplicateTokens: 0, evictions: r.evictions,
    })),
  };
}
