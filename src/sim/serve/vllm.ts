import { padTo, vllmCaptureSizes } from './graphs';
import { BLOCK_SIZE, blockKey, sequenceOf, type Sequence } from './tokens';
import type { BatchRow, EngineEvent, EngineRun, EngineStep, RequestStatus, ServingConfig } from './types';

interface Block {
  id: number;
  ref: number;
  key: string | null;
}

interface Req {
  seq: Sequence;
  arrive: number;
  prompt: number;
  out: number;
  outputs: number;
  placeholders: number;
  computed: number;
  hit: number;
  blocks: number[];
  status: RequestStatus;
  firstToken: number | null;
  finished: number | null;
  prefillTokens: number;
  decodeTokens: number;
  evictions: number;
}

interface Batch {
  step: number;
  rows: BatchRow[];
}

export function simulateVllm(config: ServingConfig): EngineRun {
  const maxSeqs = config.maxSeqs ?? 256;
  const blockCount = Math.max(1, Math.floor(config.kvTokens / BLOCK_SIZE));
  const blocks: Block[] = Array.from({ length: blockCount + 1 }, (_, id) => ({ id, ref: 0, key: null }));
  const freeQueue: number[] = Array.from({ length: blockCount }, (_, i) => i + 1);
  const cached = new Map<string, number>();
  const capture = vllmCaptureSizes(maxSeqs, config.budget);
  const reqs: Req[] = config.requests.map((r, i) => ({
    seq: sequenceOf(r, i), arrive: r.arrive, prompt: r.prompt, out: Math.max(1, r.out),
    outputs: 0, placeholders: 0, computed: 0, hit: 0, blocks: [], status: 'pending',
    firstToken: null, finished: null, prefillTokens: 0, decodeTokens: 0, evictions: 0,
  }));
  const byId = new Map(reqs.map((r) => [r.seq.id, r]));
  const waiting: Req[] = [];
  const running: Req[] = [];
  const steps: EngineStep[] = [];
  let inflight: Batch | null = null;

  const total = (r: Req) => r.prompt + r.outputs + r.placeholders;
  const known = (r: Req) => r.prompt + r.outputs;

  function allocate(r: Req, count: number) {
    for (let i = 0; i < count; i++) {
      const block = blocks[freeQueue.shift()!];
      if (block.key !== null && cached.get(block.key) === block.id) cached.delete(block.key);
      block.key = null;
      block.ref = 1;
      r.blocks.push(block.id);
    }
  }

  function release(r: Req) {
    for (const id of [...r.blocks].reverse()) {
      const block = blocks[id];
      block.ref -= 1;
      if (block.ref === 0) freeQueue.push(id);
    }
    r.blocks = [];
  }

  function cacheBlocks(r: Req, upto: number) {
    const full = Math.floor(Math.min(upto, known(r)) / BLOCK_SIZE);
    for (let i = 0; i < full && i < r.blocks.length; i++) {
      const block = blocks[r.blocks[i]];
      if (block.key !== null) continue;
      const key = blockKey(r.seq, i);
      if (cached.has(key)) continue;
      block.key = key;
      cached.set(key, block.id);
    }
  }

  function preempt(victim: Req, events: EngineEvent[]) {
    running.splice(running.indexOf(victim), 1);
    release(victim);
    victim.computed = 0;
    victim.placeholders = 0;
    victim.hit = 0;
    victim.status = 'preempted';
    victim.evictions += 1;
    waiting.unshift(victim);
    events.push({ kind: 'preempt', ids: [victim.seq.id] });
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
      let need = 0;
      for (;;) {
        need = Math.ceil((r.computed + n) / BLOCK_SIZE) - r.blocks.length;
        if (need <= freeQueue.length) break;
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
      allocate(r, need);
      cacheBlocks(r, r.computed + n);
      const row: BatchRow = { id: r.seq.id, from: r.computed, n, duplicate: [], sampled: false };
      rows.push(row);
      rowOf.set(r, row);
      budget -= n;
    }
    while (!preempted && waiting.length > 0 && budget > 0 && running.length < maxSeqs) {
      const r = waiting[0];
      const hits: number[] = [];
      if (r.computed === 0) {
        for (let i = 0; (i + 1) * BLOCK_SIZE <= total(r) - 1; i++) {
          const id = cached.get(blockKey(r.seq, i));
          if (id === undefined) break;
          hits.push(id);
        }
      }
      const hitTokens = hits.length * BLOCK_SIZE;
      const freeAfterTouch = freeQueue.length - hits.filter((id) => blocks[id].ref === 0).length;
      if (Math.ceil(total(r) / BLOCK_SIZE) - hits.length > freeAfterTouch) {
        events.push({ kind: 'blocked', ids: [r.seq.id] });
        break;
      }
      const n = Math.min(total(r) - hitTokens, budget);
      for (const id of hits) {
        if (blocks[id].ref === 0) freeQueue.splice(freeQueue.indexOf(id), 1);
        blocks[id].ref += 1;
      }
      r.blocks = [...hits];
      r.computed = hitTokens;
      r.hit = hitTokens;
      allocate(r, Math.ceil((hitTokens + n) / BLOCK_SIZE) - r.blocks.length);
      cacheBlocks(r, hitTokens + n);
      waiting.shift();
      running.push(r);
      if (r.status === 'preempted') events.push({ kind: 'resume', ids: [r.seq.id], hit: hitTokens });
      r.status = 'running';
      const row: BatchRow = { id: r.seq.id, from: hitTokens, n, duplicate: [], sampled: false };
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
      cacheBlocks(r, r.computed - r.placeholders);
      got.push(r.seq.id);
      if (r.outputs >= r.out) {
        const at = running.indexOf(r);
        if (at >= 0) running.splice(at, 1);
        release(r);
        r.status = 'finished';
        r.finished = step;
        done.push(r.seq.id);
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
      id: r.seq.id, status: r.status, prompt: r.prompt, outputs: r.outputs, inflight: r.placeholders,
      computed: r.computed, hit: r.hit,
    }));
    const kvUsed = (blockCount - freeQueue.length) * BLOCK_SIZE;
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
      id: r.seq.id, firstToken: r.firstToken, finished: r.finished, prefillTokens: r.prefillTokens,
      decodeTokens: r.decodeTokens, duplicateTokens: 0, evictions: r.evictions,
    })),
  };
}
