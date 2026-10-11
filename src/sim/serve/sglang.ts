import { padTo, sglangDecodeSizes, sglangPrefillSizes } from './graphs';
import type { Token } from './hash';
import { RadixCacheSim } from './radix-cache';
import { tokenIds } from './tokens';
import type { BatchRow, EngineEvent, EngineRun, EngineStep, RequestStatus, ServingConfig } from './types';

export const INIT_NEW_TOKEN_RATIO = 0.7;
export const MIN_NEW_TOKEN_RATIO_FACTOR = 0.14;
export const NEW_TOKEN_RATIO_DECAY_STEPS = 600;
export const RETRACT_DECODE_STEPS = 20;
export const CLIP_MAX_NEW_TOKENS = 4096;
export const IN_BATCH_PREFIX_THRESHOLD = 32;
export const LPM_QUEUE_LIMIT = 128;

type Status = RequestStatus | 'prefill';

interface Req {
  id: string;
  tokens: Token[];
  arrive: number;
  prompt: number;
  out: number;
  outputs: number;
  inflight: number;
  computed: number;
  hit: number;
  status: Status;
  firstToken: number | null;
  finished: number | null;
  prefillTokens: number;
  decodeTokens: number;
  duplicateTokens: number;
  evictions: number;
}

interface Batch {
  step: number;
  mode: 'extend' | 'decode';
  rows: BatchRow[];
}

export function simulateSglang(config: ServingConfig): EngineRun {
  const capacity = config.kvTokens;
  const chunk = config.budget;
  const policy = config.policy ?? 'fcfs';
  const decodeSizes = sglangDecodeSizes(config.maxSeqs ?? 256);
  const prefillSizes = sglangPrefillSizes(chunk);
  const minRatio = INIT_NEW_TOKEN_RATIO * MIN_NEW_TOKEN_RATIO_FACTOR;
  const decay = (INIT_NEW_TOKEN_RATIO - minRatio) / NEW_TOKEN_RATIO_DECAY_STEPS;
  let ratio = INIT_NEW_TOKEN_RATIO;
  const reqs: Req[] = config.requests.map((r, i) => ({
    id: r.id, tokens: tokenIds(r, i, r.prompt + Math.max(1, r.out)), arrive: r.arrive, prompt: r.prompt, out: Math.max(1, r.out),
    outputs: 0, inflight: 0, computed: 0, hit: 0, status: 'pending',
    firstToken: null, finished: null, prefillTokens: 0, decodeTokens: 0, duplicateTokens: 0, evictions: 0,
  }));
  const byId = new Map(reqs.map((r) => [r.id, r]));
  const cache = new RadixCacheSim(capacity);
  const waiting: Req[] = [];
  const running: Req[] = [];
  let chunked: Req | null = null;
  let lastBatch: Batch | null = null;
  let batchFull = false;
  let shrunk = false;
  let inflight: Batch | null = null;
  const steps: EngineStep[] = [];

  const active = (r: Req) => r.status === 'running' || r.status === 'prefill';

  function matchPrefix(r: Req) {
    return cache.matchPrefix(r.tokens.slice(0, r.prompt + r.outputs - 1));
  }

  function decodeHeadroom(): number {
    return running.reduce((s, r) => s + ratio * Math.min(r.out - r.outputs - r.inflight, CLIP_MAX_NEW_TOKENS), 0);
  }

  function processOutputs(batch: Batch, step: number, events: EngineEvent[]) {
    const got: string[] = [];
    const done: string[] = [];
    for (const row of batch.rows) {
      if (!row.sampled) continue;
      const r = byId.get(row.id)!;
      r.inflight -= 1;
      r.outputs += 1;
      r.firstToken ??= batch.step;
      got.push(r.id);
      if (batch.mode === 'extend' && active(r)) cache.checkpoint(r.id, Math.min(r.computed, r.prompt));
      if (r.outputs >= r.out && r.status !== 'finished') {
        if (active(r)) cache.release(r.id, r.computed, true);
        if (running.includes(r)) {
          running.splice(running.indexOf(r), 1);
          shrunk = true;
        }
        if (waiting.includes(r)) waiting.splice(waiting.indexOf(r), 1);
        r.status = 'finished';
        r.finished = step;
        done.push(r.id);
      }
    }
    events.push({ kind: 'process', ids: got, batch: batch.step });
    if (done.length > 0) events.push({ kind: 'finish', ids: done });
  }

  function orderWaiting(): Req[] {
    const candidates = [...waiting];
    if (policy !== 'lpm' || candidates.length > LPM_QUEUE_LIMIT) return candidates;
    const matched = new Map(candidates.map((r) => [r, matchPrefix(r).prefixLength]));
    const seen = new Set<Token>();
    const deprioritized = new Set<Req>();
    for (const r of candidates) {
      if (matched.get(r)! > IN_BATCH_PREFIX_THRESHOLD) continue;
      let p = 0;
      while (p < r.prompt && seen.has(r.tokens[p])) p++;
      if (p >= IN_BATCH_PREFIX_THRESHOLD) deprioritized.add(r);
      else for (let q = 0; q < r.prompt; q++) seen.add(r.tokens[q]);
    }
    return candidates.sort((a, b) => Number(deprioritized.has(a)) - Number(deprioritized.has(b)) || (deprioritized.has(a) ? 0 : matched.get(b)! - matched.get(a)!));
  }

  function buildPrefill(events: EngineEvent[]): BatchRow[] {
    const rows: BatchRow[] = [];
    let remaining = chunk;
    const computedKeys = new Set<Token>();
    for (const r of reqs) if (active(r)) for (let p = 0; p < r.computed; p++) computedKeys.add(r.tokens[p]);

    function addRow(r: Req, from: number, n: number) {
      const duplicate: number[] = [];
      const cached = cache.cachedPrefixLength(r.tokens.slice(0, from + n));
      for (let p = from; p < from + n; p++) {
        if (p < cached || computedKeys.has(r.tokens[p])) duplicate.push(p);
        computedKeys.add(r.tokens[p]);
      }
      cache.extend(r.id, n);
      r.computed = from + n;
      r.prefillTokens += n;
      r.duplicateTokens += duplicate.length;
      const sampled = r.computed === r.prompt + r.outputs;
      if (sampled) r.inflight += 1;
      rows.push({ id: r.id, from, n, duplicate, sampled });
      remaining -= n;
    }

    if (chunked !== null) {
      const r = chunked;
      const rest = r.prompt + r.outputs - r.computed;
      const n = Math.min(rest, remaining);
      if (n === rest) chunked = null;
      addRow(r, r.computed, n);
    }
    if (remaining <= 0 || chunked !== null || waiting.length === 0 || (batchFull && rows.length === 0)) return rows;
    let kvBudget = cache.availableSize() + cache.evictableSize() - decodeHeadroom();
    for (const r of orderWaiting()) {
      if (remaining <= 0) break;
      const match = matchPrefix(r);
      const matched = match.prefixLength;
      const extend = r.prompt + r.outputs - matched;
      const truncated = extend > remaining;
      const cost = truncated ? remaining : extend + Math.min(r.out - r.outputs, CLIP_MAX_NEW_TOKENS);
      if (cost >= kvBudget) {
        batchFull = true;
        events.push({ kind: 'blocked', ids: [r.id] });
        break;
      }
      kvBudget -= cost;
      waiting.splice(waiting.indexOf(r), 1);
      r.status = 'prefill';
      cache.admit(r.id, r.tokens, r.prompt, match);
      r.hit = matched;
      r.computed = matched;
      if (truncated) {
        chunked = r;
        addRow(r, matched, remaining);
        break;
      }
      addRow(r, matched, extend);
    }
    return rows;
  }

  function buildDecode(events: EngineEvent[]): BatchRow[] {
    const needsToken = (r: Req) => r.outputs + r.inflight < r.out;
    if (shrunk || running.some((r) => !needsToken(r))) batchFull = false;
    shrunk = false;
    if (!running.some(needsToken)) return [];
    if (cache.availableSize() + cache.evictableSize() < running.filter(needsToken).length) {
      const order = [...running].sort((a, b) => b.outputs - a.outputs || a.prompt - b.prompt);
      while (order.length > 1 && cache.availableSize() + cache.evictableSize() < running.filter(needsToken).length) {
        const victim = order.pop()!;
        running.splice(running.indexOf(victim), 1);
        cache.release(victim.id, 0, false);
        victim.computed = 0;
        victim.hit = 0;
        victim.status = 'waiting';
        victim.evictions += 1;
        batchFull = false;
        waiting.push(victim);
        events.push({ kind: 'retract', ids: [victim.id] });
      }
      const decoded = running.reduce((s, r) => s + r.outputs, 0);
      const maxNew = running.reduce((s, r) => s + r.out, 0);
      ratio = Math.min(1, (decoded + RETRACT_DECODE_STEPS * running.length) / (maxNew + 1));
    } else ratio = Math.max(ratio - decay, minRatio);
    return running.filter(needsToken).map((r) => {
      cache.extend(r.id, 1);
      const row: BatchRow = { id: r.id, from: r.computed, n: 1, duplicate: [], sampled: true };
      r.computed += 1;
      r.decodeTokens += 1;
      r.inflight += 1;
      return row;
    });
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
    const pendingChunk = chunked as Req | null;
    if (pendingChunk !== null) cache.checkpoint(pendingChunk.id, pendingChunk.computed);
    if (lastBatch?.mode === 'extend') {
      if (lastBatch.rows.some((row) => byId.get(row.id)!.status !== 'prefill' || byId.get(row.id) === chunked)) batchFull = false;
      for (const row of lastBatch.rows) {
        const r = byId.get(row.id)!;
        if (r !== chunked && r.status === 'prefill') {
          r.status = 'running';
          running.push(r);
        }
      }
    }
    let rows = buildPrefill(events);
    let mode: Batch['mode'] | null = rows.length > 0 ? 'extend' : null;
    if (mode === null) {
      rows = buildDecode(events);
      if (rows.length > 0) mode = 'decode';
    }
    const tokens = rows.reduce((s, r) => s + r.n, 0);
    let size: number | null = null;
    if (mode === 'decode') size = padTo(decodeSizes, rows.length);
    if (mode === 'extend') {
      const bucket = padTo(prefillSizes, tokens);
      size = bucket !== null && bucket <= 2 * tokens ? bucket : null;
    }
    const views = reqs.map((r) => ({
      id: r.id, status: (r.status === 'prefill' ? 'running' : r.status) as RequestStatus, prompt: r.prompt,
      outputs: r.outputs, inflight: r.inflight, computed: r.computed, hit: r.hit,
    }));
    const batch: Batch | null = mode === null ? null : { step, mode, rows };
    if (batch !== null) events.unshift({ kind: 'launch', ids: rows.map((r) => r.id), batch: step });
    if (config.overlap) {
      if (inflight !== null) processOutputs(inflight, step, events);
      inflight = batch;
    } else if (batch !== null) processOutputs(batch, step, events);
    steps.push({
      step, rows, tokens,
      mode: mode === null ? 'idle' : mode === 'decode' ? 'decode' : 'prefill',
      graph: mode === null ? 'none' : size === null ? 'eager' : mode === 'decode' ? 'decode' : 'prefill',
      padded: size === null ? 0 : mode === 'decode' ? size - rows.length : size - tokens,
      requests: views, events, kvUsed: capacity - cache.availableSize(), newTokenRatio: ratio,
    });
    lastBatch = batch;
    if (inflight === null && reqs.every((r) => r.status === 'finished')) break;
  }
  return {
    engine: 'sglang',
    kvCapacity: capacity,
    steps,
    requests: reqs.map((r) => ({
      id: r.id, firstToken: r.firstToken, finished: r.finished, prefillTokens: r.prefillTokens,
      decodeTokens: r.decodeTokens, duplicateTokens: r.duplicateTokens, evictions: r.evictions,
    })),
  };
}
