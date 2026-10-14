export interface SamplerSpec {
  temperature: number;
  topK?: number;
  topP?: number;
  seed: number;
  note?: string;
}

export interface SampleDraw {
  logp: number[];
  scaled: number[];
  order: number[];
  kept: boolean[];
  final: number[];
  u?: number;
  token: number;
}

export function isGreedy(spec: SamplerSpec | undefined): boolean {
  return spec === undefined || spec.temperature === 0;
}

function softmax(values: readonly number[]): number[] {
  const top = Math.max(...values.filter((v) => v !== -Infinity));
  const e = values.map((v) => (v === -Infinity ? 0 : Math.exp(v - top)));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / z);
}

export function drawToken(row: ArrayLike<number>, spec: SamplerSpec | undefined, u: number): SampleDraw {
  const logp = Array.from(row);
  const order = logp.map((_, v) => v).sort((a, b) => logp[b] - logp[a] || a - b);
  if (spec === undefined || isGreedy(spec)) {
    const scaled = softmax(logp);
    const final = logp.map((_, v) => (v === order[0] ? 1 : 0));
    return { logp, scaled, order, kept: final.map((p) => p > 0), final, token: order[0] };
  }
  const scaled = softmax(logp.map((x) => x / spec.temperature));
  const kept = logp.map(() => true);
  if (spec.topK !== undefined) order.forEach((v, rank) => (kept[v] = rank < (spec.topK as number)));
  if (spec.topP !== undefined) {
    const mass = order.filter((v) => kept[v]).reduce((a, v) => a + scaled[v], 0);
    let before = 0;
    for (const v of order) {
      if (!kept[v]) continue;
      kept[v] = before < (spec.topP as number);
      before += scaled[v] / mass;
    }
  }
  const total = order.filter((v) => kept[v]).reduce((a, v) => a + scaled[v], 0);
  const final = scaled.map((p, v) => (kept[v] ? p / total : 0));
  let cumulative = 0;
  let token = order.find((v) => kept[v]) as number;
  for (const v of order) {
    if (!kept[v]) continue;
    cumulative += final[v];
    token = v;
    if (u < cumulative) break;
  }
  return { logp, scaled, order, kept, final, u, token };
}

export function uniforms(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let x = state;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}
