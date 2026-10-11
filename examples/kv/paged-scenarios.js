import { RadixCacheSim, simulateVllm } from '@yhnocoder/vizflow/serve';

const chars = (text) => [...text];

export const SYSTEM_PROMPT = chars('你是乐于助人的助手');

export const SHARED_REQUESTS = [
  { id: 'A', prompt: [...SYSTEM_PROMPT, ...chars('请写诗')], outputs: chars('春眠不觉'), maxTokens: 4, arrival: 0 },
  { id: 'B', prompt: [...SYSTEM_PROMPT, ...chars('请算数')], outputs: chars('一加一等于二三四'), maxTokens: 8, arrival: 1 },
  { id: 'C', prompt: chars('把下面这段英文翻译成中文并保持原意不变谢谢'), outputs: chars('好的以下是译文内容如'), maxTokens: 10, arrival: 6 },
  { id: 'D', prompt: [...SYSTEM_PROMPT, ...chars('请讲故事')], outputs: chars('从前有座山山里有'), maxTokens: 8, arrival: 10 },
];

export const VLLM_CONFIG = {
  numGpuBlocks: 12,
  groups: [{ kind: 'full', blockSize: 4 }],
  maxNumBatchedTokens: 32,
  maxNumSeqs: 4,
  maxModelLen: 64,
};

export function vllmScenario() {
  return { config: VLLM_CONFIG, requests: SHARED_REQUESTS, snaps: simulateVllm(VLLM_CONFIG, SHARED_REQUESTS) };
}

export const RADIX_SLOTS = 48;

export function radixScenario() {
  const cache = new RadixCacheSim(RADIX_SLOTS);
  const live = new Map();
  const finished = new Set();
  const steps = [];
  const lastArrival = Math.max(...SHARED_REQUESTS.map((r) => r.arrival));
  for (let step = 0; step <= lastArrival || live.size > 0; step++) {
    const actions = [];
    const record = (action) => actions.push({ ...action, events: cache.takeEvents() });
    for (const [id, state] of live) {
      const token = state.spec.outputs[state.sampled - 1];
      const slots = cache.extend(id, [token]);
      state.sampled += 1;
      record({ kind: 'decode', req: id, token, slots });
    }
    for (const spec of SHARED_REQUESTS) {
      if (spec.arrival !== step) continue;
      const match = cache.admit(spec.id, spec.prompt);
      live.set(spec.id, { spec, sampled: 1 });
      record({ kind: 'admit', req: spec.id, prefixLength: match.prefixLength, promptLength: spec.prompt.length });
      if (spec.maxTokens > 1) {
        cache.checkpoint(spec.id);
        record({ kind: 'checkpoint', req: spec.id });
      }
    }
    for (const [id, state] of [...live]) {
      if (state.sampled < state.spec.maxTokens) continue;
      cache.release(id);
      live.delete(id);
      finished.add(id);
      record({ kind: 'release', req: id });
    }
    const snapshot = cache.snapshot();
    steps.push({
      step,
      snapshot,
      actions,
      running: [...live.keys()],
      finished: [...finished],
      sampled: Object.fromEntries([...live].map(([id, s]) => [id, s.sampled])),
    });
  }
  return { numSlots: RADIX_SLOTS, requests: SHARED_REQUESTS, steps };
}

export const V4_CONFIG = {
  numGpuBlocks: 26,
  groups: [
    { kind: 'compressed', tokensPerBlock: 16, ratio: 4 },
    { kind: 'compressed', tokensPerBlock: 16, ratio: 8 },
    { kind: 'sliding_window', blockSize: 4, window: 8 },
  ],
  maxNumBatchedTokens: 64,
  maxNumSeqs: 4,
  maxModelLen: 64,
};

const V4_PREFIX = chars('文档开头的二十二个字作为共享前缀用于测试内容');

export const V4_REQUESTS = [
  { id: 'A', prompt: V4_PREFIX, outputs: chars('甲乙丙丁戊己庚辛壬癸子丑'), maxTokens: 12, arrival: 0 },
  { id: 'B', prompt: [...V4_PREFIX, ...chars('后续追问再加八字')], outputs: chars('好的呀'), maxTokens: 3, arrival: 13 },
];

export function v4Scenario() {
  return { config: V4_CONFIG, requests: V4_REQUESTS, snaps: simulateVllm(V4_CONFIG, V4_REQUESTS) };
}
