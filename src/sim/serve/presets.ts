import type { ServingConfig } from './types';

export interface ServingPreset {
  name: string;
  title: string;
  text: string;
  config: ServingConfig;
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

export const SERVING_PRESETS: ServingPreset[] = [
  {
    name: 'three',
    title: '共享前缀的三个请求',
    text: 'A 的提示词 1000 个 token，B 和 C 各 20 个，前 16 个相同。vLLM 在同一步里复用 B 刚登记的前缀块；SGLang 把 C 拆成两块并重算共享的 16 个 token，C 的首 token 晚一批。',
    config: {
      budget: 512, kvTokens: 65536, overlap: true, policy: 'lpm',
      requests: [
        { id: 'A', arrive: 1, prompt: 1000, out: 8 },
        { id: 'B', arrive: 1, prompt: 20, shared: 16, group: 'p', out: 8 },
        { id: 'C', arrive: 1, prompt: 20, shared: 16, group: 'p', out: 8 },
      ],
    },
  },
  {
    name: 'sys',
    title: '共享系统提示词',
    text: '六个请求的前 256 个 token 相同，后三个晚两步到达。看前缀命中在两个引擎里分别从哪一步开始。',
    config: {
      budget: 512, kvTokens: 65536, overlap: true, policy: 'lpm',
      requests: range(6).map((i) => ({ id: `S${i}`, arrive: i > 2 ? 3 : 1, prompt: 300, shared: 256, group: 'sys', out: 12 })),
    },
  },
  {
    name: 'long',
    title: '长提示词插进 decode',
    text: '四个短请求正在 decode，第 6 步来了一个 2000 个 token 的提示词。vLLM 把它的 chunk 和 decode 放在同一批；SGLang 先跑完它的 prefill，decode 中的请求停下等待。',
    config: {
      budget: 512, kvTokens: 65536, overlap: true, policy: 'fcfs',
      requests: [...range(4).map((i) => ({ id: `R${i}`, arrive: 1, prompt: 40, out: 30 })), { id: 'L', arrive: 6, prompt: 2000, out: 6 }],
    },
  },
  {
    name: 'kv',
    title: 'KV 不足',
    text: 'KV 只有 1024 个 token，六个请求陆续到达。vLLM 先接纳再在不够时抢占；SGLang 接纳时就预留未来 decode 需要的 KV。',
    config: {
      budget: 512, kvTokens: 1024, overlap: true, policy: 'fcfs',
      requests: range(6).map((i) => ({ id: `K${i}`, arrive: 1 + i, prompt: 200, out: 120 })),
    },
  },
  {
    name: 'kv-prefix',
    title: 'KV 不足且共享前缀',
    text: '六个请求共享 128 个 token 的文档前缀，KV 只有 1536 个 token。被抢占或撤回的请求重新开始时能命中多少缓存。',
    config: {
      budget: 256, kvTokens: 1536, overlap: true, policy: 'lpm',
      requests: range(6).map((i) => ({ id: `P${i}`, arrive: 1 + 2 * i, prompt: 180, shared: 128, group: 'doc', out: 90 })),
    },
  },
  {
    name: 'stream',
    title: '持续到达的请求',
    text: '十六个长度不同的请求每步到达一个，每步预算 256 个 token。',
    config: {
      budget: 256, kvTokens: 8192, overlap: true, policy: 'fcfs',
      requests: range(16).map((i) => ({ id: `T${String(i).padStart(2, '0')}`, arrive: 1 + i, prompt: 30 + ((i * 37) % 150), out: 5 + ((i * 13) % 25) })),
    },
  },
  {
    name: 'chunks',
    title: '小预算下的分块',
    text: '每步只有 128 个 token。两个 300 个 token 的提示词和一个晚到的短请求怎么分块。',
    config: {
      budget: 128, kvTokens: 65536, overlap: true, policy: 'fcfs',
      requests: [{ id: 'X', arrive: 1, prompt: 300, out: 4 }, { id: 'Y', arrive: 1, prompt: 300, out: 4 }, { id: 'Z', arrive: 2, prompt: 50, out: 10 }],
    },
  },
];
