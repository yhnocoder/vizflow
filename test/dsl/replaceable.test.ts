// @vitest-environment jsdom
import katex from 'katex';
import { expect, it } from 'vitest';
import { type CompiledPage, compilePage, loadPage, parseYaml } from '../../src/dsl';
import { elemTexOf } from '../../src/explain/elem';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { examplesUrl, readFileUrl } from './load';

const fixture = loadFixture<never>('qwen3.json') as unknown as { sequence: number[]; weights: Record<string, EncodedTensor>; loss: number; grads: Record<string, EncodedTensor> };
const PAGE = new URL('qwen3/qwen3-dsl.yaml', examplesUrl).href;

async function qwen3(config: Record<string, unknown>): Promise<CompiledPage> {
  const spec = parseYaml(await readFileUrl(PAGE), PAGE) as { config: Record<string, unknown> };
  Object.assign(spec.config, config);
  return compilePage(spec as never, { base: PAGE, read: readFileUrl });
}

it.each([
  ['flash_attention_2', false],
  ['eager', true],
  ['flash_attention_2', true],
])('qwen3-dsl.yaml with %s and fused_linear_cross_entropy %s matches transformers in float64', async (attn, fused) => {
  const page = await qwen3({ dtype: 'fp64', attn_implementation: attn, fused_linear_cross_entropy: fused });
  for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
  page.sequence!.set(fixture.sequence);
  expectTensor(page.nodes.get('loss')!.value, { shape: [], values: [fixture.loss] }, 'loss');
  for (const [path, grad] of Object.entries(fixture.grads)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
});

it('renders every step text, label and node TeX of the FlashAttention-2 and Liger steps', async () => {
  const page = await qwen3({ attn_implementation: 'flash_attention_2', fused_linear_cross_entropy: true });
  const failures = new Set<string>();
  const renders = (tex: string, what: string) => {
    try {
      katex.renderToString(tex, { throwOnError: true, trust: true });
    } catch (error) {
      failures.add(`${what}: ${(error as Error).message}`);
    }
  };
  const math = (text: string) => [...text.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)].map((m) => m[1] ?? m[2]);
  const steps = createSteps(page.graph, page.steps);
  steps.list.forEach((info, i) => {
    const { title, body } = steps.text(i);
    for (const tex of [...math(title), ...math(body), ...math(info.label)]) renders(tex, `step ${i} ${title}`);
  });
  expect(steps.list.some((s) => s.dir === 'backward' && s.title.startsWith('head 0，kv 块 0，query 块 1'))).toBe(true);
  expect(steps.list.some((s) => s.title === '交出前向算好的梯度')).toBe(true);
  steps.dispose();
  for (const n of page.graph.nodes) {
    renders(n.tex, n.name);
    renders(elemTexOf(n, n.shape.map(() => 0)), `${n.name} elem`);
  }
  expect([...failures]).toEqual([]);
});

const ATTENTION = `title: t
model: M
import: { flash_attn_func: vizflow/nn/flash-attn-func }
config: { bm: BM, bn: BN, bbm: BBM, bbn: BBN }
data:
  q: [[[0.3, -0.8, 0.1, 0.5], [1.1, 0.2, -0.4, 0.3]], [[-0.5, 0.9, 0.2, -0.1], [0.4, -1.2, 0.6, 0.2]], [[0.7, 0.1, -0.3, 0.8], [-0.3, 0.6, 0.9, -0.7]], [[0.2, 0.4, -0.6, 0.1], [0.5, -0.2, 0.3, 0.9]]]
  k: [[[0.6, 0.2, -0.1, 0.4]], [[0.1, -0.7, 0.5, 0.3]], [[-0.8, 0.5, 0.2, -0.6]], [[0.3, 0.1, 0.9, -0.2]]]
  v: [[[0.5, -1.0, 0.3, 0.2]], [[-0.2, 0.4, 1.1, -0.5]], [[0.9, 0.6, -0.8, 0.2]], [[0.4, -0.3, 0.7, 0.6]]]
  w: [[[0.2, -0.5, 0.7, 0.1], [-0.3, 0.4, 0.6, -0.8]], [[0.9, 0.3, -0.2, 0.5], [0.1, -0.6, 0.4, 0.2]], [[-0.7, 0.8, 0.3, -0.4], [0.5, 0.2, -0.9, 0.6]], [[0.3, 0.1, -0.5, 0.8], [-0.2, 0.7, 0.4, -0.6]]]
modules:
  M:
    args: { bm: dim, bn: dim, bbm: dim, bbn: dim }
    inputs:
      q: { type: tensor, shape: [T, H, D], dtype: fp32 }
      k: { type: tensor, shape: [T, G, D], dtype: fp32 }
      v: { type: tensor, shape: [T, G, D], dtype: fp32 }
      w: { type: tensor, shape: [T, H, D], dtype: fp32 }
    outputs:
      loss: { type: tensor, shape: [], dtype: fp32 }
    init:
      INIT
    forward:
      FORWARD
      - loss: (o * w).sum()
    unshown: UNSHOWN
    steps:
      - id: all
        show: q
`;

function attention(init: string, forward: string, unshown: string, block: number[]): Promise<CompiledPage> {
  const [bm, bn, bbm, bbn] = block.map(String);
  const yaml = ATTENTION.replace('BM', bm).replace('BN', bn).replace('BBM', bbm).replace('BBN', bbn).replace('INIT', init).replace('      FORWARD', forward).replace('UNSHOWN', unshown);
  const url = new URL('attention.yaml', examplesUrl).href;
  return loadPage(url, { read: (u) => (u === url ? Promise.resolve(yaml) : readFileUrl(u)) });
}

it.each([
  [2, 1, 1, 2],
  [1, 1, 3, 2],
])('flash_attn_func with blocks %i %i %i %i matches causal softmax attention with GQA in float64', async (...block) => {
  const flash = await attention('kernel: { use: flash_attn_func, args: { softmax_scale: 0.5, block_m: bm, block_n: bn, bwd_block_m: bbm, bwd_block_n: bbn } }', '      - o: kernel(q, k, v)', '[o, loss]', block);
  const eager = await attention(
    '{}',
    ['kr: k.transpose(0, 1).repeat_interleave(2, 0)', 'vr: v.transpose(0, 1).repeat_interleave(2, 0)', 's: q.transpose(0, 1) @ kr.transpose(1, 2) * 0.5', 'sm: s + causal_mask(T, dtype=fp32)', 'p: sm.softmax(-1)', 'o: (p @ vr).transpose(0, 1)'].map((line) => `      - ${line}`).join('\n'),
    '[kr, vr, s, sm, p, o, loss]',
    block,
  );
  for (const name of ['o', 'loss']) expectTensor(flash.nodes.get(name)!.value, { shape: [...eager.nodes.get(name)!.shape], values: [...eager.nodes.get(name)!.value.data] }, name);
  for (const name of ['q', 'k', 'v']) expectTensor(flash.nodes.get(name)!.grad, { shape: [...eager.nodes.get(name)!.shape], values: [...eager.nodes.get(name)!.grad!.data] }, `grad of ${name}`);
});
