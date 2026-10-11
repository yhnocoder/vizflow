import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, type PageFile, parseYaml } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface PageFixture {
  sequence: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  grads: Record<string, EncodedTensor>;
  routing: Record<string, { expert_idx: number[]; top_k_pos: number[]; token_idx: number[] }>;
}

const PAGES = [
  { page: 'qwen3_moe-dsl.yaml', fixture: 'qwen3_moe.json', model: 'Qwen3MoeModel' },
  { page: 'qwen3_5_moe-dsl.yaml', fixture: 'qwen3_5_moe.json', model: 'Qwen3_5MoeTextModel' },
  { page: 'qwen3_8_moe-dsl.yaml', fixture: 'qwen3_5_moe.json', model: 'Qwen3_5MoeTextModel' },
];

describe.each(PAGES)('$page against transformers $model in float64', ({ page: file, fixture: fixtureFile }) => {
  const fixture = loadFixture<never>(fixtureFile) as unknown as PageFixture;
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample(file);
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
  });

  it('gives every HF parameter a weight on the page', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
  });

  it('matches the loss and the hidden states', () => {
    expect(page.graph.findLoss()!.value.data[0]).toBeCloseTo(fixture.loss, 9);
    for (const [name, tensor] of Object.entries(fixture.nodes)) expectTensor(page.nodes.get(name)!.value, tensor, name);
  });

  it('matches the gradient of every weight', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
  });

  it('sends each expert its tokens in the order of the HF expert loop', () => {
    for (const [path, order] of Object.entries(fixture.routing)) {
      for (const [column, expected] of Object.entries(order)) {
        expect(Array.from(page.nodes.get(`${path}.${column}`)!.value.data), `${path}.${column}`).toEqual(expected);
      }
    }
  });

  it('builds the router and expert steps', () => {
    const titles = createSteps(page.graph, page.steps).list.map((s) => s.title);
    expect(titles).toContain('top-$2$：选出 $K$ 个专家的编号');
    expect(titles).toContain('softmax：按行求和');
    expect(titles).toContain('列出每个专家的 (槽位, 位置)');
  });
});

describe('SparseMoE with norm_topk_prob at its default', () => {
  it('casts the top-k probabilities back to the logits dtype without normalising them', async () => {
    const page = await compilePage(parseYaml(`
title: t
import: { SparseMoE: vizflow/nn/moe }
dims: { d: 2 }
inputs:
  x: { value: [[0.3, -0.8], [1.1, 0.2], [-0.5, 0.9]] }
body:
  - mlp: { use: SparseMoE, params: { E: 3, K: 2, mf: 2 }, inputs: { x: x }, outputs: { out: y } }
  - loss: sum(y)
`, 'mem:/page.yaml') as PageFile);
    expect(page.nodes.get('mlp.rw')).toBeUndefined();
    expect(Array.from(page.nodes.get('mlp.topc')!.value.data)).toEqual(Array.from(page.nodes.get('mlp.top')!.value.data));
    const titles = createSteps(page.graph, page.steps).list.map((s) => s.title);
    expect(titles).toContain('路由权重转回分数的 dtype');
    expect(titles).not.toContain('归一化成路由权重');
  });
});
