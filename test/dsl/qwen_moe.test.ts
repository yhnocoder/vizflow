import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface PageFixture {
  sequence: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  grads: Record<string, EncodedTensor>;
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

  it('builds the router steps', () => {
    const titles = createSteps(page.graph, page.steps).list.map((s) => s.title);
    expect(titles).toContain('top-$2$：选出 $K$ 个专家');
  });
});
