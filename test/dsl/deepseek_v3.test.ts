import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface DeepseekV3Fixture {
  sequence: number[];
  router_bias: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  grads: Record<string, EncodedTensor>;
}

const fixture = loadFixture<never>('deepseek_v3.json') as unknown as DeepseekV3Fixture;

describe('deepseek_v3-dsl.yaml against transformers DeepseekV3Model in float64', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample('deepseek_v3-dsl.yaml');
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
  });

  it('gives every HF parameter a weight on the page and uses the same router bias', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
    expect(page.nodes.get('moe_layer.mlp.bias')!.value.toNested()).toEqual(fixture.router_bias);
  });

  it('matches the loss and the hidden states of both layers', () => {
    expect(page.graph.findLoss()!.value.data[0]).toBeCloseTo(fixture.loss, 9);
    expectTensor(page.nodes.get('x1')!.value, fixture.nodes.x1, 'dense layer output');
    expectTensor(page.nodes.get('x2')!.value, fixture.nodes.x2, 'MoE layer output');
    expectTensor(page.nodes.get('h')!.value, fixture.nodes.h, 'final norm output');
  });

  it('matches the gradient of every weight', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
  });

  it('routes each position to K experts from the top group, and expert selection has no gradient', () => {
    const sel = page.nodes.get('moe_layer.mlp.sel')!;
    for (const row of sel.value.toNested() as number[][]) {
      expect(row.reduce((a, b) => a + b, 0)).toBe(2);
      expect(row.slice(0, 3).some((v) => v === 1) && row.slice(3).some((v) => v === 1)).toBe(false);
    }
    for (const name of ['sc', 'gs', 'gm', 'em', 'choice', 'sel']) expect(page.nodes.get(`moe_layer.mlp.${name}`)!.grad).toBeNull();
    const steps = createSteps(page.graph, page.modes?.train ?? page.steps);
    expect(steps.list.map((s) => s.title)).toContain('专家的权重');
  });
});
