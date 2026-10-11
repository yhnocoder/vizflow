import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { expectRouting } from './deepseek_moe';
import { loadExample } from './load';

interface DeepseekV3Fixture {
  sequence: number[];
  router_bias: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  grads: Record<string, EncodedTensor>;
  routing: { topk_indices: number[][]; experts: Record<string, number[]> };
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

  it('routes each position to the same K experts as HF, all from the top group, and expert selection has no gradient', () => {
    expectRouting(page, fixture.routing);
    const dtype = (name: string) => page.nodes.get(`moe_layer.mlp.${name}`)!.dtype;
    expect([dtype('logits'), dtype('rw'), dtype('score_mask_b'), dtype('experts.scaled'), dtype('experts.cast')]).toEqual(['fp32', 'fp32', 'bool', 'fp32', 'bf16']);
    const steps = createSteps(page.graph, page.steps);
    expect(steps.list.map((s) => s.title)).toContain('乘 routed_scaling_factor');
  });
});
