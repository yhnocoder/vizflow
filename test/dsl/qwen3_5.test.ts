import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface Qwen35Fixture {
  sequence: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  grads: Record<string, EncodedTensor>;
}

const fixture = loadFixture<never>('qwen3_5.json') as unknown as Qwen35Fixture;

describe('qwen3_5-dsl.yaml against transformers Qwen3_5TextModel in float64', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample('qwen3_5-dsl.yaml');
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
  });

  it('gives every HF parameter a weight on the page', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
  });

  it('matches the loss and the hidden states of both layers', () => {
    expect(page.graph.findLoss()!.value.data[0]).toBeCloseTo(fixture.loss, 9);
    expectTensor(page.nodes.get('x1')!.value, fixture.nodes.x1, 'linear_attention layer output');
    expectTensor(page.nodes.get('x')!.value, fixture.nodes.x, 'full_attention layer output');
    expectTensor(page.nodes.get('h')!.value, fixture.nodes.h, 'final norm output');
  });

  it('matches the gradient of every weight', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
  });

  it('builds forward and backward steps with the recurrence unrolled per position', () => {
    const steps = createSteps(page.graph, page.modes?.train ?? page.steps);
    const titles = steps.list.map((s) => s.title);
    for (const t of [1, 2, 3, 4]) expect(titles).toContain(`位置 $${t}$：delta rule 写入`);
    expect(page.nodes.get('linear_layer.linear_attn.gated_delta_rule.S[3]')!.shape).toEqual([2, 2, 2]);
  });
});
