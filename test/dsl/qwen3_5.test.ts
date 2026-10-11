import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, type PageFile, compilePage, parseYaml } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { examplesUrl, loadExample, readFileUrl } from './load';

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

  it('builds forward and backward steps with the chunk loop unrolled per chunk', () => {
    const steps = createSteps(page.graph, page.steps);
    const titles = steps.list.map((s) => s.title);
    for (const i of [1, 2]) expect(titles).toContain(`块 $${i}$：新状态`);
    expect(titles).not.toContain('块 $3$：新状态');
    const rule = 'linear_layer.linear_attn.chunk_gated_delta_rule';
    expect(page.nodes.get(`${rule}.S[2][2]`)!.shape).toEqual([2, 2]);
    expect(page.nodes.get(`${rule}.qp`)!.shape).toEqual([2, 4, 2]);
    expect(page.nodes.get(`${rule}.qp`)!.realShape).toEqual([48, 64, 128]);
    expect(page.nodes.get(`${rule}.pde[1]`)!.shape).toEqual([2, 2, 2]);
  });
});

describe('qwen3_5-dsl.yaml with other chunk sizes', () => {
  it.each([1, 3, 4, 64])('gives the same loss, hidden states and gradients with chunk %i', async (chunk) => {
    const url = new URL('qwen3_5-dsl.yaml', examplesUrl).href;
    const spec = parseYaml(await readFileUrl(url), url) as PageFile & { dims: Record<string, number> };
    spec.dims.chunk = chunk;
    const page = await compilePage(spec, { base: url, read: readFileUrl });
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
    expect(page.nodes.get('linear_layer.linear_attn.chunk_gated_delta_rule.qp')!.shape[1]).toBe(Math.ceil(3 / chunk) * chunk);
    expect(page.graph.findLoss()!.value.data[0]).toBeCloseTo(fixture.loss, 9);
    expectTensor(page.nodes.get('x1')!.value, fixture.nodes.x1, 'linear_attention layer output');
    for (const [path, grad] of Object.entries(fixture.grads)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
  });
});
