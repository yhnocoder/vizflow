import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface DeepseekV4Fixture {
  sequence: number[];
  tid2eid: number[][];
  router_bias: number[];
  indexer_params_without_grad: string[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  selected: number[][];
  grads: Record<string, EncodedTensor>;
}

const fixture = loadFixture<never>('deepseek_v4.json') as unknown as DeepseekV4Fixture;
const INDEXER = 'csa_layer.self_attn.indexer.';
const isIndexer = (path: string) => path.startsWith(INDEXER);

describe('deepseek_v4-dsl.yaml against transformers DeepseekV4Model in float64', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample('deepseek_v4-dsl.yaml');
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
  });

  it('gives every HF parameter a weight on the page and the same routing buffers', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
    expect(params.filter(isIndexer).length).toBe(fixture.indexer_params_without_grad.length);
    expect(page.nodes.get('hca_layer.mlp.gate.bias')!.value.toNested()).toEqual(fixture.router_bias);
    expect(page.nodes.get('csa_layer.mlp.gate.tid2eid')!.value.toNested()).toEqual(fixture.tid2eid);
  });

  it('matches the loss, the streams after each layer and the final norm', () => {
    expect(page.graph.findLoss()!.value.data[0]).toBeCloseTo(fixture.loss, 9);
    expectTensor(page.nodes.get('X1')!.value, fixture.nodes.X1, 'streams after the CSA layer');
    expectTensor(page.nodes.get('X2')!.value, fixture.nodes.X2, 'streams after the HCA layer');
    expectTensor(page.nodes.get('h')!.value, fixture.nodes.h, 'final norm output');
  });

  it('matches the gradient of every weight outside the indexer', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) {
      if (!isIndexer(path)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
    }
  });

  it('returns the same entry indices as the HF indexer, -1 for the invalid ones, and keeps those entries in the mask', () => {
    const sorted = (row: number[]) => [...row].sort((a, b) => a - b);
    const idx = page.nodes.get(`${INDEXER}idx`)!.value.toNested() as number[][];
    const mask = page.nodes.get(`${INDEXER.replace('indexer.', '')}Ms`)!.value.toNested() as number[][];
    idx.forEach((row, t) => {
      expect(sorted(row), `position ${t}`).toEqual(sorted(fixture.selected[t]));
      const kept = mask[t].flatMap((v, w) => (v === 0 ? [w] : []));
      expect(kept, `position ${t}`).toEqual(sorted(fixture.selected[t].filter((w) => w >= 0)));
      mask[t].forEach((v) => expect([0, -Infinity]).toContain(v));
    });
  });

  it('gives the indexer no gradient, as HF runs it under torch.no_grad', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) {
      if (!isIndexer(path)) continue;
      expect(grad.values.every((v) => v === 0), path).toBe(true);
      const pageGrad = page.nodes.get(path)!.grad;
      expect(pageGrad === null || pageGrad.data.every((v) => v === 0), path).toBe(true);
    }
    const nodes = page.graph.nodes.filter((n) => n.name.startsWith(INDEXER) && n.kind === 'op');
    expect(nodes.length).toBeGreaterThan(20);
    for (const node of nodes) expect(node.grad, node.name).toBeNull();
    for (const name of ['csa_layer.mlp.gate.idx', 'hca_layer.mlp.gate.idx']) expect(page.nodes.get(name)!.grad, name).toBeNull();
    const steps = createSteps(page.graph, page.steps);
    expect(steps.list.some((s) => s.dir === 'backward' && s.module?.includes('.indexer'))).toBe(false);
  });
});
