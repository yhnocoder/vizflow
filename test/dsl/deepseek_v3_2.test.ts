import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface DeepseekV32Fixture {
  sequence: number[];
  router_bias: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  selected: Record<string, number[][]>;
  grads: Record<string, EncodedTensor>;
}

const fixture = loadFixture<never>('deepseek_v3_2.json') as unknown as DeepseekV32Fixture;
const LAYERS = ['dense_layer', 'moe_layer'];
const KI = 2;
const isIndexer = (path: string) => path.includes('.indexer.');

describe('deepseek_v3_2-dsl.yaml against transformers DeepseekV32Model in float64', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample('deepseek_v3_2-dsl.yaml');
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
  });

  it('gives every HF parameter a weight on the page, the indexer included', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
    for (const layer of LAYERS) expect(params).toContain(`${layer}.self_attn.indexer.k_norm.bias`);
    expect(page.nodes.get('moe_layer.mlp.bias')!.value.toNested()).toEqual(fixture.router_bias);
  });

  it('matches the loss and the hidden states of both layers', () => {
    expect(page.graph.findLoss()!.value.data[0]).toBeCloseTo(fixture.loss, 9);
    expectTensor(page.nodes.get('x1')!.value, fixture.nodes.x1, 'dense layer output');
    expectTensor(page.nodes.get('x2')!.value, fixture.nodes.x2, 'MoE layer output');
    expectTensor(page.nodes.get('h')!.value, fixture.nodes.h, 'final norm output');
  });

  it('matches the gradient of every weight outside the indexer', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) {
      if (!isIndexer(path)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
    }
  });

  it('selects the same keys as the HF indexer, at most kI per row inside the causal window', () => {
    for (const layer of LAYERS) {
      const sel = page.nodes.get(`${layer}.self_attn.indexer.sel`)!.value.toNested() as number[][];
      const mask = page.nodes.get(`${layer}.self_attn.M_sparse`)!.value.toNested() as number[][];
      sel.forEach((row, t) => {
        expect(row.reduce((a, b) => a + b, 0)).toBe(KI);
        const kept = row.flatMap((v, s) => (v === 1 && s <= t ? [s] : []));
        expect(kept.length).toBe(Math.min(KI, t + 1));
        expect(kept).toEqual(fixture.selected[layer][t]);
        mask[t].forEach((m, s) => expect(m).toBe(kept.includes(s) ? 0 : -Infinity));
      });
    }
  });

  it('gives the indexer no gradient, as HF runs it under torch.no_grad', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) {
      if (!isIndexer(path)) continue;
      expect(grad.values.every((v) => v === 0)).toBe(true);
      const pageGrad = page.nodes.get(path)!.grad;
      expect(pageGrad === null || pageGrad.data.every((v) => v === 0), path).toBe(true);
    }
    for (const layer of LAYERS) {
      const prefix = `${layer}.self_attn.indexer.`;
      const nodes = page.graph.nodes.filter((n) => n.name.startsWith(prefix) && n.kind === 'op');
      expect(nodes.length).toBeGreaterThan(10);
      for (const node of nodes) expect(node.grad, node.name).toBeNull();
    }
    const steps = createSteps(page.graph, page.steps);
    const titles = steps.list.map((s) => s.title);
    expect(titles).toContain('top-$k_I$：每个位置选出 $k_I$ 个 key');
    expect(steps.list.some((s) => s.dir === 'backward' && s.module?.includes('.indexer'))).toBe(false);
  });
});
