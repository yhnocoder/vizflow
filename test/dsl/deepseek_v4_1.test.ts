// @vitest-environment jsdom
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { elemTexOf } from '../../src/explain/elem';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface DeepseekV41Fixture {
  ids: number[];
  labels: number[];
  token_types: number[];
  pixels: number[][][];
  patches: EncodedTensor;
  multipliers: EncodedTensor;
  primes: EncodedTensor;
  offsets: EncodedTensor;
  hashes: EncodedTensor;
  router_bias: number[];
  router_bias_vl: number[];
  params_without_grad: string[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  selected: Record<string, number[][]>;
  candidates: number[][];
  grads: Record<string, EncodedTensor>;
}

const fixture = loadFixture<never>('deepseek_v4_1.json') as unknown as DeepseekV41Fixture;
const LAYERS = ['0', '1', '2', '20', '24', '25'];
const noGrad = new Set(fixture.params_without_grad);
const ints = (encoded: EncodedTensor) => encoded.values.map(Number);

describe('deepseek_v4_1-dsl.yaml against the DeepSeek-V4.1 reference model in float64', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample('deepseek_v4_1-dsl.yaml');
    page.graph.batch(() => {
      for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    });
  });

  it('reads the same sequence, image and hash constants as the reference', () => {
    expect(page.sequence!.get()).toEqual([...fixture.ids, fixture.labels[fixture.labels.length - 1]]);
    expect(page.nodes.get('types')!.value.toNested()).toEqual(fixture.token_types);
    expect(page.nodes.get('pixels')!.value.toNested()).toEqual(fixture.pixels);
    expectTensor(page.nodes.get('patches')!.value, { shape: [8, 3, 2, 2], values: fixture.patches.values }, 'patches');
    expect([...page.nodes.get('engram_hash.mult')!.value.int64!].map(String)).toEqual(fixture.multipliers.values);
    expect(Array.from(page.nodes.get('engram_hash.primes')!.value.data)).toEqual(ints(fixture.primes));
    expect(Array.from(page.nodes.get('engram_hash.offsets')!.value.data)).toEqual(ints(fixture.offsets));
    expect(Array.from(page.nodes.get('hash1')!.value.data)).toEqual(ints(fixture.hashes));
  });

  it('gives every reference parameter a weight on the page and the same router biases', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
    for (const layer of LAYERS) {
      expect(page.nodes.get(`layer${layer}.ffn.gate.bias`)!.value.toNested()).toEqual(fixture.router_bias);
      expect(page.nodes.get(`layer${layer}.ffn.gate.bias_vl`)!.value.toNested()).toEqual(fixture.router_bias_vl);
    }
  });

  it('matches the image path, the merged embeddings, the engram output and every layer', () => {
    expectTensor(page.nodes.get('vis')!.value, fixture.nodes.vision, 'ViT output');
    expectTensor(page.nodes.get('img')!.value, fixture.nodes.aligner, 'aligner output');
    expectTensor(page.nodes.get('h')!.value, fixture.nodes.merged, 'merged embeddings');
    expectTensor(page.nodes.get('layer1.Xe')!.value, fixture.nodes.engram1, 'engram output in layer 1');
    for (const layer of LAYERS) {
      expectTensor(page.nodes.get(`X${layer}`)!.value, fixture.nodes[`X${layer}`], `streams after layer ${layer}`);
      expectTensor(page.nodes.get(`pre${layer}`)!.value, fixture.nodes[`pre${layer}`], `pre after layer ${layer}`);
    }
    expectTensor(page.nodes.get('hn')!.value, fixture.nodes.final_norm, 'final norm output');
    expectTensor(page.nodes.get('z')!.value, fixture.nodes.logits, 'logits');
    expect(page.graph.findLoss()!.value.data[0]).toBeCloseTo(fixture.loss, 9);
  });

  it('selects the same compressed entries and candidate blocks as the reference indexers', () => {
    for (const layer of ['2', '20', '24']) {
      expect(page.nodes.get(`layer${layer}.attn.idx`)!.value.toNested(), `layer ${layer}`).toEqual(fixture.selected[`layer${layer}`]);
    }
    expect(page.nodes.get('layer20.attn.cand')!.value.toNested()).toEqual(fixture.candidates);
  });

  it('matches the gradient of every parameter', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) {
      if (noGrad.has(path)) continue;
      expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
    }
  });

  it('gives the indexers, the candidate blocks and the router biases no gradient', () => {
    for (const path of noGrad) {
      expect(fixture.grads[path].values.every((v) => v === 0), path).toBe(true);
      const grad = page.nodes.get(path)!.grad;
      expect(grad === null || grad.data.every((v) => v === 0), path).toBe(true);
    }
    const selection = page.graph.nodes.filter((n) => n.kind === 'op' && /^layer\d+\.attn\.indexer\./.test(n.name));
    expect(selection.length).toBeGreaterThan(40);
    for (const node of selection) expect(node.grad, node.name).toBeNull();
    for (const layer of LAYERS) expect(page.nodes.get(`layer${layer}.ffn.gate.idx`)!.grad, `layer ${layer} routing`).toBeNull();
    const steps = createSteps(page.graph, page.steps);
    expect(steps.list.some((s) => s.dir === 'backward' && s.module?.includes('.indexer'))).toBe(false);
  }, 60000);

  it('renders every step text, label and node TeX, with the modules folded and unfolded', () => {
    const steps = createSteps(page.graph, page.steps);
    const renders = (tex: string, what: string) => expect(() => katex.renderToString(tex, { throwOnError: true }), `${what}: ${tex}`).not.toThrow();
    const math = (text: string) => [...text.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)].map((m) => m[1] ?? m[2]);
    const renderAll = (state: string) => {
      steps.list.forEach((info, i) => {
        const { title, body } = steps.text(i);
        for (const tex of [...math(title), ...math(body), ...math(info.label)]) renders(tex, `${state} step ${i} ${info.id}`);
      });
    };
    renderAll('folded');
    steps.setAllFolded(false);
    renderAll('unfolded');
    for (const n of page.graph.nodes) {
      renders(n.tex, n.name);
      renders(elemTexOf(n, n.shape.map(() => 0)), `${n.name} elem`);
    }
  }, 60000);
});
