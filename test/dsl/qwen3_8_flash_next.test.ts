// @vitest-environment jsdom
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { elemTexOf } from '../../src/explain/elem';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface Int64Tensor {
  shape: number[];
  values: string[];
}

interface FlashNextFixture {
  sequence: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  nodes: Record<string, EncodedTensor>;
  int64: Record<string, Int64Tensor>;
  no_grad: string[];
  grads: Record<string, EncodedTensor>;
}

describe('qwen3_8_flash_next-dsl.yaml against transformers Qwen4ExpTextModel in float64', () => {
  const fixture = loadFixture<never>('qwen3_8_flash_next.json') as unknown as FlashNextFixture;
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample('qwen3_8_flash_next-dsl.yaml');
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
  });

  it('gives every HF parameter a weight on the page', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
  });

  it('holds the HF hash constants and computes the same n-gram ids', () => {
    for (const [path, expected] of Object.entries(fixture.int64)) {
      const value = page.nodes.get(path)!.value;
      expect([...value.shape], path).toEqual(expected.shape);
      const values = value.int64 === undefined ? [...value.data].map((v) => String(v)) : [...value.int64].map((v) => v.toString());
      expect(values, path).toEqual(expected.values);
    }
  });

  it('matches the loss, both layer outputs, h and the indexer selection', () => {
    expect(page.graph.findLoss()!.value.data[0]).toBeCloseTo(fixture.loss, 9);
    for (const [name, tensor] of Object.entries(fixture.nodes)) expectTensor(page.nodes.get(name)!.value, tensor, name);
  });

  it('matches the gradient of every weight, with zeros where HF leaves the indexer without a gradient', () => {
    expect(fixture.no_grad.every((name) => name.includes('.indexer.'))).toBe(true);
    for (const [path, grad] of Object.entries(fixture.grads)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
  });

  it('builds the PLE, hyper-connection and indexer steps', () => {
    const titles = createSteps(page.graph, page.steps).list.map((s) => s.title);
    expect(titles).toContain('$2$-gram：对每个 head 的质数取模');
    expect(titles).toContain('$3$-gram：对每个 head 的质数取模');
    expect(titles).toContain('子层的输出乘写回系数');
    expect(titles).toContain('每个 query 选分数最高的块');
  });

  it('renders every step text, label and node TeX', () => {
    const steps = createSteps(page.graph, page.steps);
    const renders = (tex: string, what: string) => expect(() => katex.renderToString(tex, { throwOnError: true }), `${what}: ${tex}`).not.toThrow();
    const math = (text: string) => [...text.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)].map((m) => m[1] ?? m[2]);
    steps.list.forEach((info, i) => {
      const { title, body } = steps.text(i);
      for (const tex of [...math(title), ...math(body), ...math(info.label)]) renders(tex, `step ${i}`);
    });
    for (const n of page.graph.nodes) {
      renders(n.tex, n.name);
      renders(elemTexOf(n, n.shape.map(() => 0)), `${n.name} elem`);
    }
  });
});
