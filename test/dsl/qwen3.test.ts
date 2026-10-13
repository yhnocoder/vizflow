// @vitest-environment jsdom
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, parseYaml } from '../../src/dsl';
import { elemTexOf } from '../../src/explain/elem';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { examplesUrl, loadExample, readFileUrl } from './load';

interface Qwen3Fixture {
  sequence: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  logits: EncodedTensor;
  grads: Record<string, EncodedTensor>;
}

const fixture = loadFixture<never>('qwen3.json') as unknown as Qwen3Fixture;
const PAGE = 'qwen3/qwen3-dsl.yaml';

async function loadInFloat64(): Promise<CompiledPage> {
  const url = new URL(PAGE, examplesUrl).href;
  const spec = parseYaml(await readFileUrl(url), url) as { config: Record<string, unknown> };
  spec.config.dtype = 'fp64';
  return compilePage(spec as never, { base: url, read: readFileUrl });
}

describe('qwen3-dsl.yaml against transformers Qwen3ForCausalLM in float64', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadInFloat64();
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
  });

  it('names every parameter by its state_dict key', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
  });

  it('matches the logits and the loss', () => {
    expectTensor(page.nodes.get('z')!.value, fixture.logits, 'logits');
    expectTensor(page.nodes.get('loss')!.value, { shape: [], values: [fixture.loss] }, 'loss');
  });

  it('matches the gradient of every parameter', () => {
    for (const [path, grad] of Object.entries(fixture.grads)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
  });
});

describe('qwen3-dsl.yaml as written', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample(PAGE);
  });

  it('computes in bf16 and casts the logits to float32 before the loss', () => {
    expect(page.nodes.get('e')!.dtype).toBe('bf16');
    expect(page.nodes.get('zf')!.dtype).toBe('fp32');
    expect(page.nodes.get('loss')!.dtype).toBe('fp32');
  });

  it('builds the forward and backward steps of both layers', () => {
    const steps = createSteps(page.graph, page.steps);
    const titles = steps.list.map((s) => s.title);
    expect(titles[0]).toBe('模型如何预测下一个词');
    expect(steps.list.filter((s) => s.title === '乘 sin').length).toBeGreaterThan(0);
    expect(page.steps.modules.map((m) => m.id)).toEqual(expect.arrayContaining(['layers[0]', 'layers[1]', 'layers[0].self_attn', 'layers[1].mlp']));
    steps.dispose();
  });

  it('renders every step text, label and node TeX, folded or not', () => {
    const failures = new Set<string>();
    const renders = (tex: string, what: string) => {
      try {
        katex.renderToString(tex, { throwOnError: true, trust: true });
      } catch (error) {
        failures.add(`${what}: ${(error as Error).message}`);
      }
    };
    const math = (text: string) => [...text.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)].map((m) => m[1] ?? m[2]);
    const steps = createSteps(page.graph, page.steps);
    const check = () =>
      steps.list.forEach((info, i) => {
        const { title, body } = steps.text(i);
        for (const tex of [...math(title), ...math(body), ...math(info.label)]) renders(tex, `step ${i} ${title}`);
      });
    check();
    steps.setAllFolded(true);
    check();
    steps.dispose();
    for (const m of page.steps.modules) if (m.formula !== undefined) renders(m.formula, `formula of ${m.id}`);
    for (const n of page.graph.nodes) {
      renders(n.tex, n.name);
      renders(elemTexOf(n, n.shape.map(() => 0)), `${n.name} elem`);
    }
    expect([...failures]).toEqual([]);
  });

  it('replaces the token sequence and recomputes ids and y from it', async () => {
    const fresh = await loadExample(PAGE);
    const sequence = fresh.sequence!;
    expect(sequence.tokens).toEqual(['on', 'mat', 'the', 'sat', 'cat']);
    expect(sequence.get()).toEqual([2, 4, 3, 0]);
    sequence.set([4, 1, 0, 3]);
    expect(Array.from(fresh.nodes.get('ids')!.value.data)).toEqual([4, 1, 0]);
    expect(Array.from(fresh.nodes.get('y')!.value.data)).toEqual([1, 0, 3]);
    expect(() => sequence.set([0, 1, 2])).toThrow(/expected 4 token ids/);
  });
});
