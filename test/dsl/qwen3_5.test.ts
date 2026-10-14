// @vitest-environment jsdom
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, parseYaml } from '../../src/dsl';
import { elemTexOf } from '../../src/explain/elem';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { examplesUrl, loadExample, readFileUrl } from './load';

interface Qwen3_5Fixture {
  sequence: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  logits: EncodedTensor;
  grads: Record<string, EncodedTensor>;
}

const fixture = loadFixture<never>('qwen3_5.json') as unknown as Qwen3_5Fixture;
const PAGE = 'qwen3_5/qwen3_5-dsl.yaml';

async function loadInFloat64(chunkSize: number): Promise<CompiledPage> {
  const url = new URL(PAGE, examplesUrl).href;
  const spec = parseYaml(await readFileUrl(url), url) as { config: Record<string, unknown> };
  spec.config.dtype = 'fp64';
  spec.config.chunk_size = chunkSize;
  return compilePage(spec as never, { base: url, read: readFileUrl });
}

describe.each([1, 2, 3, 64])('qwen3_5-dsl.yaml with chunk_size %i against transformers Qwen3_5ForCausalLM in float64', (chunkSize) => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadInFloat64(chunkSize);
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
    page.sequence!.set(fixture.sequence);
  });

  it('names every parameter by its state_dict key', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    expect(new Set(params)).toEqual(new Set(Object.keys(fixture.weights)));
  });

  it('matches the logits, the loss and the gradient of every parameter', () => {
    expectTensor(page.nodes.get('z')!.value, fixture.logits, 'logits');
    expectTensor(page.nodes.get('loss')!.value, { shape: [], values: [fixture.loss] }, 'loss');
    for (const [path, grad] of Object.entries(fixture.grads)) expectTensor(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
  });
});

describe('qwen3_5-dsl.yaml as written', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample(PAGE);
  });

  it('builds each layer from its layer type', () => {
    const ids = page.modules.map((m) => m.id);
    expect(ids).toEqual(expect.arrayContaining(['model.layers[0].linear_attn', 'model.layers[1].self_attn', 'model.layers[0].linear_attn.chunk_gated_delta_rule']));
    expect(ids).not.toContain('model.layers[0].self_attn');
    expect(ids).not.toContain('model.layers[1].linear_attn');
    expect(page.nodes.get('model.layers[0].linear_attn.g')!.dtype).toBe('fp32');
    expect(page.nodes.get('model.e')!.dtype).toBe('bf16');
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
    for (const m of page.modules) if (m.formula !== undefined) renders(m.formula, `formula of ${m.id}`);
    for (const n of page.graph.nodes) {
      renders(n.tex, n.name);
      renders(elemTexOf(n, n.shape.map(() => 0)), `${n.name} elem`);
    }
    expect([...failures]).toEqual([]);
  });
});
