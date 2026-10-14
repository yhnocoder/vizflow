// @vitest-environment jsdom
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, parseYaml } from '../../src/dsl';
import { elemTexOf } from '../../src/explain/elem';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { examplesUrl, loadExample, readFileUrl } from './load';

interface DeepseekV3Fixture {
  sequence: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  logits: EncodedTensor;
  grads: Record<string, EncodedTensor>;
  routing: { topk_indices: number[][]; experts: Record<string, number[]> };
}

const fixture = loadFixture<never>('deepseek_v3.json') as unknown as DeepseekV3Fixture;
const PAGE = 'deepseek_v3/deepseek_v3-dsl.yaml';
const MOE = 'model.layers[1].mlp';

async function loadInFloat64(): Promise<CompiledPage> {
  const url = new URL(PAGE, examplesUrl).href;
  const spec = parseYaml(await readFileUrl(url), url) as { config: Record<string, unknown> };
  spec.config.dtype = 'fp64';
  return compilePage(spec as never, { base: url, read: readFileUrl });
}

describe('deepseek_v3-dsl.yaml against transformers DeepseekV3ForCausalLM in float64', () => {
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

  it('routes each position to the same experts as HF and gives each expert the same tokens', () => {
    const node = (name: string) => page.nodes.get(`${MOE}.${name}`)!;
    const picked = node('gate.topk_indices').value.toNested() as number[][];
    picked.forEach((row, t) => expect([...row].sort(), `position ${t}`).toEqual([...fixture.routing.topk_indices[t]].sort()));
    const experts = Array.from(node('experts.expert_idx').value.data);
    const tokens = Array.from(node('experts.token_idx').value.data);
    const tokensOf: Record<string, number[]> = {};
    experts.forEach((e, r) => (tokensOf[String(e)] ??= []).push(tokens[r]));
    expect(tokensOf).toEqual(fixture.routing.experts);
    for (const name of ['gate.scores_for_choice', 'gate.group_scores', 'gate.masked_scores', 'gate.topk_indices']) expect(node(name).grad, name).toBeNull();
  });
});

describe('deepseek_v3-dsl.yaml as written', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample(PAGE);
  });

  it('builds a dense first layer and an MoE second layer from first_k_dense_replace', () => {
    expect(page.nodes.has('model.layers[0].mlp.gate_proj.weight')).toBe(true);
    expect(page.nodes.has('model.layers[1].mlp.gate.weight')).toBe(true);
    expect(page.nodes.get('model.layers[1].mlp.gate.router_logits')!.dtype).toBe('fp32');
    expect(page.nodes.get('model.layers[1].mlp.experts.cast')!.dtype).toBe('bf16');
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
