// @vitest-environment jsdom
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, parseYaml } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import type { Tensor } from '../../src/tensor';
import { type EncodedTensor, decode, loadFixture, nested } from '../torch/fixtures';
import { examplesUrl, loadExample, readFileUrl } from './load';

interface PackedFixture {
  input_ids: number[];
  position_ids: number[];
  labels: number[];
  weights: Record<string, EncodedTensor>;
  loss: number;
  logits: EncodedTensor;
  grads: Record<string, EncodedTensor>;
}

const fixture = loadFixture<never>('qwen3_packed.json') as unknown as PackedFixture;
const PAGE = 'qwen3/qwen3-packed-dsl.yaml';
const VOCAB = ['on', 'mat', 'the', 'sat', 'cat'];
const FLOAT32_TABLES = 1e-7;

describe('qwen3-packed-dsl.yaml against transformers DataCollatorWithFlattening and Qwen3ForCausalLM in float64', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    const url = new URL(PAGE, examplesUrl).href;
    const spec = parseYaml(await readFileUrl(url), url) as { config: Record<string, unknown>; data: Record<string, unknown> };
    Object.assign(spec.config, { dtype: 'fp64', d_head: 4, rope_theta: 10 });
    const shifted = [...fixture.labels.slice(1), -100];
    Object.assign(spec.data, { tokens: fixture.input_ids.map((id) => VOCAB[id]), position_ids: fixture.position_ids, y: shifted });
    page = await compilePage(spec as never, { base: url, read: readFileUrl });
    for (const [path, tensor] of Object.entries(fixture.weights)) page.nodes.get(path)!.set(nested(tensor));
  });

  it('uses the same packed batch as the data collator', async () => {
    const written = await loadExample(PAGE);
    expect(written.nodes.get('ids')!.value.toNested()).toEqual(fixture.input_ids);
    expect(written.nodes.get('position_ids')!.value.toNested()).toEqual(fixture.position_ids);
    expect(written.nodes.get('y')!.value.toNested()).toEqual([...fixture.labels.slice(1), -100]);
  });

  it('matches the logits, the loss and the gradient of every parameter up to the float32 rotary tables', () => {
    const close = (actual: Tensor | null, expected: EncodedTensor, what: string) => {
      expect([...(actual as Tensor).shape], what).toEqual(expected.shape);
      const far = decode(expected.values).flatMap((e, i) => (Math.abs((actual as Tensor).data[i] - e) <= FLOAT32_TABLES * (1 + Math.abs(e)) ? [] : [`[${i}] ${(actual as Tensor).data[i]} vs ${e}`]));
      expect(far, what).toEqual([]);
    };
    close(page.nodes.get('z')!.value, fixture.logits, 'logits');
    close(page.nodes.get('loss')!.value, { shape: [], values: [fixture.loss] }, 'loss');
    for (const [path, grad] of Object.entries(fixture.grads)) close(page.nodes.get(path)!.grad, grad, `grad of ${path}`);
  });

  it('gives the second sequence the same logits as running it alone', async () => {
    const url = new URL('qwen3/qwen3-dsl.yaml', examplesUrl).href;
    const spec = parseYaml(await readFileUrl(url), url) as { config: Record<string, unknown>; data: Record<string, unknown> };
    Object.assign(spec.config, { dtype: 'fp64', d_head: 4, rope_theta: 10 });
    spec.data.tokens = ['on', 'mat', 'on', 'mat'];
    const alone = await compilePage(spec as never, { base: url, read: readFileUrl });
    for (const [path, tensor] of Object.entries(fixture.weights)) alone.nodes.get(path)!.set(nested(tensor));
    const packed = Array.from(page.nodes.get('z')!.value.data.slice(15));
    Array.from(alone.nodes.get('z')!.value.data.slice(0, 10)).forEach((v, i) => expect(packed[i]).toBeCloseTo(v, 12));
  });

  it('renders every step text and label', async () => {
    const written = await loadExample(PAGE);
    const failures: string[] = [];
    const math = (text: string) => [...text.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)].map((m) => m[1] ?? m[2]);
    const steps = createSteps(written.graph, written.steps);
    steps.list.forEach((info, i) => {
      const { title, body } = steps.text(i);
      for (const tex of [...math(title), ...math(body), ...math(info.label)]) {
        try {
          katex.renderToString(tex, { throwOnError: true, trust: true });
        } catch (error) {
          failures.push(`step ${i} ${title}: ${(error as Error).message}`);
        }
      }
    });
    steps.dispose();
    expect(failures).toEqual([]);
  });
});
