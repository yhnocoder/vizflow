// @vitest-environment jsdom
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { loadExample } from './load';

const LAYERS = [0, 1];

describe('qwen3-paged-dsl.yaml, attention through a paged KV cache', () => {
  let paged: CompiledPage;
  let plain: CompiledPage;

  beforeAll(async () => {
    paged = await loadExample('qwen3/qwen3-paged-dsl.yaml');
    plain = await loadExample('qwen3/qwen3-infer-dsl.yaml');
  });

  it('generates the same tokens and logits as the page without paging', () => {
    expect(paged.nodes.get('ids')!.value.toNested()).toEqual(plain.nodes.get('ids')!.value.toNested());
    expect(Array.from(paged.nodes.get('z')!.value.data)).toEqual(Array.from(plain.nodes.get('z')!.value.data));
  });

  it('writes position t to slot block_table[t // page_size] * page_size + t % page_size and leaves the other slots empty', () => {
    const T = paged.nodes.get('ids')!.value.size;
    const slots = Array.from({ length: T }, (_, t) => [0, 2, 3][Math.floor(t / 2)] * 2 + (t % 2));
    expect(Array.from(paged.nodes.get('model.paged_attention_indices.write_index')!.value.data)).toEqual(slots);
    for (const i of LAYERS) {
      const cache = paged.nodes.get(`model.layers[${i}].self_attn.eager_paged_attention_forward.k_cache`)!.value.toNested() as number[][][];
      const key = plain.nodes.get(`model.layers[${i}].self_attn.kr`)!.value.toNested() as number[][][];
      cache.forEach((slot, s) => {
        const t = slots.indexOf(s);
        expect(slot).toEqual(t < 0 ? key.map(() => key[0][0].map(() => 0)) : key.map((head) => head[t]));
      });
    }
  });

  it('renders every step text and label in both modes', () => {
    const failures: string[] = [];
    const math = (text: string) => [...text.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)].map((m) => m[1] ?? m[2]);
    for (const [graph, spec] of [[paged.graph, paged.modes!.infer], [paged.train!.graph, paged.train!.steps]] as const) {
      const steps = createSteps(graph, spec);
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
    }
    expect(failures).toEqual([]);
  });
});
