import { describe, expect, it } from 'vitest';
import { type PageFile, compilePage } from '../../src/dsl';
import { parseYaml } from '../../src/dsl/source';
import { loadExample } from './load';

describe('dtype in page files', () => {
  it('follows the HF Qwen3 dtypes in the Qwen3 example', async () => {
    const page = await loadExample('qwen3-dsl.yaml');
    const dtype = (name: string) => page.nodes.get(name)!.dtype;
    expect(page.graph.dtype).toBe('bf16');
    expect(['ids', 'y', 'E', 'x0'].map(dtype)).toEqual(['int64', 'int64', 'bf16', 'bf16']);
    expect(['sq', 'ms', 'r', 'xh', 'out'].map((n) => dtype(`layer.input_layernorm.${n}`))).toEqual(['fp32', 'fp32', 'fp32', 'fp32', 'bf16']);
    const gqa = 'layer.self_attn.eager_attention_forward';
    expect([`${gqa}.ex[1]`, `${gqa}.Z[1]`, `${gqa}.P[1]`, `${gqa}.o`].map(dtype)).toEqual(['fp32', 'fp32', 'bf16', 'bf16']);
    expect(['z', 'LSE', 'logp', 'loss'].map(dtype)).toEqual(['bf16', 'fp32', 'fp32', 'fp32']);
  });

  it('rejects an unknown dtype and a dtype that names no tensor', async () => {
    const page = (dtype: string, item = 'b: a * 2') =>
      compilePage(parseYaml(`title: t\ndtype: ${dtype}\ndims: { V: 2 }\ninputs: { a: { value: [1, 2] } }\nbody:\n  - ${item}\n`, 'page.yaml') as PageFile);
    await expect(page('float')).rejects.toThrow(/unknown dtype 'float'/);
    await expect(page('bf16', "b: { op: a * 2, dtype: c }")).rejects.toThrow(/c/);
    const ok = await page('bf16', "b: { op: a * 2, dtype: fp32 }");
    expect(['a', 'b'].map((n) => ok.nodes.get(n)!.dtype)).toEqual(['bf16', 'fp32']);
  });

  it('rounds every value to its node dtype when the page sets rounding', async () => {
    const page = (rounding: string) =>
      compilePage(parseYaml(`title: t\ndtype: bf16\n${rounding}dims: { V: 2 }\ninputs: { a: { value: [0.1, 0.3] } }\nbody:\n  - b: a * 3\n`, 'page.yaml') as PageFile);
    const exact = await page('');
    expect([...exact.nodes.get('b')!.value.data]).toEqual([0.1 * 3, 0.3 * 3]);
    const rounded = await page('rounding: true\n');
    expect([...rounded.nodes.get('a')!.value.data]).toEqual([0.10009765625, 0.30078125]);
    expect([...rounded.nodes.get('b')!.value.data]).toEqual([0.30078125, 0.90234375]);
    await expect(page('rounding: bf16\n')).rejects.toThrow(/rounding must be true or false/);
  });
});
