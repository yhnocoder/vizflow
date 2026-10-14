import { describe, expect, it } from 'vitest';
import { loadPage } from '../../src/dsl';
import { loadExample } from './load';

function page(dtype: string, forward: string, inputDtype = 'dtype') {
  const yaml = `title: t
model: M
config: { d: 2, dtype: ${dtype} }
data:
  x: [[1, 2]]
modules:
  M:
    args: { d: dim, dtype: dtype }
    inputs:
      x: { type: tensor, shape: [T, d], dtype: ${inputDtype} }
    outputs:
      l: { type: tensor, shape: [], dtype: fp32 }
    init:
      proj: { use: Linear, args: { in_features: d, out_features: d, bias: False } }
${forward}
    steps:
      - id: proj
      - id: l
`;
  return loadPage('mem:/page.yaml', { read: async () => yaml });
}

describe('dtype', () => {
  it('follows the HF Qwen3 dtypes in the Qwen3 example', async () => {
    const p = await loadExample('qwen3/qwen3-dsl.yaml');
    const dtype = (name: string) => p.nodes.get(name)!.dtype;
    expect(p.graph.dtype).toBe('bf16');
    expect(['ids', 'y', 'model.embed_tokens.weight', 'model.e', 'model.rotary_emb.inv_freq', 'model.cos', 'model.M'].map(dtype)).toEqual(['int64', 'int64', 'bf16', 'bf16', 'fp32', 'bf16', 'bf16']);
    const norm = 'model.layers[0].input_layernorm';
    expect(['xf', 'rms', 'xh', 'xc'].map((n) => dtype(`${norm}.${n}`))).toEqual(['fp32', 'fp32', 'fp32', 'bf16']);
    const gqa = 'model.layers[0].self_attn.eager_attention_forward';
    expect(['S', 'ex', 'Z', 'P', 'attn_output'].map((n) => dtype(`${gqa}.${n}`))).toEqual(['bf16', 'fp32', 'fp32', 'bf16', 'bf16']);
    expect(['z', 'zf', 'logp', 'loss'].map(dtype)).toEqual(['bf16', 'fp32', 'fp32', 'fp32']);
  });

  it('sets the parameter dtype from config and rejects an unknown one', async () => {
    const ok = await page('bf16', '    forward:\n      - y: proj(x)\n      - l: y.float().mean()');
    expect(['x', 'proj.weight', 'y', 'l'].map((n) => ok.nodes.get(n)!.dtype)).toEqual(['bf16', 'bf16', 'bf16', 'fp32']);
    await expect(page('float', '    forward:\n      - l: x.mean()')).rejects.toThrow(/config\.dtype: dtype must be one of bool, /);
  });

  it('lets matmul reject an input whose dtype differs from the weight', async () => {
    await expect(page('bf16', '    forward:\n      - y: proj(x.float())\n      - l: y.mean()')).rejects.toThrow(/expected inputs of the same dtype/);
  });
});
