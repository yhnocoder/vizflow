import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { tokenAxes } from '../../src/scene3d/hover';
import { loadExample } from '../dsl/load';

describe('tokenAxes', () => {
  it('follows the token axis through op semantics, not through equal sizes', () => {
    const g = graph({ dims: { T: 4, d: 4 } });
    const x = g.input('x', ['T', 'd'], Array.from({ length: 16 }, (_, i) => i / 10));
    const W = g.param('W', ['d', 'd'], Array.from({ length: 16 }, (_, i) => i / 10));
    const cos = g.buffer('cos', ['T', 'd'], Array.from({ length: 16 }, () => 1));
    const k = g.matmul(x, W, { name: 'k' });
    const kT = g.transpose(k, 0, 1, { name: 'kT' });
    const s = g.matmul(k, kT, { name: 's' });
    const r = g.mul(k, cos, { name: 'r' });
    const m = g.mean(r, -1, { name: 'm' });
    const all = g.sum(r, 0, { name: 'all' });
    const wx = g.matmul(W, k, { name: 'wx' });
    const axes = tokenAxes(g.nodes);
    expect([x, W, cos, k, kT, s, r, m, all, wx].map((n) => axes.get(n))).toEqual([0, undefined, undefined, 0, 1, 0, 0, 0, undefined, undefined]);
  });

  it('marks the Qwen3 activations, keeps weights and tables out', async () => {
    const page = await loadExample('qwen3/qwen3-dsl.yaml');
    const axes = tokenAxes(page.graph.nodes);
    const axis = (name: string) => axes.get(page.nodes.get(name)!);
    const attn = 'layer.self_attn';
    const gqa = `${attn}.eager_attention_forward`;
    expect(['ids', 'x0', 'layer.input_layernorm.out', `${gqa}.S[1]`, `${gqa}.P[1]`, `${gqa}.o`, 'z', 'logp'].map(axis)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(['E', `${attn}.Wq.T`, 'layer.input_layernorm.weight', `${gqa}.kT[1]`].map(axis)).toEqual([undefined, undefined, undefined, 1]);
  });
});
