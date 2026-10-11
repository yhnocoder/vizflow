// @vitest-environment jsdom
import { beforeAll, describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import type { CompiledPage } from '../../src/dsl';
import { positionsOf } from '../../src/explain';
import { analyzeCache, checkCache, createSteps } from '../../src/steps';
import { loadExample } from './load';

describe('qwen3-infer-dsl: generate and optimizer', () => {
  let page: CompiledPage;
  beforeAll(async () => {
    page = await loadExample('qwen3-infer-dsl.yaml');
  });

  it('fills the sequence greedily from the prompt', () => {
    const ids = page.sequence!.get();
    const logp = page.nodes.get('logp')!;
    const V = logp.shape[1];
    expect(ids.slice(0, 2)).toEqual([2, 4]);
    for (let t = 1; t < ids.length - 1; t++) {
      const row = Array.from(logp.value.data.subarray(t * V, (t + 1) * V));
      expect(ids[t + 1]).toBe(row.indexOf(Math.max(...row)));
    }
  });

  it('builds prefill, sample and decode steps without backward', () => {
    const steps = createSteps(page.graph, page.modes!.infer);
    const segments = [...new Set(steps.list.map((s) => s.segment))];
    expect(segments).toEqual(expect.arrayContaining(['prefill', 'decode 1', 'decode 2']));
    expect(steps.list.every((s) => s.dir === 'forward')).toBe(true);
    const decode = steps.list.filter((s) => s.segment === 'decode 1').map((s) => s.title);
    expect(decode.slice(0, 4)).toEqual(['decode 1：第 3 个 token 进入模型', 'decode 1：更新缓存', 'decode 1：如果不用缓存', 'decode 1：从缓存恢复']);
    expect(decode[decode.length - 1]).toBe('采样第 4 个 token');
    const compact = steps.list.filter((s) => s.segment === 'decode 2').map((s) => s.title);
    expect(compact).toEqual(['decode 2：第 4 个 token 进入模型', 'decode 2：读取 4 行缓存', 'decode 2：逐位置运算', '采样第 5 个 token']);
  });

  it('lights only the new position, reads every cached row and hides future score columns', () => {
    const steps = createSteps(page.graph, page.modes!.infer);
    const state = () => steps.state();
    const node = (name: string) => page.nodes.get(name)!;
    const k = node('layer.self_attn.apply_rotary_pos_emb.k_embed');
    const kp = node('layer.self_attn.eager_attention_forward.kp[1]');
    const S = node('layer.self_attn.eager_attention_forward.S[1]');
    steps.go(steps.list.findIndex((s) => s.title === 'decode 1：更新缓存'));
    expect(state().row!(k, [0, 0, 0])).toBe('inactive');
    expect(state().row!(k, [2, 0, 0])).toBeUndefined();
    expect(state().row!(k, [3, 0, 0])).toBe('hidden');
    steps.go(steps.list.findIndex((s) => s.title === 'decode 1：如果不用缓存'));
    expect(state().row!(k, [0, 0, 0])).toBeUndefined();
    steps.go(steps.list.findIndex((s) => s.title.startsWith('decode 1：打分')));
    expect(state().row!(kp, [0, 0])).toBeUndefined();
    expect(state().row!(S, [1, 0])).toBe('inactive');
    expect(state().row!(S, [2, 2])).toBeUndefined();
    expect(state().row!(S, [2, 3])).toBe('hidden');
    const lm = steps.list.findIndex((s) => s.segment === 'prefill' && s.nodes.includes(node('z')));
    steps.go(lm);
    expect(state().row!(node('z'), [0, 0])).toBe('inactive');
    expect(state().row!(node('z'), [1, 0])).toBeUndefined();
  });

  it('applies one AdamW step on entering the update step and restores it on leaving', () => {
    const steps = createSteps(page.graph, page.modes!.train);
    const loss = page.graph.findLoss()!;
    const before = loss.value.data[0];
    const W = page.nodes.get('W')!;
    const w0 = Array.from(W.value.data);
    steps.go(steps.list.length - 1);
    expect(steps.list[steps.current].id).toBe('u:adamw');
    expect(loss.value.data[0]).toBeLessThan(before);
    expect(Array.from(W.value.data)).not.toEqual(w0);
    steps.go(0);
    expect(Array.from(W.value.data)).toEqual(w0);
    expect(loss.value.data[0]).toBe(before);
  });
});

describe('cache analysis', () => {
  const scopeOf = (page: CompiledPage) => {
    const label = page.graph.nodes.find((n) => n.op === 'nll_loss')!.inputs[1];
    const scored = new Set<Node>();
    for (const n of page.graph.nodes) if (n.inputs.some((i) => i === label || scored.has(i))) scored.add(n);
    return new Set(page.graph.nodes.filter((n) => !scored.has(n)));
  };

  it('dates every element by the last position it depends on, ignoring masked entries', async () => {
    const page = await loadExample('qwen3-infer-dsl.yaml');
    const positions = positionsOf(page.graph, 4);
    const Z = page.nodes.get('layer.self_attn.eager_attention_forward.Z[1]')!;
    const S = page.nodes.get('layer.self_attn.eager_attention_forward.S[1]')!;
    expect([0, 1, 2, 3].map((t) => positions.horizon(Z, t))).toEqual([0, 1, 2, 3]);
    expect(positions.horizon(S, 1)).toBe(1);
    expect(positions.horizon(S, 4 * 2 + 1)).toBe(2);
  });

  it('reports the tensors a decode step reads when no module caches them', async () => {
    const page = await loadExample('qwen3-infer-dsl.yaml');
    const logits = page.nodes.get('logp')!;
    const problem = checkCache(page.graph, 4, 2, [], scopeOf(page), [logits]);
    expect(problem).toContain("'layer.self_attn.eager_attention_forward.kT[1]'");
    expect(problem).toContain("'layer.self_attn.eager_attention_forward.V[1]'");
  });

  it('tells a growing cache, a sliding window and a fixed state apart', async () => {
    const page = await loadExample('qwen3_5-dsl.yaml');
    const caches = page.modes!.infer.infer!.caches;
    const analysis = analyzeCache(page.graph, 4, caches, scopeOf(page), [page.nodes.get('logp')!]);
    const kind = (name: string) => analysis.kind(caches.find((c) => c.nodes.some((n) => n.name === name))!);
    expect(kind('linear_layer.linear_attn.gated_delta_rule.S[1]')).toBe('state');
    expect(kind('linear_layer.linear_attn.mixed_qkv')).toBe('window');
    expect(kind('full_layer.self_attn.v')).toBe('rows');
    const mla = await loadExample('deepseek_v3-dsl.yaml');
    expect(mla.modes!.infer.infer!.caches.map((c) => c.nodes.map((n) => n.name))).toEqual([
      ['dense_layer.self_attn.kv_a_layernorm.out', 'dense_layer.self_attn.apply_rotary_pos_emb.k_embed'],
      ['moe_layer.self_attn.kv_a_layernorm.out', 'moe_layer.self_attn.apply_rotary_pos_emb.k_embed'],
    ]);
  });
});
