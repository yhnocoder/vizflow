// @vitest-environment jsdom
import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
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
    expect(decode).toEqual(['decode 1：逐行计算', 'decode 1：注意力读取缓存', 'decode 1：逐行计算', '采样第 4 个 token']);
  });

  it('shows one row per decode step and keeps the cache rows', () => {
    const steps = createSteps(page.graph, page.modes!.infer);
    const at = steps.list.findIndex((s) => s.title === 'decode 1：注意力读取缓存');
    steps.go(at);
    const state = steps.state();
    const o = page.nodes.get('layer.self_attn.o')!;
    const k = page.nodes.get('layer.self_attn.kr')!;
    expect(state.active(k)).toBe(true);
    expect(state.row!(o, 1)).toBe('inactive');
    expect(state.row!(o, 2)).toBeUndefined();
    expect(state.row!(o, 3)).toBe('hidden');
    expect(state.row!(k, 0)).toBeUndefined();
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
