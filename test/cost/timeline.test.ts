// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { builtinOps, graph, opDef } from '../../src/core';
import { FLOP_RULES, type MemoryKind, type StepCost, costTimeline, createSteps } from '../../src/steps';
import { loadExample } from '../dsl/load';

const byKind = (step: StepCost) => {
  const out: Partial<Record<MemoryKind, string[]>> = {};
  for (const m of step.memory) (out[m.kind] ??= []).push(m.node.name);
  return out;
};

const bytesOf = (step: StepCost, kind: MemoryKind) => step.memory.filter((m) => m.kind === kind).reduce((sum, m) => sum + m.bytes, 0);

describe('costTimeline on a small graph', () => {
  const g = graph({ dims: { T: 2, d: 3, f: 4 } });
  const x = g.buffer('x', ['T', 'd'], [[1, 2, 3], [4, 5, 6]]);
  const W = g.param('W', ['d', 'f'], [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]);
  const h = g.matmul(x, W, { name: 'h' });
  const a = g.relu(h, { name: 'a' });
  const s = g.sum(a, null, { name: 's' });
  const steps = createSteps(g, { forward: [{ show: h, title: 'h' }, { show: a, title: 'a' }, { show: s, title: 's' }] });
  const t = costTimeline(g, steps.list);

  it('keeps a forward result until its last consumer runs, and keeps what backward saves', () => {
    expect(byKind(t.steps[1])).toEqual({ buffer: ['x'], param: ['W'], temp: ['h'] });
    expect(byKind(t.steps[2])).toEqual({ buffer: ['x'], param: ['W'], saved: ['a'] });
    expect(byKind(t.steps[3])).toEqual({ buffer: ['x'], param: ['W'], saved: ['a'], temp: ['s'] });
    expect(t.steps[1].bytes).toBe((6 + 12 + 8) * 4);
  });

  it('frees saved tensors after their backward and keeps parameter gradients', () => {
    const backward = t.steps.filter((_, i) => steps.list[i].dir === 'backward');
    expect(byKind(backward[0])).toEqual({ buffer: ['x'], param: ['W'], saved: ['a'], grad: ['a'] });
    expect(byKind(backward[backward.length - 1])).toEqual({ buffer: ['x'], param: ['W'], param_grad: ['W'] });
  });

  it('counts FLOPs per step from the op definitions', () => {
    expect(t.steps.map((step) => step.flops)).toEqual([0, 2 * 2 * 3 * 4, 8, 8, 0, 2 * 8 + 2 * 2 * 3 * 4]);
    expect(t.forward).toBe(48 + 16);
    expect(t.backward).toBe(16 + 48);
    expect(t.steps[0]).toMatchObject({ overview: true, bytes: t.peak, forward: t.forward, backward: t.backward });
  });
});

describe('op costs', () => {
  it('gives every builtin op a cost and exactly one row in the FLOP table', () => {
    for (const def of builtinOps) {
      expect(def.cost, def.name).toBeDefined();
      expect(FLOP_RULES.filter((rule) => rule.ops.includes(def.name)), def.name).toHaveLength(1);
    }
    expect(FLOP_RULES.flatMap((rule) => rule.ops).every((name) => opDef(name) !== undefined)).toBe(true);
  });

  it('matches the FLOP table', () => {
    const flops = (name: string, inputs: number[][], out: number[]) => {
      const { forward, backward } = opDef(name)!.cost!.flops(inputs, out);
      const product = (t: number[]) => t.reduce((p, v) => p * v, 1);
      return [product(forward), ...backward.map(product)];
    };
    expect(flops('matmul', [[2, 3], [3, 5]], [2, 5])).toEqual([60, 60, 60]);
    expect(flops('matmul', [[2, 3, 4], [2, 4, 5]], [2, 3, 5])).toEqual([240, 240, 240]);
    expect(flops('repeat_kv', [[1, 3, 4]], [2, 3, 4])).toEqual([0, 24]);
    for (const name of ['view', 'reshape', 'transpose', 'unsqueeze', 'contiguous']) expect(flops(name, [[2, 3, 4]], [2, 3, 4]), name).toEqual([0, 0]);
    expect(flops('add', [[2, 3], [3]], [2, 3])).toEqual([6, 0, 6]);
    expect(flops('sub', [[2, 3], [2, 3]], [2, 3])).toEqual([6, 0, 6]);
    expect(flops('div', [[2, 3], [2, 3]], [2, 3])).toEqual([6, 6, 18]);
    expect(flops('silu', [[2, 3]], [2, 3])).toEqual([6, 12]);
    expect(flops('softmax', [[2, 3]], [2, 3])).toEqual([30, 24]);
    expect(flops('logsumexp', [[2, 3]], [2, 1])).toEqual([24, 18]);
    expect(flops('mean', [[2, 3]], [2])).toEqual([6, 6]);
    expect(flops('nll_loss', [[2, 5], [2]], [2])).toEqual([2, 2, 0]);
    expect(flops('embedding', [[2], [5, 3]], [2, 3])).toEqual([0, 6, 6]);
    expect(flops('causal_conv1d', [[4, 3], [3, 2]], [4, 3])).toEqual([48, 48, 48]);
  });
});

describe('costTimeline at real sizes', () => {
  const g = graph({ dims: { T: 2, d: 3, H: 2 }, real: { T: 8, d: 3, H: 6 } });
  const x = g.input('x', ['T', 'd'], [[1, 2, 3], [4, 5, 6]]);
  const W = g.param('W', ['d', 'd'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const heads = [1, 2].map((h) => g.realCopies(2, 'H', () => g.matmul(x, W, { name: `y${h}` })));
  const steps = createSteps(g, { forward: [{ show: heads, title: 'y' }] });

  it('scales loop copies by real over drawn, batched tensors by B, and follows a new T', () => {
    expect(heads[0].copies).toEqual({ drawn: 2, real: ['H'] });
    expect(g.realShapes({ T: 4 })[heads[0].id]).toEqual([4, 3]);
    const t = costTimeline(g, steps.list, { real: true, batch: 2, dims: { T: 4 } });
    const y = t.steps[1].memory.find((m) => m.node === heads[0])!;
    expect(y.shape).toEqual([2, 4, 3]);
    expect(y.scale).toEqual({ batch: 2, copies: { drawn: 2, real: 6 } });
    expect(y.bytes).toBe(2 * 4 * 3 * 4 * 3);
    expect(t.steps[1].memory.find((m) => m.node === W)!.bytes).toBe(9 * 4);
    expect(t.steps[1].flops).toBe(2 * (2 * 4 * 3 * 3) * 2 * 3);
  });
});

describe('real shapes under a new T', () => {
  it('re-derives sizes computed from T, including ones the compiler registered from expressions', async () => {
    const g = (await loadExample('deepseek_v3-dsl.yaml')).graph;
    const shapes = g.realShapes({ T: 1024 });
    expect(shapes[g.node('moe_layer.self_attn.eager_attention_forward.P[1]')!.id]).toEqual([1024, 1024]);
    expect(g.realShapes().every((shape, id) => shape.join() === g.nodes[id].realShape.join())).toBe(true);
  });
});

describe('costTimeline on the MoE pages', () => {
  it('counts every parameter of Qwen3-30B-A3B once the 47 placeholder layers are repeated', async () => {
    const page = await loadExample('qwen3_moe-dsl.yaml');
    const steps = createSteps(page.graph, page.steps);
    const t = costTimeline(page.graph, steps.list, { real: true, modules: steps.modules });
    const QWEN3_30B_A3B_PARAMS = 30_532_122_624;
    expect(bytesOf(t.steps[1], 'param')).toBe(QWEN3_30B_A3B_PARAMS * 2);
  });

  it.each(['qwen3_5_moe-dsl.yaml', 'qwen3_8_moe-dsl.yaml', 'qwen3_8_flash_next-dsl.yaml'])('takes the chunk loop count of %s from T and chunk at real size', async (file) => {
    const page = await loadExample(file);
    const node = page.graph.node('linear_layer.linear_attn.chunk_gated_delta_rule.S[1][1]')!;
    const realCount = (dims: Record<string, number>) => {
      const sizes = page.graph.realSizes(dims);
      return node.copies.real.map((d) => (typeof d === 'number' ? d : sizes[d])).reduce((p, v) => p * v, 1);
    };
    const heads = page.graph.realSizes().Hv;
    expect(realCount({})).toBe(heads * 64);
    expect(realCount({ T: 1000 })).toBe(heads * 16);
  });

  it.each(['qwen3_moe-dsl.yaml', 'qwen3_5_moe-dsl.yaml', 'qwen3_8_moe-dsl.yaml', 'qwen3_8_flash_next-dsl.yaml'])('computes the cost of %s at real size within five seconds', async (file) => {
    const page = await loadExample(file);
    const steps = createSteps(page.graph, page.steps);
    const start = performance.now();
    const t = costTimeline(page.graph, steps.list, { real: true, modules: steps.modules, dims: { T: 2048 } });
    expect(performance.now() - start).toBeLessThan(5000);
    expect(t.forward).toBeGreaterThan(0);
    expect(t.uncounted).toEqual([]);
  });
});

describe('costTimeline on the Qwen3 page', () => {
  it('counts every parameter of Qwen3-8B once the placeholder layers are repeated', async () => {
    const page = await loadExample('qwen3/qwen3-dsl.yaml');
    const steps = createSteps(page.graph, page.steps);
    const t = costTimeline(page.graph, steps.list, { real: true, modules: steps.modules });
    const QWEN3_8B_PARAMS = 8_190_735_360;
    expect(bytesOf(t.steps[1], 'param')).toBe(QWEN3_8B_PARAMS * 2);
    const last = t.steps.filter((_, i) => steps.list[i].dir === 'backward' && steps.list[i].kind !== 'overview').at(-1)!;
    expect(bytesOf(last, 'param_grad')).toBe(bytesOf(last, 'param'));
  });

  it('ends backward with one gradient per parameter and nothing else alive', async () => {
    const page = await loadExample('qwen3/qwen3-dsl.yaml');
    const steps = createSteps(page.graph, page.steps);
    const t = costTimeline(page.graph, steps.list, { real: true });
    const last = t.steps.filter((_, i) => steps.list[i].dir === 'backward' && steps.list[i].kind !== 'overview').at(-1)!;
    expect(bytesOf(last, 'param_grad')).toBe(bytesOf(last, 'param'));
    expect(bytesOf(last, 'saved') + bytesOf(last, 'grad') + bytesOf(last, 'temp')).toBe(0);
    expect([steps.list[t.peakStep].dir, steps.list[t.peakStep].title]).toEqual(['backward', 'log_softmax：穿过求和']);
    expect(t.backward / t.forward).toBeCloseTo(2, 1);
  });
});
