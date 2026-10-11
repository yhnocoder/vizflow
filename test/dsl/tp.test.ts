import { beforeAll, describe, expect, it } from 'vitest';
import type { Node } from '../../src/core';
import { type CompiledPage, compilePage, loadPage } from '../../src/dsl';
import { parallelMode, parallelNotes, parallelRows } from '../../src/scene3d/parallel';
import type { SceneController } from '../../src/scene3d/controller';
import { createSteps } from '../../src/steps';
import { examplesUrl, readFileUrl } from './load';

const PAGE = 'qwen3-tp-dsl.yaml';

function strip(text: string): string {
  return text
    .replace(/^mesh: .*\n/m, '')
    .replace(/, mesh: \{ tp: \d+ \}/g, '')
    .replace(/, place: \{ tp: \w+(\(\d+\))? \}/g, '')
    .replace(/all_reduce\((\w+), tp\)/g, 'identity($1)');
}

function load(transform: (text: string) => string = (text) => text): Promise<CompiledPage> {
  return loadPage(new URL(PAGE, examplesUrl).href, { read: async (url) => transform(await readFileUrl(url)) });
}

function fakeScene(page: CompiledPage, dir: 'forward' | 'backward', mode: 'train' | 'prefill' | 'decode' = 'train'): SceneController {
  const infer = mode !== 'train';
  const steps = createSteps(page.graph, infer ? page.modes!.infer : page.modes!.train);
  if (infer) steps.go(steps.list.findIndex((info) => info.segment?.startsWith(mode) === true));
  return { options: { parallel: { placements: page.placements!, infer } }, state: { dir }, g: page.graph, steps } as unknown as SceneController;
}

describe('qwen3 tensor parallel page', () => {
  let tp: CompiledPage;
  let single: CompiledPage;

  beforeAll(async () => {
    [tp, single] = await Promise.all([load(), load(strip)]);
  });

  it('computes exactly the same tensors and gradients as the page without mesh, place and all_reduce', () => {
    expect(single.placements).toBeUndefined();
    expect(tp.placements).toBeDefined();
    let compared = 0;
    for (const [path, node] of single.nodes) {
      const other = tp.nodes.get(path);
      expect(other, path).toBeDefined();
      expect(Array.from(other!.value.data), path).toEqual(Array.from(node.value.data));
      if (node.grad !== null) expect(Array.from(other!.grad!.data), `${path}.grad`).toEqual(Array.from(node.grad.data));
      compared++;
    }
    expect(compared).toBeGreaterThan(100);
  });

  it('places the projections by column and by row, the heads on their own device, and both all_reduce outputs as Replicate', () => {
    const p = tp.placements!;
    const at = (path: string) => p.of((tp.nodes.get(path) ?? tp.graph.node(path))!);
    expect(at('layer.self_attn.Wq.T')).toEqual({ kind: 'shard', dim: 1 });
    expect(at('layer.self_attn.q')).toEqual({ kind: 'shard', dim: 1 });
    expect(at('layer.self_attn.eager_attention_forward.P[1]')).toEqual({ kind: 'local', rank: 0 });
    expect(at('layer.self_attn.eager_attention_forward.P[2]')).toEqual({ kind: 'local', rank: 1 });
    expect(at('layer.self_attn.o')).toEqual({ kind: 'shard', dim: 1 });
    expect(at('layer.self_attn.att_partial')).toEqual({ kind: 'partial' });
    expect(at('layer.self_attn.att')).toEqual({ kind: 'replicate' });
    expect(at('layer.mlp.down_partial')).toEqual({ kind: 'partial' });
    expect(at('layer.mlp.down')).toEqual({ kind: 'replicate' });
    expect(at('layer.a')).toEqual({ kind: 'replicate' });
    const reduced = [...tp.nodes].filter(([, node]) => p.grad(node).reduce).map(([path]) => path);
    expect(reduced).toEqual(expect.arrayContaining(['layer.n1', 'layer.n', 'layer.self_attn.q_norm.weight', 'layer.self_attn.k_norm.weight']));
    expect(reduced).not.toContain('layer.self_attn.att_partial');
  });

  it('splits every Partial tensor and every Partial gradient into per-device values that add up to the global value', () => {
    const p = tp.placements!;
    const nodes = new Set<Node>(tp.nodes.values());
    let partials = 0;
    let grads = 0;
    for (const node of nodes) {
      const ranks = p.ranks(node);
      if (ranks !== undefined) {
        partials++;
        ranks.reduce((acc, t) => acc.map((v, i) => v + t.data[i]), new Array(node.value.size).fill(0)).forEach((v, i) => expect(v).toBeCloseTo(node.value.data[i], 12));
      }
      const gradRanks = p.gradRanks(node);
      if (gradRanks !== undefined) {
        grads++;
        gradRanks.reduce((acc, t) => acc.map((v, i) => v + t.data[i]), new Array(node.value.size).fill(0)).forEach((v, i) => expect(v).toBeCloseTo(node.grad!.data[i], 12));
      }
    }
    expect(partials).toBe(2);
    expect(grads).toBe(4);
  });

  it('reports a missing all_reduce at compile time', async () => {
    await expect(load((text) => text.replace('all_reduce(att_partial, tp)', 'identity(att_partial)'))).rejects.toThrow(/节点 'layer\.a'（add）：Partial 与非 Partial 相加不合法/);
  });

  it('reports a Partial tensor that reaches RMSNorm', async () => {
    const spec = {
      title: 't',
      import: { RMSNorm: 'vizflow/nn/rmsnorm' },
      mesh: { tp: 2 },
      dims: { T: 2, d: 4, f: 6 },
      inputs: { x: { value: [[0.5, -1, 0.3, 0.8], [1.2, 0.1, -0.4, 0.6]] } },
      weights: { Wu: { shape: ['f', 'd'], place: { tp: 'shard(0)' } }, Wd: { shape: ['d', 'f'], place: { tp: 'shard(1)' } } },
      body: [{ y: 'x @ Wu.T @ Wd.T' }, { norm: { use: 'RMSNorm', inputs: { x: 'y' }, outputs: { out: 'h' } } }, { loss: 'mean(h ** 2)' }],
    };
    await expect(compilePage(spec as never)).rejects.toThrow(/节点 'norm\.sq'（pow）：Partial 不能直接进入非线性运算，要先 all_reduce/);
  });

  it('reports a head split that does not fall on a head boundary', async () => {
    await expect(load((text) => text.replace('dims: { T: 3, d: 4, f: 6, V: 5, H: 2, Hkv: 2, dh: 2 }', 'dims: { T: 3, d: 4, f: 6, V: 5, H: 3, Hkv: 3, dh: 2 }'))).rejects.toThrow(/切分的边界没有落在/);
  });

  it('checks the place field against the mesh', async () => {
    await expect(load((text) => text.replace('place: { tp: shard(0) }', 'place: { pp: shard(0) }'))).rejects.toThrow(/'pp' is not an axis of the mesh/);
    await expect(load((text) => text.replace(/^mesh: .*\n/m, '').replace(', mesh: { tp: 8 }', ''))).rejects.toThrow(/place needs a mesh on the page/);
    await expect(load((text) => text.replace('mesh: { tp: 8 }', 'mesh: { tp: { size: 16, nodes: 3 } }'))).rejects.toThrow(/nodes must be a positive integer that divides 16/);
  });

  it('reads the nodes of a mesh axis and shows the links on the all_reduce card', async () => {
    const page = await load((text) => text.replace('mesh: { tp: 8 }', 'mesh: { tp: { size: 16, nodes: 2 } }'));
    expect(page.placements!.mesh).toEqual({ axis: 'tp', size: 2, nodes: 1, real: { size: 16, nodes: 2 } });
    const rows = parallelRows(fakeScene(page, 'forward'), page.nodes.get('layer.self_attn.att')!);
    expect(rows.find((r) => r.label === '链路')?.value).toBe('2 个节点，每个节点 8 张卡；节点内 NVLink，节点之间 RDMA');
  });

  it('shows the all_reduce traffic on the cards with the token count of the mode', () => {
    const att = tp.nodes.get('layer.self_attn.att')!;
    const train = parallelRows(fakeScene(tp, 'forward'), att);
    expect(train.find((r) => r.label === '放置')?.value).toBe('Replicate');
    expect(train.find((r) => r.label === '消息大小')?.value).toContain('=4096\\times 4096\\times 2\\,\\mathrm{B}=32\\,\\mathrm{MiB}');
    expect(train.find((r) => r.label === '每卡发送')?.value).toContain('=56\\,\\mathrm{MiB}');
    expect(train.find((r) => r.label === '链路')?.value).toBe('NVLink，8 张卡在同一节点');
    const prefill = parallelRows(fakeScene(tp, 'forward', 'prefill'), att);
    expect(prefill.find((r) => r.label === '消息大小')?.value).toContain('=32\\,\\mathrm{MiB}');
    const decode = parallelRows(fakeScene(tp, 'forward', 'decode'), att);
    expect(decode.find((r) => r.label === '消息大小')?.value).toContain('=64\\times 4096\\times 2\\,\\mathrm{B}=512\\,\\mathrm{KiB}');
    const n1 = parallelRows(fakeScene(tp, 'backward'), tp.nodes.get('layer.n1')!);
    expect(n1.find((r) => r.label === '通信')?.value).toBe('反向 all_reduce');
    const partial = parallelRows(fakeScene(tp, 'backward'), tp.nodes.get('layer.self_attn.att')!);
    expect(partial.find((r) => r.label === '通信')?.value).toContain('反向不通信');
  });

  it('marks the backward all_reduce on the step that finishes the Partial gradient', () => {
    const scene = fakeScene(tp, 'backward');
    const steps = scene.steps!;
    const n1 = tp.nodes.get('layer.n1')!;
    const marked = steps.list.filter((info) => parallelNotes(scene, info).some((note) => note.nodes.includes(n1)));
    expect(marked.map((info) => info.id)).toEqual(['b:6']);
    const g = steps.list.find((info) => info.dir === 'backward' && info.nodes.includes(tp.nodes.get('layer.self_attn.att_partial')!))!;
    expect(parallelNotes(scene, g).map((note) => note.text)).toContain('all_reduce 的反向不通信');
  });

  it('uses the training and inference modes of the page and tells prefill from decode by the step', () => {
    expect(createSteps(tp.graph, tp.modes!.infer).list.some((info) => info.dir === 'backward')).toBe(false);
    expect(createSteps(tp.graph, tp.modes!.train).list.some((info) => info.dir === 'backward')).toBe(true);
    expect(parallelMode(fakeScene(tp, 'forward'))).toBe('train');
    expect(parallelMode(fakeScene(tp, 'forward', 'prefill'))).toBe('prefill');
    expect(parallelMode(fakeScene(tp, 'forward', 'decode'))).toBe('decode');
  });
});
