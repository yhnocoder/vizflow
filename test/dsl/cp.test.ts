import { beforeAll, describe, expect, it } from 'vitest';
import type { Node } from '../../src/core';
import { type CompiledPage, compilePage, loadPage } from '../../src/dsl';
import type { SceneController } from '../../src/scene3d/controller';
import { parallelNotes, parallelRows } from '../../src/scene3d/parallel';
import { createSteps } from '../../src/steps';
import { examplesUrl, readFileUrl } from './load';

const PAGE = 'qwen3-cp-dsl.yaml';

function strip(text: string): string {
  return text
    .replace(/^mesh: .*\n/m, '')
    .replace(/, mesh: \{ cp: \d+ \}/g, '')
    .replace(/, place: \{ cp: '[^']*' \}/g, '')
    .replace(/(send_recv|all_reduce)\((\w+), cp\)/g, 'identity($2)');
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

function softmaxRow(row: number[]): number[] {
  const max = Math.max(...row);
  const e = row.map((x) => Math.exp(x - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
}

describe('qwen3 context parallel page', () => {
  let cp: CompiledPage;
  let single: CompiledPage;
  const node = (path: string): Node => (cp.nodes.get(path) ?? cp.graph.node(path))!;

  beforeAll(async () => {
    [cp, single] = await Promise.all([load(), load(strip)]);
  });

  it('computes exactly the same tensors and gradients as the page without mesh, place and communication', () => {
    expect(single.placements).toBeUndefined();
    expect(single.graph.nodes.some((n) => n.op === 'send_recv' || n.op === 'all_reduce')).toBe(false);
    let compared = 0;
    for (const [path, n] of single.nodes) {
      const other = cp.nodes.get(path);
      expect(other, path).toBeDefined();
      expect(Array.from(other!.value.data), path).toEqual(Array.from(n.value.data));
      if (n.grad !== null) expect(Array.from(other!.grad!.data), `${path}.grad`).toEqual(Array.from(n.grad.data));
      compared++;
    }
    expect(compared).toBeGreaterThan(100);
  });

  it('gives the same attention output as one causal softmax over all keys', () => {
    const qs = node('layer.self_attn.qs').value;
    const kr = node('layer.self_attn.kr').value;
    const v = node('layer.self_attn.v').value;
    const [T, H, dh] = qs.shape;
    for (let h = 0; h < H; h++) {
      const merged = node(`layer.self_attn.om[${h + 1}]`).value;
      for (let t = 0; t < T; t++) {
        const scores = Array.from({ length: t + 1 }, (_, s) => Array.from({ length: dh }, (_, c) => qs.get([t, h, c]) * kr.get([s, 0, c])).reduce((a, b) => a + b, 0));
        const p = softmaxRow(scores);
        for (let c = 0; c < dh; c++) {
          const expected = p.reduce((acc, w, s) => acc + w * v.get([s, 0, c]), 0);
          expect(Math.abs(merged.get([t, c]) - expected)).toBeLessThan(1e-12);
        }
      }
    }
  });

  it('splits the sequence head-tail: rank 0 holds rows 0 and 3, rank 1 rows 1 and 2', () => {
    const p = cp.placements!;
    const owners = (path: string) => [0, 1, 2, 3].map((t) => p.owner(node(path), [t, 0]));
    expect(owners('ids')).toEqual([0, 1, 1, 0]);
    expect(owners('y')).toEqual([0, 1, 1, 0]);
    expect(owners('layer.self_attn.v')).toEqual([0, 1, 1, 0]);
    expect(owners('layer.self_attn.v1')).toEqual([1, 0, 0, 1]);
    expect(owners('layer.self_attn.om[1]')).toEqual([0, 1, 1, 0]);
    expect(p.of(node('loss_part'))).toEqual({ kind: 'partial' });
    expect(p.of(node('loss'))).toEqual({ kind: 'replicate' });
  });

  it('builds the round masks from the causal mask and the blocks each rank holds in that round', () => {
    const mask = (path: string): number[] => Array.from(node(path).value.data, (x) => (x === 0 ? 1 : 0));
    expect(mask('layer.self_attn.M0')).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1]);
    expect(mask('layer.self_attn.M1')).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]);
    const work = (path: string, rank: number) => [0, 1, 2, 3].filter((t) => [0, 1, 1, 0][t] === rank).reduce((sum, t) => sum + mask(path).slice(t * 4, t * 4 + 4).reduce((a, b) => a + b, 0), 0);
    expect([work('layer.self_attn.M0', 0), work('layer.self_attn.M0', 1), work('layer.self_attn.M1', 0), work('layer.self_attn.M1', 1)]).toEqual([3, 3, 2, 2]);
  });

  it('reports a round that uses the mask of another round', async () => {
    await expect(load((text) => text.replace("op: 'S1[h] + M1'", "op: 'S1[h] + M0'"))).rejects.toThrow(/softmax 之前，不在本卡的格子要是负无穷/);
  });

  it('reports cp_mask arguments that do not split the sequence', async () => {
    await expect(load((text) => text.replace("'cp_mask(T, 2, 1)'", "'cp_mask(T, 3, 1)'"))).rejects.toThrow(/cp_mask: T = 4 must split into 2n = 6 equal chunks/);
  });

  it('checks the head_tail placement against the sequence length', async () => {
    const spec = {
      title: 't',
      mesh: { cp: 2 },
      dims: { T: 2, V: 3 },
      vocab: ['a', 'b', 'c'],
      sequence: { tokens: ['a', 'b', 'c'], input: 'ids', label: 'y', place: { cp: 'shard(0, head_tail)' } },
      inputs: {},
      weights: { E: { shape: ['V', 'V'] } },
      body: [{ z: 'embedding(ids, E)' }, { loss: 'mean(z)' }],
    };
    await expect(compilePage(spec as never)).rejects.toThrow(/cannot be split into 4 equal chunks/);
  });

  it('shows the per-round send_recv traffic on the card and the direction of the gradient', () => {
    const k1 = node('layer.self_attn.k1');
    const forward = parallelRows(fakeScene(cp, 'forward'), k1);
    expect(forward.find((r) => r.label === '放置')?.value).toBe('Shard(0)，这一维分成 4 块，依次在 rank 1、0、0、1');
    expect(forward.find((r) => r.label === '通信')?.value).toContain('rank $r$ 发给 rank $r+1$');
    expect(forward.find((r) => r.label === '每卡每轮发送')?.value).toBe('$\\frac{T}{n}\\times h\\times 2\\,\\mathrm{B}=512\\times 1024\\times 2\\,\\mathrm{B}=1\\,\\mathrm{MiB}$');
    expect(forward.find((r) => r.label === '轮数')?.value).toContain('$n-1=7$');
    const backward = parallelRows(fakeScene(cp, 'backward'), k1);
    expect(backward.find((r) => r.label === '通信')?.value).toContain('rank $r$ 发给 rank $r-1$');
    const decode = parallelRows(fakeScene(cp, 'forward', 'decode'), k1);
    expect(decode.find((r) => r.label === '说明')?.value).toContain('不对 decode 做 CP');
  });

  it('sizes the all_reduce of a weight gradient by the whole weight', () => {
    const rows = parallelRows(fakeScene(cp, 'backward'), node('layer.self_attn.Wq.T'));
    expect(rows.find((r) => r.label === '梯度放置')?.value).toBe('Partial，all_reduce 之后是 Replicate');
    expect(rows.find((r) => r.label === '消息大小')?.value).toBe('整个参数的梯度，$S=4096\\times 4096\\times 2\\,\\mathrm{B}=32\\,\\mathrm{MiB}$');
  });

  it('marks the ring steps with send_recv forward and the gradient going back on the backward step', () => {
    const scene = fakeScene(cp, 'backward');
    const steps = scene.steps!;
    const k1 = node('layer.self_attn.k1');
    const ring = steps.list.filter((info) => info.nodes.includes(k1) && info.kind !== 'overview');
    const forward = ring.find((info) => info.dir === 'forward')!;
    const backward = ring.find((info) => info.dir === 'backward')!;
    expect(parallelNotes(scene, forward).map((n) => n.text)).toContain('send_recv：每张卡把手里的块发给下一张卡，同时从上一张卡收一块');
    const note = parallelNotes(scene, backward).find((n) => n.text.startsWith('send_recv 的反向'))!;
    expect(note.text).toContain('reduce_scatter');
    expect(note.nodes).toEqual([node('layer.self_attn.kr'), node('layer.self_attn.v')]);
  });
});
