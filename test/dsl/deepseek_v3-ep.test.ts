import { beforeAll, describe, expect, it } from 'vitest';
import { commTransfers, rankDevice } from '../../src/core';
import { combineTokenBytes, dispatchTokenBytes } from '../../src/sim/deepep';
import { type CompiledPage, loadPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { examplesUrl, readFileUrl } from './load';

const PAGE = 'deepseek_v3-ep.yaml';
const SHARED = ['mlp.Wr', 'mlp.shared_experts.Wg', 'mlp.shared_experts.Wu', 'mlp.shared_experts.Wd'];
const STACKED = ['Wg', 'Wu', 'Wd'];
const DIGITS = 12;

function closeTo(actual: ArrayLike<number>, expected: ArrayLike<number>, what: string): void {
  expect(actual.length, what).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) expect(actual[i], `${what}[${i}]`).toBeCloseTo(expected[i], DIGITS);
}

describe('deepseek_v3-ep.yaml', () => {
  let ep: CompiledPage;
  let dense: CompiledPage;

  beforeAll(async () => {
    ep = await loadPage(new URL(PAGE, examplesUrl).href, { read: readFileUrl });
    const denseRead = async (url: string) => {
      const text = await readFileUrl(url);
      return url.endsWith(PAGE) ? text.replace('./deepseek_v3/moe-ep.yaml', './deepseek_v3/moe.yaml') : text;
    };
    dense = await loadPage(new URL(PAGE, examplesUrl).href, { read: denseRead });
    for (const path of SHARED) dense.nodes.get(path)!.set(ep.nodes.get(path)!.value.toNested());
    for (const name of STACKED) {
      const stacked = ep.nodes.get(`mlp.${name}`)!.value.toNested() as unknown[];
      stacked.forEach((block, e) => dense.nodes.get(`mlp.expert[${e + 1}].${name}`)!.set(block as number[][]));
    }
  });

  it('gives the same MoE output as the dense moe.yaml with the same expert weights', () => {
    closeTo(ep.nodes.get('moe')!.value.data, dense.nodes.get('moe')!.value.data, 'moe');
    closeTo(ep.nodes.get('mlp.routed')!.value.data, dense.nodes.get('mlp.routed')!.value.data, 'routed');
  });

  it('places tokens by position, routed rows on the rank of their expert, and the output back by token', () => {
    const p = ep.placements!;
    expect(p.mesh).toMatchObject({ axis: 'ep', size: 4, nodes: 2, real: { size: 32, nodes: 4 } });
    for (const name of ['x', 'mlp.sel', 'mlp.rw', 'mlp.routed', 'moe']) expect(p.of(ep.nodes.get(name)!), name).toEqual({ kind: 'shard', dim: 0 });
    for (const name of ['mlp.R', 'mlp.P', 'mlp.ex', 'mlp.pr', 'mlp.gate_e', 'mlp.Yw']) expect(p.of(ep.nodes.get(name)!).kind, name).toBe('routed');
    for (const name of ['mlp.Wg', 'mlp.Wu', 'mlp.Wd']) expect(p.of(ep.nodes.get(name)!), name).toEqual({ kind: 'shard', dim: 0 });
    const ex = Array.from(ep.nodes.get('mlp.ex')!.value.data);
    ex.forEach((e, r) => expect(p.owner(ep.nodes.get('mlp.R')!, [r, 0])).toBe(Math.floor(e / 2)));
  });

  it('gives every row the weight of its own expert, which combine then only sums', () => {
    const sel = ep.nodes.get('mlp.sel')!.value;
    const rw = ep.nodes.get('mlp.rw')!.value;
    const R = ep.nodes.get('mlp.R')!.value;
    const x = ep.nodes.get('x')!.value;
    const ex = Array.from(ep.nodes.get('mlp.ex')!.value.data);
    const pr = Array.from(ep.nodes.get('mlp.pr')!.value.data);
    const tokens = ex.map((e, r) => {
      const t = Array.from({ length: sel.shape[0] }, (_, i) => i).find((i) => sel.get([i, e]) === 1 && Array.from({ length: x.shape[1] }, (_, c) => x.get([i, c])).every((v, c) => v === R.get([r, c])));
      expect(t, `row ${r}`).toBeDefined();
      return t!;
    });
    pr.forEach((v, r) => expect(v).toBe(rw.get([tokens[r], ex[r]])));
    expect(ep.nodes.get('mlp.routed')!.inputs[2]).toBe(ep.nodes.get('mlp.sel'));
  });

  it('sends cross-node tokens over RDMA to the same-index GPU and forwards them over NVLink', () => {
    const p = ep.placements!;
    const transfers = commTransfers(p, ep.nodes.get('mlp.R')!).filter((t) => t.link !== 'local');
    expect(transfers.some((t) => t.link === 'rdma') && transfers.some((t) => t.link === 'nvlink')).toBe(true);
    for (const t of transfers) {
      if (t.link === 'rdma') expect([t.hop, rankDevice(p.mesh, t.src).gpu]).toEqual([0, rankDevice(p.mesh, t.dst).gpu]);
      else expect(t.hop).toBe(1);
    }
    const back = commTransfers(p, ep.nodes.get('mlp.routed')!).filter((t) => t.link === 'rdma');
    expect(back.length).toBeGreaterThan(0);
    expect(back.every((t) => t.hop === 1)).toBe(true);
  });

  it('sizes every message by the hidden size of the real model', () => {
    const p = ep.placements!;
    const K = 2;
    for (const t of commTransfers(p, ep.nodes.get('mlp.R')!)) expect([t.bytes > 0, t.bytes % dispatchTokenBytes(7168, 'bf16', K)]).toEqual([true, 0]);
    for (const t of commTransfers(p, ep.nodes.get('mlp.routed')!)) expect([t.bytes > 0, t.bytes % combineTokenBytes(7168, K)]).toEqual([true, 0]);
  });

  it('has the dispatch, grouped GEMM and combine steps and no backward steps without a loss', () => {
    const list = createSteps(ep.graph, ep.steps).list;
    expect(list.filter((s) => s.dir === 'backward')).toEqual([]);
    const titles = list.map((s) => s.title);
    for (const title of ['dispatch：token 发到专家所在的卡', 'grouped GEMM：gate_proj 和 up_proj', 'combine：各行加回自己的 token']) expect(titles).toContain(title);
  });
});
