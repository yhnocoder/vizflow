import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, loadPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { examplesUrl, readFileUrl } from './load';

const PAGE = 'deepseek_v3-sorted.yaml';
const SHARED = ['E', 'W', 'norm.weight', 'mlp.Wr', 'mlp.shared_experts.Wg', 'mlp.shared_experts.Wu', 'mlp.shared_experts.Wd'];
const STACKED = ['Wg', 'Wu', 'Wd'];
const DIGITS = 12;

function closeTo(actual: ArrayLike<number>, expected: ArrayLike<number>, what: string): void {
  expect(actual.length, what).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) expect(actual[i], `${what}[${i}]`).toBeCloseTo(expected[i], DIGITS);
}

describe('deepseek_v3-sorted.yaml', () => {
  let sorted: CompiledPage;
  let dense: CompiledPage;
  let experts: number;

  beforeAll(async () => {
    sorted = await loadPage(new URL(PAGE, examplesUrl).href, { read: readFileUrl });
    const denseRead = async (url: string) => {
      const text = await readFileUrl(url);
      return url.endsWith(PAGE) ? text.replace('./deepseek_v3/moe-sorted.yaml', './deepseek_v3/moe.yaml') : text;
    };
    dense = await loadPage(new URL(PAGE, examplesUrl).href, { read: denseRead });
    experts = sorted.nodes.get('mlp.Wg')!.shape[0];
    for (const path of SHARED) dense.nodes.get(path)!.set(sorted.nodes.get(path)!.value.toNested());
    for (const name of STACKED) {
      const stacked = sorted.nodes.get(`mlp.${name}`)!.value.toNested() as unknown[];
      stacked.forEach((block, e) => dense.nodes.get(`mlp.expert[${e + 1}].${name}`)!.set(block as number[][]));
    }
  });

  it('routes the toy sequence unevenly across experts', () => {
    const counts = new Array<number>(experts).fill(0);
    for (const e of sorted.nodes.get('mlp.ex')!.value.data) counts[e]++;
    expect(Math.min(...counts)).toBeLessThanOrEqual(1);
    expect(Math.max(...counts)).toBeGreaterThanOrEqual(3);
    const rows = Array.from(sorted.nodes.get('mlp.rows')!.value.data);
    const ex = Array.from(sorted.nodes.get('mlp.ex')!.value.data);
    rows.forEach((t, r) => expect(sorted.nodes.get('mlp.sel')!.value.get([t, ex[r]])).toBe(1));
    expect([...ex].sort((a, b) => a - b)).toEqual(ex);
  });

  it('gives the same MoE output, loss and input gradient as the dense moe.yaml with the same expert weights', () => {
    closeTo(sorted.nodes.get('mlp.out')!.value.data, dense.nodes.get('mlp.out')!.value.data, 'mlp.out');
    expect(sorted.graph.findLoss()!.value.data[0]).toBeCloseTo(dense.graph.findLoss()!.value.data[0], DIGITS);
    closeTo(sorted.nodes.get('n')!.grad!.data, dense.nodes.get('n')!.grad!.data, 'n.grad');
    closeTo(sorted.nodes.get('E')!.grad!.data, dense.nodes.get('E')!.grad!.data, 'E.grad');
  });

  it('gives every expert block the gradient of the matching dense expert, and zero to experts without rows', () => {
    const counts = new Array<number>(experts).fill(0);
    for (const e of sorted.nodes.get('mlp.ex')!.value.data) counts[e]++;
    for (const name of STACKED) {
      const grad = sorted.nodes.get(`mlp.${name}`)!.grad!;
      const block = grad.size / experts;
      for (let e = 0; e < experts; e++) {
        const own = grad.data.subarray(e * block, (e + 1) * block);
        closeTo(own, dense.nodes.get(`mlp.expert[${e + 1}].${name}`)!.grad!.data, `${name}.grad[${e}]`);
        if (counts[e] === 0) expect(own.every((v) => v === 0)).toBe(true);
      }
    }
  });

  it('sends each token the summed gradient of its rows through dispatch, and each row its token gradient through combine', () => {
    const rows = Array.from(sorted.nodes.get('mlp.rows')!.value.data);
    const d = sorted.nodes.get('mlp.R')!.shape[1];
    const rGrad = sorted.nodes.get('mlp.R')!.grad!;
    const ywGrad = sorted.nodes.get('mlp.Yw')!.grad!;
    const routedGrad = sorted.nodes.get('mlp.routed')!.grad!;
    rows.forEach((t, r) => {
      for (let c = 0; c < d; c++) expect(ywGrad.get([r, c])).toBe(routedGrad.get([t, c]));
    });
    const summed = new Float64Array(sorted.nodes.get('n')!.value.size);
    rows.forEach((t, r) => {
      for (let c = 0; c < d; c++) summed[t * d + c] += rGrad.get([r, c]);
    });
    const viaDispatch = sorted.graph.jacobian(sorted.nodes.get('mlp.R')!, sorted.nodes.get('n')!);
    expect(viaDispatch.size).toBe(rows.length * d * summed.length);
    for (let j = 0; j < summed.length; j++) {
      let total = 0;
      for (let i = 0; i < rows.length * d; i++) total += viaDispatch.data[i * summed.length + j] * rGrad.data[i];
      expect(total).toBeCloseTo(summed[j], DIGITS);
    }
    for (const name of ['rows', 'ex']) expect(sorted.nodes.get(`mlp.${name}`)!.grad).toBeNull();
  });

  it('has the dispatch, grouped GEMM and combine steps', () => {
    const titles = createSteps(sorted.graph, sorted.steps).list.map((s) => s.title);
    for (const title of ['排列（dispatch）', 'grouped GEMM：gate_proj 和 up_proj', '加回（combine）', '排列的反向：按 token 加回', '加回的反向：按行取出']) {
      expect(titles).toContain(title);
    }
  });
});
