import { describe, expect, it } from 'vitest';
import { type PageFile, compilePage } from '../../src/dsl';
import { parseYaml } from '../../src/dsl/source';

const q = [[0.3, -0.2], [0.5, 0.1], [-0.4, 0.7]];
const k = [[0.2, 0.4], [-0.1, 0.3], [0.6, -0.5], [0.25, 0.15]];
const v = [[1, 2], [3, -1], [0.5, 0.5], [-2, 1]];

function page(buffers: string, extra = '') {
  return compilePage(
    parseYaml(
      `title: t\ndims: { Tq: 3, Tk: 4, d: 2 }\ninputs:\n  q: { value: ${JSON.stringify(q)} }\n  k: { value: ${JSON.stringify(k)} }\n  v: { value: ${JSON.stringify(v)} }\nbuffers:\n${buffers}\nbody:\n  - S: q @ k.T\n  - P: softmax(S + M, -1)\n  - o: P @ v\n${extra}`,
      'page.yaml',
    ) as PageFile,
  );
}

function causalAttention(qs: number[][], ks: number[][], vs: number[][], offset: number): number[][] {
  return qs.map((qi, i) => {
    const visible = ks.slice(0, offset + i + 1);
    const scores = visible.map((kj) => qi[0] * kj[0] + qi[1] * kj[1]);
    const top = Math.max(...scores);
    const e = scores.map((s) => Math.exp(s - top));
    const z = e.reduce((a, b) => a + b, 0);
    return [0, 1].map((c) => e.reduce((acc, w, j) => acc + (w / z) * vs[j][c], 0));
  });
}

describe('packed sequence buffers', () => {
  it('builds a block-diagonal causal mask over cached and new rows, and positions that restart per sequence', async () => {
    const compiled = await page(`  M: { value: 'packed_mask([2, 1], [1, 0])', shape: [Tq, Tk] }\n  pos: { value: 'packed_positions([2, 1], [1, 0])', shape: [Tq] }\n  seg: { value: 'packed_segments([2, 1])', shape: [Tq] }`);
    const M = compiled.nodes.get('M')!.value.toNested();
    const I = -Infinity;
    expect(M).toEqual([[0, 0, I, I], [0, 0, 0, I], [I, I, I, 0]]);
    expect([...compiled.nodes.get('pos')!.value.data]).toEqual([1, 2, 0]);
    expect([...compiled.nodes.get('seg')!.value.data]).toEqual([0, 0, 1]);
  });

  it('gives each packed sequence the same attention output as running it alone', async () => {
    const compiled = await page(`  M: { value: 'packed_mask([2, 1], [1, 0])', shape: [Tq, Tk] }`);
    const o = compiled.nodes.get('o')!.value.toNested() as number[][];
    const first = causalAttention(q.slice(0, 2), k.slice(0, 3), v.slice(0, 3), 1);
    const second = causalAttention(q.slice(2), k.slice(3), v.slice(3), 0);
    [...first, ...second].forEach((row, i) => row.forEach((x, c) => expect(o[i][c]).toBeCloseTo(x, 12)));
  });

  it('limits each query to the last window keys of its own sequence', async () => {
    const compiled = await page(`  M: { value: 'packed_mask([3], [1], 2)', shape: [Tq, Tk] }`);
    const I = -Infinity;
    expect(compiled.nodes.get('M')!.value.toNested()).toEqual([[0, 0, I, I], [I, 0, 0, I], [I, I, 0, 0]]);
  });

  it('reports bad arguments', async () => {
    await expect(page(`  M: { value: 'packed_mask(2)', shape: [Tq, Tk] }`)).rejects.toThrow(/packed_mask takes a list of new-token counts/);
    await expect(page(`  M: { value: 'packed_mask([2, 1], [1])', shape: [Tq, Tk] }`)).rejects.toThrow(/2 new-token counts but 1 cached lengths/);
    await expect(page(`  M: { value: 'packed_mask([3], [1], 0)', shape: [Tq, Tk] }`)).rejects.toThrow(/window size must be a positive integer/);
    await expect(page(`  M: { value: 'packed_mask([2, 2])', shape: [Tq, Tk] }`)).rejects.toThrow(/do not fit the shape/);
  });
});
