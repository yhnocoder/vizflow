import { describe, expect, it } from 'vitest';
import { loadPage } from '../../src/dsl';
import { examplesUrl, loadExample, readFileUrl } from './load';

const single = (tokens: string) =>
  loadPage(new URL('qwen3-packed-dsl.yaml', examplesUrl).href, {
    read: (u) =>
      readFileUrl(u).then((s) =>
        s
          .replace(/^dims: \{ T: 5,/m, `dims: { T: ${tokens.split(', ').length - 1},`)
          .replace(/^sequence: .*$/m, `sequence: { tokens: [${tokens}], input: ids, label: y }`)
          .replace(/^ {2}layer\.self_attn\.(M|apply_rotary_pos_emb\.(cos|sin)): .*\n/gm, '')
          .replace("op: 'sum(lt) / 3'", 'op: mean(lt)'),
      ),
  });

describe('qwen3-packed-dsl', () => {
  it('gives each packed sequence the same log probabilities as running it alone, and ignores the last position of each', async () => {
    const packed = await loadExample('qwen3-packed-dsl.yaml');
    const first = await single('the, cat, sat, on');
    const second = await single('on, mat, the, cat');
    const rows = (page: typeof packed, from: number, to: number) => (page.nodes.get('logp')!.value.toNested() as number[][]).slice(from, to);
    const close = (a: number[][], b: number[][]) => a.forEach((row, i) => row.forEach((x, j) => expect(x).toBeCloseTo(b[i][j], 10)));
    close(rows(packed, 0, 3), rows(first, 0, 3));
    close(rows(packed, 3, 5), rows(second, 0, 2));
    const lt = Array.from(packed.nodes.get('lt')!.value.data);
    expect(lt[2]).toBe(0);
    expect(lt[4]).toBe(0);
    expect(Array.from(packed.nodes.get('y')!.value.data)).toEqual([4, 3, -100, 1, -100]);
    expect(packed.nodes.get('loss')!.value.data[0]).toBeCloseTo((lt[0] + lt[1] + lt[3]) / 3, 12);
  });
});
