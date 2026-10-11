import { beforeAll, describe, expect, it } from 'vitest';
import { FLOAT_FORMATS, roundToFormat } from '../../src/autograd/rounding';
import type { CompiledPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { loadExample } from './load';

const [M, N, K, BK, BN] = [2, 4, 8, 4, 2];
const KB = K / BK;
const E4M3 = FLOAT_FORMATS.fp8_e4m3;

type Matrix = number[][];

function quantize(x: Matrix, rowsPerScale: number): { q: Matrix; sf: Matrix } {
  const sf: Matrix = [];
  const q: Matrix = x.map((row) => row.map(() => 0));
  for (let rb = 0; rb < x.length / rowsPerScale; rb++) {
    sf.push([]);
    for (let j = 0; j < KB; j++) {
      const rows = x.slice(rb * rowsPerScale, (rb + 1) * rowsPerScale);
      const amax = Math.max(...rows.flatMap((row) => row.slice(j * BK, (j + 1) * BK).map(Math.abs)));
      const scale = Math.fround(amax / 448);
      const inverse = Math.fround(1 / scale);
      sf[rb].push(scale);
      for (let r = rb * rowsPerScale; r < (rb + 1) * rowsPerScale; r++) {
        for (let k = j * BK; k < (j + 1) * BK; k++) q[r][k] = roundToFormat(Math.fround(x[r][k] * inverse), E4M3);
      }
    }
  }
  return { q, sf };
}

function products(a: Matrix, b: Matrix, m: number, n: number, j: number): number[] {
  return Array.from({ length: BK }, (_, i) => a[m][j * BK + i] * b[n][j * BK + i]);
}

function truncatedSum(terms: number[]): number {
  const top = Math.floor(Math.log2(Math.max(...terms.map(Math.abs))));
  const unit = 2 ** (top - 13);
  return Math.fround(terms.reduce((sum, t) => sum + Math.sign(t) * Math.floor(Math.abs(t) / unit) * unit, 0));
}

function rows(page: CompiledPage, name: string): Matrix {
  return page.nodes.get(name)!.value.toNested() as Matrix;
}

describe('deepgemm-fp8.yaml', () => {
  let page: CompiledPage;
  let A: { q: Matrix; sf: Matrix };
  let B: { q: Matrix; sf: Matrix };

  beforeAll(async () => {
    page = await loadExample('deepgemm-fp8.yaml');
    A = quantize(rows(page, 'A'), 1);
    B = quantize(rows(page, 'B'), BN);
  });

  it('quantizes A per 1 x block_k tile and B per block_n x block_k block with fp32 amax / 448 scales', () => {
    expect(rows(page, 'sfa')).toEqual(A.sf);
    expect(rows(page, 'sfb')).toEqual(B.sf);
    expect(rows(page, 'Aq')).toEqual(A.q);
    expect(rows(page, 'Bq')).toEqual(B.q);
    expect(page.nodes.get('Aq')!.dtype).toBe('fp8_e4m3');
    expect(page.nodes.get('Bq')!.dtype).toBe('fp8_e4m3');
  });

  it('stores values that fp8 e4m3 represents exactly, with every block reaching 448', () => {
    for (const name of ['Aq', 'Bq']) {
      const values = rows(page, name);
      for (const v of values.flat()) {
        expect(roundToFormat(v, E4M3)).toBe(v);
        expect(Math.abs(v)).toBeLessThanOrEqual(448);
      }
      const rowsPerScale = name === 'Aq' ? 1 : BN;
      for (let rb = 0; rb < values.length / rowsPerScale; rb++) {
        for (let j = 0; j < KB; j++) {
          const block = values.slice(rb * rowsPerScale, (rb + 1) * rowsPerScale).flatMap((row) => row.slice(j * BK, (j + 1) * BK));
          expect(Math.max(...block.map(Math.abs)), `${name} block ${rb}, ${j}`).toBe(448);
        }
      }
    }
  });

  it('matches the exact block sum of the quantized operands', () => {
    const exact = Array.from({ length: M }, (_, m) =>
      Array.from({ length: N }, (_, n) =>
        Array.from({ length: KB }, (_, j) => A.sf[m][j] * B.sf[Math.floor(n / BN)][j] * products(A.q, B.q, m, n, j).reduce((s, p) => s + p, 0)).reduce((s, v) => s + v, 0),
      ),
    );
    rows(page, 'Dx').forEach((row, m) => row.forEach((v, n) => expect(v).toBeCloseTo(exact[m][n], 12)));
    rows(page, 'D').forEach((row, m) => row.forEach((v, n) => expect(Math.abs(v - exact[m][n])).toBeLessThan(2 ** -8 * Math.abs(exact[m][n]))));
  });

  it('accumulates each K block with truncation below the top 14 bits and promotes it into an fp32 sum', () => {
    const acc = (j: number) => Array.from({ length: M }, (_, m) => Array.from({ length: N }, (_, n) => truncatedSum(products(A.q, B.q, m, n, j))));
    const [acc1, acc2] = [acc(0), acc(1)];
    expect(rows(page, 'acc1')).toEqual(acc1);
    expect(rows(page, 'acc2')).toEqual(acc2);
    expect(acc1[0][0]).toBe(-83952);
    const promoted = acc1.map((row, m) =>
      row.map((_, n) => {
        const nb = Math.floor(n / BN);
        const first = Math.fround(Math.fround(A.sf[m][0] * B.sf[nb][0]) * acc1[m][n]);
        return Math.fround(first + Math.fround(A.sf[m][1] * B.sf[nb][1]) * acc2[m][n]);
      }),
    );
    expect(rows(page, 'Dacc')).toEqual(promoted);
  });

  it('has no loss and so no backward steps', () => {
    expect(page.graph.findLoss()).toBeUndefined();
    expect(createSteps(page.graph, page.steps).list.every((step) => step.dir === 'forward')).toBe(true);
  });
});
