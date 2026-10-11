import { describe, expect, it } from 'vitest';
import { type ScaledMmAccum, Tensor, checkOp, graph, opDef } from '../../src/core';
import { TC_ACC_BITS, tcAccumulate } from '../../src/autograd/ops/matmul';
import { FLOAT_FORMATS, roundToFormat } from '../../src/autograd/rounding';
import { loadPage } from '../../src/dsl';

const ACCUMS: ScaledMmAccum[] = ['fp32', 'tc', 'no_promote'];

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(random: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

function quantized(rows: number, K: number, rowsPerScale: number, random: () => number): { q: Tensor; sf: Tensor } {
  const BLOCK = 128;
  const x = Float64Array.from({ length: rows * K }, () => normal(random));
  const kBlocks = Math.ceil(K / BLOCK);
  const sf = new Tensor([Math.ceil(rows / rowsPerScale), kBlocks]);
  const q = new Tensor([rows, K]);
  for (let rb = 0; rb < sf.shape[0]; rb++) {
    for (let j = 0; j < kBlocks; j++) {
      let amax = 0;
      for (let r = rb * rowsPerScale; r < Math.min(rows, (rb + 1) * rowsPerScale); r++) {
        for (let k = j * BLOCK; k < Math.min(K, (j + 1) * BLOCK); k++) amax = Math.max(amax, Math.abs(x[r * K + k]));
      }
      const scale = Math.fround(amax / FLOAT_FORMATS.fp8_e4m3.max);
      sf.data[rb * kBlocks + j] = scale;
      for (let r = rb * rowsPerScale; r < Math.min(rows, (rb + 1) * rowsPerScale); r++) {
        for (let k = j * BLOCK; k < Math.min(K, (j + 1) * BLOCK); k++) q.data[r * K + k] = roundToFormat(x[r * K + k] / scale, FLOAT_FORMATS.fp8_e4m3);
      }
    }
  }
  return { q, sf };
}

function errors(out: Tensor, ref: Tensor): { frobenius: number; maxRelative: number } {
  let diff = 0;
  let norm = 0;
  let maxRelative = 0;
  ref.data.forEach((r, i) => {
    const d = out.data[i] - r;
    diff += d * d;
    norm += r * r;
    maxRelative = Math.max(maxRelative, Math.abs(d) / Math.abs(r));
  });
  return { frobenius: Math.sqrt(diff / norm), maxRelative };
}

describe('scaled_mm', () => {
  it('scales each K-block of A B^T by the row scale of A and the block scale of B', () => {
    const g = graph();
    const a = g.input('a', [1, 4], [[1, 2, 3, 4]]);
    const sfa = g.input('sfa', [1, 2], [[2, 0.5]]);
    const b = g.input('b', [3, 4], [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]]);
    const sfb = g.input('sfb', [2, 2], [[1, 4], [3, 0.25]]);
    for (const accum of ACCUMS) {
      const out = g.scaled_mm(a, sfa, b, sfb, 2, 2, accum);
      expect(out.shape).toEqual([1, 3]);
      expect(Array.from(out.value.data), accum).toEqual([8, 12, 18.875]);
    }
  });

  it('in fp32 mode equals the dequantized matmul', () => {
    const random = seeded(7);
    const [M, N, K, bk, bn] = [3, 5, 7, 3, 2];
    const values = (n: number) => Array.from({ length: n }, () => random() * 4 - 2);
    const g = graph();
    const a = g.input('a', [M, K], values(M * K));
    const b = g.input('b', [N, K], values(N * K));
    const sfa = g.input('sfa', [M, 3], values(M * 3));
    const sfb = g.input('sfb', [3, 3], values(9));
    const deqA = g.input('deqA', [M, K], Array.from({ length: M * K }, (_, i) => a.value.data[i] * sfa.value.get([Math.floor(i / K), Math.floor((i % K) / bk)])));
    const deqB = g.input('deqB', [N, K], Array.from({ length: N * K }, (_, i) => b.value.data[i] * sfb.value.get([Math.floor(Math.floor(i / K) / bn), Math.floor((i % K) / bk)])));
    const out = g.scaled_mm(a, sfa, b, sfb, { block_k: bk, block_n: bn });
    const ref = g.matmul(deqA, deqB.T);
    out.value.data.forEach((v, i) => expect(v).toBeCloseTo(ref.value.data[i], 12));
  });

  it('in tc mode drops product bits below the top TC_ACC_BITS of the largest product in the block', () => {
    expect(TC_ACC_BITS).toBe(14);
    const small = 2 ** -7;
    const g = graph();
    const a = g.input('a', [1, 2], [[1, small]]);
    const b = g.input('b', [1, 2], [[1, small]]);
    const one = g.input('one', [1, 1], [[1]]);
    expect(g.scaled_mm(a, one, b, one, 2, 1, 'fp32').value.data[0]).toBe(1 + 2 ** -14);
    expect(g.scaled_mm(a, one, b, one, 2, 1, 'tc').value.data[0]).toBe(1);
    expect(tcAccumulate(0, [1, 2 ** -13, -(2 ** -13)])).toBe(1);
    expect(tcAccumulate(0, [1, 3 * 2 ** -14])).toBe(1 + 2 ** -13);
    expect(tcAccumulate(0, [-1, -3 * 2 ** -14])).toBe(-1 - 2 ** -13);
  });

  it('promotes each block into an FP32 sum, while no_promote aligns later blocks to the running sum', () => {
    const g = graph();
    const a = g.input('a', [1, 2], [[1, 2 ** -14]]);
    const b = g.input('b', [1, 2], [[1, 1]]);
    const ones = g.input('ones', [1, 2], [[1, 1]]);
    expect(g.scaled_mm(a, ones, b, ones, 1, 1, 'tc').value.data[0]).toBe(1 + 2 ** -14);
    expect(g.scaled_mm(a, ones, b, ones, 1, 1, 'no_promote').value.data[0]).toBe(1);
  });

  it('without promotion is the low-precision baseline over K = 4096', () => {
    const random = seeded(20261011);
    const [M, N, K] = [16, 16, 4096];
    const A = quantized(M, K, 1, random);
    const B = quantized(N, K, 128, random);
    const def = opDef('scaled_mm')!;
    const run = (accum: ScaledMmAccum) => def.forward([A.q, A.sf, B.q, B.sf], { block_k: 128, block_n: 128, accum });
    const ref = run('fp32');
    const tc = errors(run('tc'), ref);
    const flat = errors(run('no_promote'), ref);
    expect(tc.frobenius).toBeGreaterThan(0);
    expect(flat.frobenius).toBeGreaterThan(4 * tc.frobenius);
    expect(flat.maxRelative).toBeGreaterThan(4 * tc.maxRelative);
  });

  it('matches finite differences in fp32 mode for all four operands', () => {
    const def = opDef('scaled_mm')!;
    const random = seeded(3);
    const t = (shape: number[]) => new Tensor(shape, Float64Array.from({ length: shape[0] * shape[1] }, () => random() * 2 - 1));
    expect(() => checkOp(def, [t([2, 5]), t([2, 3]), t([3, 5]), t([2, 3])], { block_k: 2, block_n: 2, accum: 'fp32' })).not.toThrow();
  });

  it('rejects scale shapes that do not match the blocks and unknown modes', () => {
    const g = graph();
    const a = g.input('a', [2, 5], Array.from({ length: 10 }, () => 1));
    const b = g.input('b', [3, 5], Array.from({ length: 15 }, () => 1));
    const sfa = g.input('sfa', [2, 3], Array.from({ length: 6 }, () => 1));
    const sfb = g.input('sfb', [2, 3], Array.from({ length: 6 }, () => 1));
    expect(() => g.scaled_mm(a, sfa, b, sfb, 3, 2)).toThrow('scaled_mm: sfa (2, 3) must be (M, ceil(K / block_k)) = (2, 2) for a (2, 5) and block_k 3');
    expect(() => g.scaled_mm(a, sfa, b, sfb, 2, 3)).toThrow(
      'scaled_mm: sfb (2, 3) must be (ceil(N / block_n), ceil(K / block_k)) = (1, 3) for b (3, 5), block_n 3 and block_k 2',
    );
    expect(() => g.scaled_mm(a, sfa, a, sfb, 2, 2)).toThrow('scaled_mm: sfb (2, 3) must be (ceil(N / block_n), ceil(K / block_k)) = (1, 3)');
    expect(() => g.scaled_mm(a, sfa, sfa, sfb, 2, 2)).toThrow('scaled_mm: a (2, 5) must be (M, K) and b (2, 3) must be (N, K)');
    expect(() => g.scaled_mm(a, sfa, b, sfb, 0, 2)).toThrow('scaled_mm: block_k 0 must be a positive integer');
    expect(() => g.scaled_mm(a, sfa, b, sfb, 2, 2, 'bf16' as ScaledMmAccum)).toThrow("scaled_mm: accum must be one of 'fp32', 'tc', 'no_promote', got \"bf16\"");
  });

  it('is callable from a YAML page with block sizes from dims and the accumulation mode as a keyword', async () => {
    const yaml = `
title: t
dims: { M: 1, N: 2, K: 4, bk: 2, bn: 1 }
inputs:
  a: { value: [[1, 2, 3, 4]] }
  sfa: { value: [[1, 0.5]] }
  b: { value: [[1, 1, 1, 1], [1, 0, 0, 1]] }
  sfb: { value: [[1, 2], [4, 1]] }
body:
  - d: 'scaled_mm(a, sfa, b, sfb, bk, bn, accum="tc")'
`;
    const result = await loadPage('mem:/page.yaml', { read: async () => yaml });
    const d = result.nodes.get('d')!;
    expect(d.attrs).toMatchObject({ block_k: 2, block_n: 1, accum: 'tc' });
    expect(Array.from(d.value.data)).toEqual([3 + 7, 4 * 1 + 0.5 * 4]);
  });

  it('defaults to 128 x 128 blocks and takes block sizes from the dims table', () => {
    const g = graph({ dims: { bk: 2 } });
    const a = g.input('a', [1, 4], [[1, 1, 1, 1]]);
    const one = g.input('one', [1, 1], [[3]]);
    expect(Array.from(g.scaled_mm(a, one, a, one).value.data)).toEqual([36]);
    const two = g.input('two', [1, 2], [[1, 2]]);
    expect(Array.from(g.scaled_mm(a, two, a, two, 'bk', 'bk').value.data)).toEqual([2 + 8]);
  });
});
