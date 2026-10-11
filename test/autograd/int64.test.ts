import { describe, expect, it } from 'vitest';
import { Tensor, formatNumber, graph } from '../../src/core';

const M = [1582356468320383169n, 78689173079975253n, 228252339180906199n];

describe('int64 values beyond 2^53', () => {
  it('keeps products, xor, select, view and remainder exact', () => {
    const g = graph({ check: true });
    const m = g.buffer('m', [3], M, { dtype: 'int64' });
    const ids = g.label('ids', [5, 2, 4]);
    const p0 = g.mul(ids, g.select(m, 0, 0));
    const p1 = g.mul(g.label('prev', [5, 5, 2]), g.select(m, 0, 1));
    const mixed = g.bitwise_xor(p0, p1);
    const sizes = g.buffer('sizes', [1, 2], [13, 17], { dtype: 'int64' });
    const ids2 = g.remainder(g.view(mixed, 3, 1), sizes);
    const wrap = (v: bigint) => BigInt.asIntN(64, v);
    const expected = [5n, 2n, 4n].map((t, i) => wrap(wrap(t * M[0]) ^ wrap([5n, 5n, 2n][i] * M[1])));
    expect([...mixed.value.int64!]).toEqual(expected);
    const floorMod = (x: bigint, y: bigint) => ((x % y) + y) % y;
    expect(ids2.value.toNested()).toEqual(expected.map((v) => [Number(floorMod(v, 13n)), Number(floorMod(v, 17n))]));
    expect(ids2.dtype).toBe('int64');
  });

  it('refuses ops that would read int64 values beyond 2^53 as floats', () => {
    const g = graph();
    const m = g.buffer('m', [3], M, { dtype: 'int64' });
    expect(() => g.add(m, 1)).toThrow("add 'add1': input 'm' holds int64 values beyond 2^53");
  });

  it('compares and combines bools and integers', () => {
    const g = graph();
    const x = g.label('x', [0, 3, 5, 3]);
    const e = g.eq(x, 3);
    expect(e.dtype).toBe('bool');
    expect(e.value.toNested()).toEqual([0, 1, 0, 1]);
    expect(g.bitwise_and(e, g.ge(x, 4)).value.toNested()).toEqual([0, 0, 0, 0]);
    expect(g.cummax(x, 0).value.toNested()).toEqual([0, 3, 5, 5]);
  });

  it('shows floats beyond 2^53 in scientific notation', () => {
    expect(formatNumber(1582356468320383169)).toBe('1.58e18');
    expect(formatNumber(1582356468320383169, { tex: true })).toBe('1.58\\times 10^{18}');
    expect(Tensor.from([2n, 3]).int64).toEqual(new BigInt64Array([2n, 3n]));
  });
});
