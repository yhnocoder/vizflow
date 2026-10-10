import { describe, expect, it } from 'vitest';
import {
  Tensor,
  broadcastIndex,
  broadcastShapes,
  broadcastTo,
  normDim,
  normDims,
  reduceMap,
  reducedShape,
  unbroadcast,
} from '../../src/tensor';

describe('broadcastShapes', () => {
  it.each([
    [[5, 3, 4, 1], [3, 1, 1], [5, 3, 4, 1]],
    [[1, 5], [5], [1, 5]],
    [[3, 1], [1, 5], [3, 5]],
    [[], [2, 3], [2, 3]],
    [[2, 3], [], [2, 3]],
    [[1], [1], [1]],
    [[4, 1], [4, 5], [4, 5]],
  ])('%j with %j gives %j', (a, b, expected) => {
    expect(broadcastShapes(a, b)).toEqual(expected);
  });

  it.each([
    [[3], [4]],
    [[2, 1], [8, 4, 3]],
    [[5, 3], [3, 5]],
  ])('%j with %j fails', (a, b) => {
    expect(() => broadcastShapes(a, b)).toThrow(/cannot broadcast \(.*\) with \(.*\)/);
  });
});

describe('broadcastIndex', () => {
  it('maps output indices to input indices', () => {
    expect(broadcastIndex([2, 3], [1, 4])).toEqual([0, 3]);
    expect(broadcastIndex([2, 3], [4])).toEqual([3]);
    expect(broadcastIndex([2, 3], [])).toEqual([]);
    expect(broadcastIndex([1, 2, 3], [2, 1])).toEqual([2, 0]);
  });

  it('materializes a broadcast tensor', () => {
    const t = broadcastTo(Tensor.from([[1], [2]]), [2, 3]);
    expect(t.toNested()).toEqual([[1, 1, 1], [2, 2, 2]]);
  });
});

describe('unbroadcast', () => {
  it('sums leading and size-1 dimensions back', () => {
    const grad = Tensor.from([[1, 2, 3], [4, 5, 6]]);
    expect(unbroadcast(grad, [2, 3]).toNested()).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(unbroadcast(grad, [3]).toNested()).toEqual([5, 7, 9]);
    expect(unbroadcast(grad, [1, 3]).toNested()).toEqual([[5, 7, 9]]);
    expect(unbroadcast(grad, [2, 1]).toNested()).toEqual([[6], [15]]);
    expect(unbroadcast(grad, []).toNested()).toBe(21);
  });
});

describe('normDim and reducedShape', () => {
  it('normalizes negative dims and rejects out-of-range ones', () => {
    expect(normDim(-1, 3)).toBe(2);
    expect(normDim(0, 3)).toBe(0);
    expect(() => normDim(3, 3)).toThrow('out of range');
    expect(() => normDim(-4, 3)).toThrow('out of range');
    expect(normDims(null, 2)).toEqual([0, 1]);
    expect(normDims([-1, 0], 3)).toEqual([0, 2]);
    expect(() => normDims([0, -3], 3)).toThrow('repeated');
  });

  it('matches PyTorch keepdim behaviour', () => {
    expect(reducedShape([2, 3, 4], [-1], false)).toEqual([2, 3]);
    expect(reducedShape([2, 3, 4], [-1], true)).toEqual([2, 3, 1]);
    expect(reducedShape([2, 3, 4], [0, 2], false)).toEqual([3]);
    expect(reducedShape([2, 3, 4], [0, 2], true)).toEqual([1, 3, 1]);
    expect(reducedShape([5], [0], false)).toEqual([]);
    expect(reducedShape([5], [0], true)).toEqual([1]);
  });

  it('maps every input offset to its reduced output offset', () => {
    const { shape, map } = reduceMap([2, 3], [1], false);
    expect(shape).toEqual([2]);
    expect(Array.from(map)).toEqual([0, 0, 0, 1, 1, 1]);
    const kept = reduceMap([2, 3], [0], true);
    expect(kept.shape).toEqual([1, 3]);
    expect(Array.from(kept.map)).toEqual([0, 1, 2, 0, 1, 2]);
  });
});
