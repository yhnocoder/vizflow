import { describe, expect, it } from 'vitest';
import { Tensor, forEachIndex, indexOf, numel, offsetOf, stridesOf } from '../../src/tensor';

describe('Tensor', () => {
  it('builds a scalar from a number', () => {
    const t = Tensor.from(3);
    expect(t.shape).toEqual([]);
    expect(t.size).toBe(1);
    expect(t.ndim).toBe(0);
    expect(t.get([])).toBe(3);
    expect(t.toNested()).toBe(3);
  });

  it('infers shape from nested arrays', () => {
    const t = Tensor.from([[1, 2, 3], [4, 5, 6]]);
    expect(t.shape).toEqual([2, 3]);
    expect(Array.from(t.data)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(t.get([1, 2])).toBe(6);
    expect(t.toNested()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });

  it('accepts flat values with an explicit shape', () => {
    const t = Tensor.from([1, 2, 3, 4], [2, 2]);
    expect(t.toNested()).toEqual([[1, 2], [3, 4]]);
  });

  it('rejects ragged values and wrong counts', () => {
    expect(() => Tensor.from([[1, 2], [3]])).toThrow('ragged');
    expect(() => Tensor.from([1, 2, 3], [2, 2])).toThrow('expected 4 values for shape (2, 2), got 3');
  });

  it('requires nested values to match the given shape exactly', () => {
    expect(() => Tensor.from([[1, 2], [3, 4], [5, 6]], [2, 3])).toThrow(
      'expected shape (2, 3), got nested values of shape (3, 2)',
    );
    expect(() => Tensor.from([[1, 2, 3, 4]], [4])).toThrow('expected shape (4,), got nested values of shape (1, 4)');
    expect(Tensor.from([[1, 2, 3], [4, 5, 6]], [2, 3]).toNested()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });

  it('supports one-dimensional tensors', () => {
    const t = Tensor.from([2.0, 1.0, 0.2, -0.6, -1.4]);
    expect(t.shape).toEqual([5]);
    expect(t.get([4])).toBe(-1.4);
  });

  it('sets values by index and clones independently', () => {
    const t = Tensor.zeros([2, 2]);
    t.set([1, 0], 7);
    const c = t.clone();
    c.set([1, 0], 9);
    expect(t.get([1, 0])).toBe(7);
    expect(c.get([1, 0])).toBe(9);
  });

  it('rejects out-of-range indices', () => {
    const t = Tensor.zeros([2, 3]);
    expect(() => t.get([2, 0])).toThrow('out of range');
    expect(() => t.get([0])).toThrow('does not match shape');
  });
});

describe('shape helpers', () => {
  it('computes numel, strides and offsets in row-major order', () => {
    expect(numel([])).toBe(1);
    expect(numel([2, 3, 4])).toBe(24);
    expect(stridesOf([2, 3, 4])).toEqual([12, 4, 1]);
    expect(offsetOf([2, 3, 4], [1, 2, 3])).toBe(23);
    expect(indexOf([2, 3, 4], 23)).toEqual([1, 2, 3]);
    expect(indexOf([], 0)).toEqual([]);
  });

  it('iterates indices in offset order', () => {
    const seen: [number[], number][] = [];
    forEachIndex([2, 2], (index, offset) => seen.push([index, offset]));
    expect(seen).toEqual([
      [[0, 0], 0],
      [[0, 1], 1],
      [[1, 0], 2],
      [[1, 1], 3],
    ]);
  });
});
