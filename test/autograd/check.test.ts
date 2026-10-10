import { describe, expect, it } from 'vitest';
import { Tensor, checkOp, opDef } from '../../src/core';

function filled(shape: number[], seed: number, offset = 0): Tensor {
  const data = Float64Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => {
    const v = Math.sin(seed * 7.3 + i * 1.7) * 1.5;
    return v + offset;
  });
  return new Tensor(shape, data);
}

function positive(shape: number[], seed: number): Tensor {
  const t = filled(shape, seed);
  return new Tensor(shape, t.data.map((v) => Math.abs(v) + 0.5));
}

function check(name: string, inputs: Tensor[], attrs: object = {}): void {
  const def = opDef(name);
  expect(def, name).toBeDefined();
  checkOp(def!, inputs, attrs);
}

const binaryShapes: [number[], number[]][] = [
  [[2, 3], [2, 3]],
  [[2, 3], [3]],
  [[2, 3], [1, 3]],
  [[2, 1], [1, 3]],
  [[], [4]],
  [[1, 5], []],
  [[3, 1, 2], [4, 1]],
];

describe('checkOp on elementwise ops', () => {
  it.each(binaryShapes)('add, sub, mul and div with %j and %j', (a, b) => {
    for (const name of ['add', 'sub', 'mul']) check(name, [filled(a, 1), filled(b, 2)]);
    check('div', [filled(a, 3), positive(b, 4)]);
  });

  it.each([[[]], [[5]], [[1, 4]], [[2, 3, 2]]])('neg, exp and log with %j', (shape) => {
    check('neg', [filled(shape, 5)]);
    check('exp', [filled(shape, 6)]);
    check('log', [positive(shape, 7)]);
  });

  it.each([
    [[2, 3], 0, 1],
    [[2, 3], -1, -2],
    [[2, 3, 4], 0, 2],
    [[2, 3, 4], 1, 1],
  ])('transpose with %j dims %i and %i', (shape, dim0, dim1) => {
    check('transpose', [filled(shape, 8)], { dim0, dim1 });
  });

});

const reduceCases: [number[], number | number[] | null, boolean][] = [
  [[2, 3], null, false],
  [[2, 3], -1, false],
  [[2, 3], -1, true],
  [[2, 3], 0, true],
  [[2, 3, 4], [0, 2], false],
  [[2, 3, 4], [-1, -3], true],
  [[5], 0, false],
  [[5], -1, true],
  [[1, 5], -1, true],
];

describe('checkOp on reductions', () => {
  it.each(reduceCases)('sum and mean with %j dim %j keepdim %s', (shape, dim, keepdim) => {
    check('sum', [filled(shape, 10)], { dim, keepdim });
    check('mean', [filled(shape, 11)], { dim, keepdim });
  });

  it.each(reduceCases.filter(([, dim]) => dim !== null))('logsumexp with %j dim %j keepdim %s', (shape, dim, keepdim) => {
    check('logsumexp', [filled(shape, 12)], { dim, keepdim });
  });

  it.each(reduceCases.filter(([, dim]) => typeof dim === 'number'))('max with %j dim %j keepdim %s', (shape, dim, keepdim) => {
    check('max', [filled(shape, 13)], { dim, keepdim });
  });
});

describe('checkOp on softmax, matmul, gather and nll_loss', () => {
  it.each([
    [[5], 0],
    [[5], -1],
    [[2, 3], -1],
    [[2, 3], 0],
    [[2, 3, 4], 1],
  ])('softmax and log_softmax with %j dim %i', (shape, dim) => {
    check('softmax', [filled(shape, 14)], { dim });
    check('log_softmax', [filled(shape, 15)], { dim });
  });

  it.each([
    [[1, 4], [4, 5]],
    [[2, 3], [3, 2]],
    [[4], [4, 5]],
    [[3, 4], [4]],
  ])('matmul with %j and %j', (a, b) => {
    check('matmul', [filled(a, 16), filled(b, 17)]);
  });

  it('gather along the last and the first dim', () => {
    check('gather', [filled([3, 5], 18), Tensor.from([[0], [4], [2]])], { dim: -1 });
    check('gather', [filled([3, 5], 19), Tensor.from([[2, 1, 0, 0, 1]])], { dim: 0 });
    check('gather', [filled([5], 20), Tensor.from([4, 0])], { dim: 0 });
  });

  it.each(['mean', 'sum', 'none'])('nll_loss with reduction %s', (reduction) => {
    check('nll_loss', [filled([4, 5], 21), Tensor.from([0, 4, -100, 2])], { reduction, ignore_index: -100 });
    check('nll_loss', [filled([5], 22), Tensor.from(3)], { reduction, ignore_index: -100 });
    check('nll_loss', [filled([1, 5], 23), Tensor.from([1])], { reduction, ignore_index: 7 });
  });

  it('does not modify the inputs it perturbs', () => {
    const x = filled([3], 9);
    const before = Array.from(x.data);
    check('exp', [x]);
    expect(Array.from(x.data)).toEqual(before);
  });
});
