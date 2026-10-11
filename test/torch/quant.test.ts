import { describe, expect, it } from 'vitest';
import { type QuantFormat, type ScaleRule, graph } from '../../src/core';
import { roundTo } from '../../src/autograd/quant';
import { ABS_TOLERANCE, type EncodedTensor, decode, expectTensor, loadFixture } from './fixtures';

interface Rounding {
  x: number[];
  y: number[];
}

interface BlockCase {
  title: string;
  fmt: QuantFormat;
  rule: ScaleRule;
  br: number;
  bc: number;
  x: EncodedTensor;
  s: EncodedTensor;
  q: EncodedTensor;
  xq: EncodedTensor;
}

interface QuantFixture {
  torch: string;
  ops: string[];
  rounding: Record<'fp8_e4m3' | 'fp8_e5m2' | 'fp4_e2m1', Rounding>;
  cases: BlockCase[];
  linear: {
    b: number;
    x: EncodedTensor;
    W: EncodedTensor;
    y: number[];
    sx: EncodedTensor;
    qx: EncodedTensor;
    xq: EncodedTensor;
    sw: EncodedTensor;
    qw: EncodedTensor;
    Wq: EncodedTensor;
    z: EncodedTensor;
    zr: EncodedTensor;
    loss: number;
    grads: { x: EncodedTensor; W: EncodedTensor };
  };
}

const fixture = loadFixture<BlockCase>('quant.json') as unknown as QuantFixture;

function exact(actual: Float64Array | number[], expected: EncodedTensor, what: string): void {
  const values = decode(expected.values);
  const mismatches = values.flatMap((e, i) => (actual[i] === e ? [] : [`[${i}] vizflow ${actual[i]} vs torch ${e}`]));
  expect(mismatches, what).toEqual([]);
}

describe(`PyTorch ${fixture.torch} reference for block quantization`, () => {
  for (const fmt of ['fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1'] as const) {
    it(`rounds to ${fmt} bit for bit`, () => {
      const { x, y } = fixture.rounding[fmt];
      exact(x.map((v) => roundTo(fmt, v)), { shape: [y.length], values: y }, fmt);
    });
  }

  for (const c of fixture.cases) {
    it(`matches the official ${c.rule} formula: ${c.title}`, () => {
      const g = graph();
      const x = g.input('x', c.x.shape, decode(c.x.values));
      const s = g.block_scale(x, c.br, c.bc, c.fmt, c.rule, { name: 's' });
      const q = g.quantize(x, s, c.br, c.bc, c.fmt, { name: 'q' });
      const xq = g.dequantize(q, s, c.br, c.bc, { name: 'xq' });
      expect([...s.shape]).toEqual(c.s.shape);
      exact(s.value.data, c.s, 'scale');
      exact(q.value.data, c.q, 'codes');
      expectTensor(xq.value, c.xq, 'dequantized');
    });
  }

  it('matches an FP8 linear with a straight-through estimator in float64', () => {
    const { linear } = fixture;
    const g = graph({ dims: { b: linear.b } });
    const x = g.input('x', linear.x.shape, decode(linear.x.values));
    const W = g.param('W', linear.W.shape, decode(linear.W.values));
    const y = g.label('y', linear.y, { classes: linear.W.shape[0] });
    const sx = g.block_scale(x, 1, 'b', { name: 'sx' });
    const qx = g.quantize(x, sx, 1, 'b', { name: 'qx' });
    const xq = g.dequantize(qx, sx, 1, 'b', { name: 'xq' });
    const sw = g.block_scale(W, 'b', 'b', { name: 'sw' });
    const qw = g.quantize(W, sw, 'b', 'b', { name: 'qw' });
    const Wq = g.dequantize(qw, sw, 'b', 'b', { name: 'Wq' });
    const z = g.matmul(xq, Wq.T, { name: 'z' });
    const loss = g.nll_loss(g.log_softmax(z, -1), y, { name: 'loss' });
    exact(sx.value.data, linear.sx, 'sx');
    exact(qx.value.data, linear.qx, 'qx');
    exact(sw.value.data, linear.sw, 'sw');
    exact(qw.value.data, linear.qw, 'qw');
    expectTensor(xq.value, linear.xq, 'xq');
    expectTensor(Wq.value, linear.Wq, 'Wq');
    expectTensor(z.value, linear.z, 'z');
    expect(Math.abs(loss.value.data[0] - linear.loss)).toBeLessThanOrEqual(ABS_TOLERANCE);
    expectTensor(x.grad, linear.grads.x, 'grad of x');
    expectTensor(W.grad, linear.grads.W, 'grad of W');
  });
});
