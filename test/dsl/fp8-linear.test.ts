import { beforeAll, describe, expect, it } from 'vitest';
import type { CompiledPage } from '../../src/dsl';
import { roundToDtype } from '../../src/autograd/rounding';
import { createSteps } from '../../src/steps';
import { type EncodedTensor, decode, expectTensor, loadFixture, nested } from '../torch/fixtures';
import { loadExample } from './load';

interface LinearFixture {
  x: EncodedTensor;
  W: EncodedTensor;
  y: number[];
  z: EncodedTensor;
  zr: EncodedTensor;
  logp: EncodedTensor;
  loss: number;
  grads: { x: EncodedTensor; W: EncodedTensor };
}

const { linear } = loadFixture<never>('quant.json') as unknown as { linear: LinearFixture };

describe('fp8-linear-dsl.yaml against the FP8 linear fixture', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadExample('fp8-linear-dsl.yaml');
    page.nodes.get('W')!.set(nested(linear.W));
  });

  it('quantizes W through the inserted W_quant instance and x through act_quant beside it', () => {
    expect(page.nodes.get('W')!.kind).toBe('param');
    expect(page.nodes.get('x')!.kind).toBe('input');
    expect([...page.nodes.get('x')!.realShape]).toEqual([3, 7168]);
    expect(page.nodes.get('z')!.inputs.map((n) => n.name)).toEqual(['act_quant.xq', 'W_quant.xq.T']);
    expect(page.nodes.get('zr')!.inputs.map((n) => n.name)).toEqual(['x', 'W.T']);
    expect(page.modules.find((m) => m.id === 'W_quant')!.folded).toBe(false);
    for (const prefix of ['act_quant', 'W_quant']) {
      const dtypes = ['s', 'q', 'xq', 'e'].map((name) => page.nodes.get(`${prefix}.${name}`)!.dtype);
      expect(dtypes, prefix).toEqual(['fp32', 'fp8_e4m3', 'bf16', 'bf16']);
    }
    expect([...page.nodes.get('act_quant.s')!.shape]).toEqual([3, 2]);
    expect([...page.nodes.get('W_quant.s')!.realShape]).toEqual([144, 56]);
    for (const name of ['act_quant.e', 'W_quant.e', 'zr', 'dz']) expect(page.nodes.get(name)!.grad, name).toBeNull();
  });

  it('rounds every value to its dtype like the fixture and matches its outputs, loss and gradients', () => {
    expect(page.graph.rounding).toBe(true);
    const values = (name: string) => [...page.nodes.get(name)!.value.data];
    expect(values('x')).toEqual(decode(linear.x.values).map((v) => roundToDtype(v, 'bf16')));
    expect(page.nodes.get('y')!.value.toNested()).toEqual(linear.y);
    expect(values('z')).toEqual(decode(linear.z.values));
    expect(values('zr')).toEqual(decode(linear.zr.values));
    expect(values('logp')).toEqual(decode(linear.logp.values));
    expect(page.graph.findLoss()!.value.data[0]).toBe(linear.loss);
    expectTensor(page.nodes.get('x')!.grad, linear.grads.x, 'grad of x');
    expectTensor(page.nodes.get('W')!.grad, linear.grads.W, 'grad of W');
  });

  it('gives every forward step a title, a label and a text', () => {
    const steps = createSteps(page.graph, page.steps);
    for (const step of steps.list.filter((s) => s.dir === 'forward')) {
      expect(step.title, step.id).not.toBe('');
      expect(step.label, step.id).not.toBe('');
      expect(step.text, step.id).not.toBe('');
    }
  });
});
