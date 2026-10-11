import { readFile } from 'node:fs/promises';
import { beforeAll, describe, expect, it } from 'vitest';
import { type Node, moduleJacobian, opDef } from '../../src/core';
import type { AnyOpDef } from '../../src/autograd/op';
import type { PageModule } from '../../src/steps';
import type { Tensor } from '../../src/tensor';
import { type CompiledPage, loadPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';
import { examplesUrl, readFileUrl } from './load';

const VARIANT = new URL('qwen3-variant.yaml', examplesUrl).href;
const EPSILON = 1e-6;
const SAMPLES = 8;

async function loadVariant(): Promise<CompiledPage> {
  const source = await readFile(new URL('qwen3-dsl.yaml', examplesUrl), 'utf8');
  const text = source
    .replace(/^dims: .*$/m, 'dims: { T: 3, d: 6, f: 10, V: 7, H: 4, Hkv: 2, dh: 8 }')
    .replace(/^vocab: .*$/m, 'vocab: [on, mat, the, sat, cat, dog, ran]')
    .replace(/theta: 10 \}/, 'theta: 10000 }')
    .replace(/^values:[\s\S]*$/m, '');
  expect(text).not.toBe(source);
  return loadPage(VARIANT, { read: (url) => (url === VARIANT ? Promise.resolve(text) : readFileUrl(url)) });
}

function moduleOutput(page: CompiledPage, module: PageModule, input: Node, output: Node, j: number, delta: number): Float64Array {
  const inside = new Set(module.nodes);
  const values = new Map<Node, Tensor>();
  const changed = input.value.clone();
  changed.data[j] += delta;
  values.set(input, changed);
  for (const n of page.graph.nodes) {
    if (n.kind !== 'op' || !inside.has(n) || n === input || n.id > output.id) continue;
    const def = opDef(n.op as string) as AnyOpDef;
    values.set(n, def.forward(n.inputs.map((x) => values.get(x) ?? x.value), n.attrs));
  }
  return Float64Array.from(values.get(output)!.data);
}

function lossOf(page: CompiledPage): number {
  return page.graph.findLoss()!.value.data[0];
}

function perturbed(page: CompiledPage, param: Node, j: number, delta: number): number {
  const tensor = param.value.clone();
  tensor.data[j] += delta;
  param.set(tensor.toNested());
  return lossOf(page);
}

describe('qwen3-dsl.yaml with 4 query heads, 2 kv heads and d_h = 8', () => {
  let page: CompiledPage;

  beforeAll(async () => {
    page = await loadVariant();
  });

  it('builds the attention for the new sizes', () => {
    const shape = (name: string) => page.nodes.get(name)!.shape;
    expect(shape('layer.self_attn.q')).toEqual([3, 32]);
    expect(shape('layer.self_attn.k')).toEqual([3, 16]);
    expect(shape('layer.self_attn.qrr')).toEqual([3, 4]);
    expect(shape('layer.self_attn.krr')).toEqual([3, 2]);
    expect(shape('layer.self_attn.kT[4]')).toEqual([8, 3]);
    expect(shape('layer.self_attn.o')).toEqual([3, 32]);
    expect(page.nodes.get('layer.self_attn.S[5]')).toBeUndefined();
    const k = page.nodes.get('layer.self_attn.apply_rotary_pos_emb.k_embed')!.value;
    const kp = (h: number) => Array.from(page.nodes.get(`layer.self_attn.kp[${h}]`)!.value.data);
    const column = (c: number) => [0, 1, 2].map((t) => k.data[t * 16 + c]);
    expect(kp(1).filter((_, i) => i % 8 === 0)).toEqual(column(0));
    expect(kp(2)).toEqual(kp(1));
    expect(kp(3).filter((_, i) => i % 8 === 0)).toEqual(column(8));
    expect(kp(4)).toEqual(kp(3));
  });

  it('uses theta from the YAML for the RoPE tables', () => {
    const cos = page.nodes.get('layer.self_attn.apply_rotary_pos_emb.cos')!.value;
    expect(cos.shape).toEqual([3, 8]);
    expect(cos.data[8 + 1]).toBeCloseTo(Math.cos(10000 ** (-2 / 8)), 12);
    expect(cos.data[8 + 5]).toBeCloseTo(Math.cos(10000 ** (-2 / 8)), 12);
  });

  it('derives the same step structure from the module tree', () => {
    const steps = createSteps(page.graph, page.steps);
    expect(steps.list.filter((s) => s.dir === 'forward')).toHaveLength(48);
    expect(steps.list.filter((s) => s.dir === 'backward')).toHaveLength(47);
    const scores = steps.list.find((s) => s.dir === 'forward' && s.kind === 'op' && s.nodes.some((n) => n.name === 'layer.self_attn.S[1]'))!;
    expect(scores.nodes.map((n) => n.name)).toEqual([1, 2, 3, 4].map((h) => `layer.self_attn.S[${h}]`));
    expect(scores.segment).toBe('self_attn');
    expect(steps.list.find((s) => s.kind === 'op' && s.nodes.some((n) => n.name === 'layer.self_attn.qsq'))!.sub).toBe('q_norm $\\cdot$ k_norm');
    for (let i = 0; i < steps.list.length; i++) expect(() => steps.text(i)).not.toThrow();
  });

  it('passes a finite-difference check on samples of every parameter', () => {
    const params = page.graph.nodes.filter((n) => n.kind === 'param');
    expect(params.length).toBe(14);
    for (const param of params) {
      const analytic = Float64Array.from(param.grad!.data);
      const original = param.value.toNested();
      const stride = Math.ceil(analytic.length / SAMPLES);
      for (let j = 0; j < analytic.length; j += stride) {
        const plus = perturbed(page, param, j, EPSILON);
        param.set(original);
        const minus = perturbed(page, param, j, -EPSILON);
        param.set(original);
        const numeric = (plus - minus) / (2 * EPSILON);
        expect(Math.abs(numeric - analytic[j]), `${param.name}[${j}]: ${analytic[j]} vs ${numeric}`).toBeLessThan(1e-6 + 1e-4 * Math.abs(numeric));
      }
    }
  });

  it('computes module Jacobians that agree with finite differences', () => {
    for (const id of ['layer.self_attn.qk_norm', 'layer.self_attn.apply_rotary_pos_emb', 'layer.self_attn', 'layer.mlp', 'layer.input_layernorm']) {
      const module = page.modules.find((m) => m.id === id)!;
      module.inputs.forEach((input, q) => {
        const output = module.outputs[q];
        const J = moduleJacobian(page.graph, module, input, output);
        expect(J.length, id).toBe(output.value.size);
        for (let j = 0; j < input.value.size; j++) {
          const plus = moduleOutput(page, module, input, output, j, EPSILON);
          const minus = moduleOutput(page, module, input, output, j, -EPSILON);
          for (let i = 0; i < output.value.size; i++) {
            const numeric = (plus[i] - minus[i]) / (2 * EPSILON);
            expect(Math.abs(J[i][j] - numeric), `${id} d${output.name}[${i}]/d${input.name}[${j}]`).toBeLessThan(1e-6 + 1e-4 * Math.abs(numeric));
          }
        }
      });
    }
  });
});
