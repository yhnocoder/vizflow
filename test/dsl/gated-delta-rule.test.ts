import { describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, parseYaml, type PageFile } from '../../src/dsl';

const INPUTS = `
inputs:
  q:    { value: [[[0.3, -0.8], [1.1, 0.2]], [[-0.5, 0.9], [0.4, -1.2]], [[0.7, 0.1], [-0.3, 0.6]], [[-1.0, 0.4], [0.8, 0.5]], [[0.2, -0.6], [-0.9, -0.1]]] }
  k:    { value: [[[0.6, 0.2], [-0.4, 1.0]], [[0.1, -0.7], [0.9, 0.3]], [[-0.8, 0.5], [0.2, -0.6]], [[0.4, 0.9], [-0.7, 0.2]], [[1.2, -0.3], [0.5, 0.8]]] }
  v:    { value: [[[0.5, -1.0], [0.3, 0.7]], [[-0.2, 0.4], [1.1, -0.5]], [[0.9, 0.6], [-0.8, 0.2]], [[-0.4, -0.3], [0.6, 1.0]], [[0.1, 0.8], [-0.6, -0.9]]] }
  g:    { value: [[-0.3, -1.2], [-0.7, -0.1], [-2.0, -0.5], [-0.05, -0.9], [-0.4, -1.5]] }
  beta: { value: [[0.6, 0.2], [0.9, 0.5], [0.3, 0.8], [0.7, 0.4], [0.5, 0.95]] }
  w:    { value: [[[0.2, -0.5], [0.7, 0.1]], [[-0.3, 0.4], [0.6, -0.8]], [[0.9, 0.3], [-0.2, 0.5]], [[0.1, -0.6], [0.4, 0.2]], [[-0.7, 0.8], [0.3, -0.4]]] }
`;

function spec(instance: string): PageFile {
  return parseYaml(`
title: t
import: { Chunk: vizflow/nn/gated-delta-rule-chunk, Recurrent: vizflow/nn/gated-delta-rule }
dims: {}
${INPUTS}
body:
  - rule: { ${instance}, inputs: { q: q, k: k, v: v, g: g, beta: beta }, outputs: { o: o } }
  - loss: sum(o * w)
`, 'mem:/page.yaml') as PageFile;
}

async function run(instance: string): Promise<CompiledPage> {
  return compilePage(spec(instance));
}

function close(a: Float64Array, b: Float64Array): void {
  expect(a.length).toBe(b.length);
  a.forEach((x, i) => expect(Math.abs(x - b[i])).toBeLessThanOrEqual(1e-12 + 1e-12 * Math.abs(b[i])));
}

describe('torch_chunk_gated_delta_rule and torch_recurrent_gated_delta_rule', () => {
  it.each([1, 2, 3, 5, 8])('give the same output and gradients with chunk %i', async (chunk) => {
    const recurrent = await run('use: Recurrent');
    const chunked = await run(`use: Chunk, params: { chunk: ${chunk} }`);
    close(chunked.nodes.get('o')!.value.data, recurrent.nodes.get('o')!.value.data);
    for (const name of ['q', 'k', 'v', 'g', 'beta']) close(chunked.nodes.get(name)!.grad!.data, recurrent.nodes.get(name)!.grad!.data);
    const padded = Math.ceil(5 / chunk) * chunk;
    expect(chunked.nodes.get('rule.qp')!.shape).toEqual([2, padded, 2]);
    expect(chunked.nodes.get(`rule.S[2][${padded / chunk}]`)!.shape).toEqual([2, 2]);
  });
});
