// @vitest-environment jsdom
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import { type Graph, Tensor, defineOp, graph } from '../../src/core';
import type { CompiledPage } from '../../src/dsl';
import { elemTexOf } from '../../src/explain/elem';
import { createSteps } from '../../src/steps';
import { autoLayout } from '../../src/scene3d/auto';
import { classAlignment, resolveLayout, targetColumn } from '../../src/scene3d/layout';
import { blockSpan } from '../../src/scene3d/modules';
import { loadExample } from './load';

const WEIGHTS: Record<string, string> = {
  E: 'E',
  'γ1': 'layer.input_layernorm.weight',
  Wq: 'layer.self_attn.Wq',
  Wk: 'layer.self_attn.Wk',
  Wv: 'layer.self_attn.Wv',
  Wo: 'layer.self_attn.Wo',
  'γq': 'layer.self_attn.gq',
  'γk': 'layer.self_attn.gk',
  'γ2': 'layer.post_attention_layernorm.weight',
  Wg: 'layer.mlp.Wg',
  Wu: 'layer.mlp.Wu',
  Wd: 'layer.mlp.Wd',
  'γ': 'norm.weight',
  W: 'head.W',
};

function buildJsPage(): Graph {
  const html = readFileSync(resolve(import.meta.dirname, '../../examples/qwen3-structure.html'), 'utf8');
  const script = /<script type="module">([\s\S]*?)<\/script>/.exec(html)![1].replace(/^import .*$/gm, '');
  document.body.innerHTML = /<body>([\s\S]*?)<script/.exec(html)![1];
  const mounted = () => ({ dispose() {} });
  const run = new Function('graph', 'defineOp', 'Tensor', 'createSteps', 'mountScene', 'board', 'stack', 'mountPanels', 'mountSteps', 'renderText', script);
  run(graph, defineOp, Tensor, createSteps, mounted, () => ({}), () => ({}), mounted, mounted, () => {});
  return (window as unknown as { vizflow: { g: Graph } }).vizflow.g;
}

describe('qwen3-dsl.yaml against qwen3-structure.html', () => {
  let js: Graph;
  let dsl: CompiledPage;

  beforeAll(async () => {
    js = buildJsPage();
    dsl = await loadExample('qwen3-dsl.yaml');
  });

  it('computes the same loss', () => {
    const jsLoss = js.findLoss()!.value.data[0];
    const dslLoss = dsl.graph.findLoss()!.value.data[0];
    expect(jsLoss.toFixed(3)).toBe('2.746');
    expect(Math.abs(dslLoss - jsLoss)).toBeLessThan(1e-9);
  });

  it('has the same weights in the same places', () => {
    const jsParams = js.nodes.filter((n) => n.kind === 'param').map((n) => n.name).sort();
    expect(jsParams).toEqual(Object.keys(WEIGHTS).sort());
    const dslParams = dsl.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name).sort();
    expect(dslParams).toEqual(Object.values(WEIGHTS).sort());
  });

  it('builds steps whose text, labels and node TeX render', () => {
    const steps = createSteps(dsl.graph, dsl.steps);
    const renders = (tex: string, what: string) => expect(() => katex.renderToString(tex, { throwOnError: true }), `${what}: ${tex}`).not.toThrow();
    const math = (text: string) => [...text.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)].map((m) => m[1] ?? m[2]);
    steps.list.forEach((_, i) => {
      const { title, body } = steps.text(i);
      for (const tex of [...math(title), ...math(body), ...math(steps.list[i].label)]) renders(tex, `step ${i}`);
    });
    for (const node of dsl.graph.nodes) {
      renders(node.tex, node.name);
      renders(elemTexOf(node, node.shape.map(() => 0)), `${node.name} elem`);
    }
    expect(steps.list.filter((s) => s.dir === 'backward').find((s) => s.nodes.includes(dsl.nodes.get('E')!))?.title).toBe('传到 embedding 表');
  });

  it('computes the same gradient for every weight', () => {
    for (const [jsName, path] of Object.entries(WEIGHTS)) {
      const expected = js.node(jsName)!;
      const actual = dsl.nodes.get(path)!;
      expect([...actual.shape], path).toEqual([...expected.shape]);
      expected.value.data.forEach((v, i) => expect(actual.value.data[i] === v, `${path} value [${i}]`).toBe(true));
      const want = expected.grad!.data;
      const got = actual.grad!.data;
      want.forEach((g, i) => expect(Math.abs(got[i] - g), `${path}[${i}]`).toBeLessThan(1e-9));
    }
  });

  it('starts the k chain beside the per-head q chains and draws one brace for q_norm and k_norm', () => {
    const { layout, modules } = autoLayout(dsl.tree);
    const g = dsl.graph;
    const { blocks } = resolveLayout(layout, { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 });
    const bottom = (path: string) => blockSpan(blocks.find((b) => b.node === dsl.nodes.get(`layer.self_attn.${path}`))!).bottom;
    for (const path of ['q[2]', 'k[1]']) expect(bottom(path)).toBeCloseTo(bottom('q[1]'), 9);
    expect(bottom('kn[1]')).toBeCloseTo(bottom('qn[1]'), 9);
    expect(modules.map((m) => m.title)).toEqual(expect.arrayContaining(['q_norm · k_norm', 'q_rope · k_rope', 'attn']));
  });
});
