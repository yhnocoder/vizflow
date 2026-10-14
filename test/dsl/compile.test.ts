import { describe, expect, it } from 'vitest';
import { type CompiledPage, loadPage, parseYaml } from '../../src/dsl';
import { runProgram } from '../../src/core';
import { levelPlan } from '../../src/scene3d/levels';
import { createSteps } from '../../src/steps';
import { decodeProgram } from '../../src/ui/figure-spec';

const BASE = 'mem:/pages/page.yaml';

function memory(files: Record<string, string>): (url: string) => Promise<string> {
  return async (url) => {
    const text = files[url];
    if (text === undefined) throw new Error(`no file ${url}`);
    return text;
  };
}

function page(yaml: string, files: Record<string, string> = {}): Promise<CompiledPage> {
  return loadPage(BASE, { read: memory({ [BASE]: yaml, ...files }) });
}

function indent(text: string, spaces: number): string {
  return text
    .trim()
    .split('\n')
    .map((line) => ' '.repeat(spaces) + line)
    .join('\n');
}

interface Root {
  args?: string;
  inputs?: string;
  outputs?: string;
  init?: string;
  forward: string;
  tex?: string;
  steps: string;
  unshown?: string;
  config?: string;
  data?: string;
  extra?: string;
}

function root(spec: Root): string {
  return `title: t
model: M
config: ${spec.config ?? '{ d: 2 }'}
data:
${indent(spec.data ?? 'x: [[1, 2], [3, 4], [5, 6]]', 2)}
${spec.extra ?? ''}
modules:
  M:
    args:
${indent(spec.args ?? 'd: dim', 6)}
    inputs:
${indent(spec.inputs ?? 'x: { type: tensor, shape: [T, d], dtype: fp32 }', 6)}
    outputs:
${indent(spec.outputs ?? 'l: { type: tensor, shape: [], dtype: fp32 }', 6)}
${spec.init === undefined ? '' : `    init:\n${indent(spec.init, 6)}`}
    forward:
${indent(spec.forward, 6)}
${spec.tex === undefined ? '' : `    tex:\n${indent(spec.tex, 6)}`}
${spec.unshown === undefined ? '' : `    unshown: ${spec.unshown}`}
    steps:
${indent(spec.steps, 6)}
`;
}

const SCALE = `
module: Scale
args:
  n: dim
  factor: { type: float, default: 2 }
inputs:
  x: { type: tensor, shape: ['...', n], dtype: dtype }
outputs:
  out: { type: tensor, shape: ['...', n], dtype: dtype }
init:
  weights:
    weight: { shape: [n], init: ones }
forward:
  - h: x * factor
  - out: h * weight
tex:
  h: '\\viz{x}\\cdot 2'
  weight: 'w'
formula: '\\step{mul}{\\viz{out}=\\viz{weight}\\odot}\\step{double}{2\\viz{x}}'
steps:
  - id: double
    show: h
    title: double
    text: 'h = {{ h | matrix:1 }}'
  - id: mul
    show: [weight, out]
    title: multiply
`;

const SCALE_FILE = 'mem:/pages/scale.yaml';

describe('parseYaml', () => {
  it('reports syntax errors with file, line and column', () => {
    expect(() => parseYaml('a: [1, 2\n', 'page.yaml')).toThrow(/page\.yaml:\d+:\d+/);
  });
});

describe('args and inputs', () => {
  it('passes args as a dict or a list, takes defaults and binds T from the data', async () => {
    const yaml = root({
      init: `
a: { use: Scale, args: { n: d, factor: 3 } }
b: { use: Scale, args: [{ n: d }] }
c: { use: Scale, args: [n] }`,
      args: 'd: dim\nn: dim',
      config: '{ d: 2, n: 2 }',
      extra: 'import: { Scale: ./scale.yaml }',
      forward: '- y: a(x)\n- z: b(y)\n- w: c(z)\n- l: w.mean()',
      steps: '- id: a\n- id: b\n- id: c\n- id: l',
    });
    const p = await page(yaml, { [SCALE_FILE]: SCALE });
    expect(Array.from(p.nodes.get('w')!.value.data)).toEqual([12, 24, 36, 48, 60, 72]);
    expect(p.nodes.get('a.weight')!.shape).toEqual([2]);
    expect(p.graph.dims.T).toBe(3);
  });

  it('rejects unknown and missing args and a misspelled list name', async () => {
    const make = (init: string) => page(root({ init, forward: '- y: a(x)\n- l: y.mean()', steps: '- id: a\n- id: l', extra: 'import: { Scale: ./scale.yaml }' }), { [SCALE_FILE]: SCALE });
    await expect(make('a: { use: Scale, args: { n: d, size: 2 } }')).rejects.toThrow(/page\.yaml:\d+:\d+ .*module Scale has no arg 'size'/);
    await expect(make('a: { use: Scale }')).rejects.toThrow(/module Scale needs arg 'n'/);
    await expect(make('a: { use: Scale, args: [dd] }')).rejects.toThrow(/name 'dd' is not defined/);
  });

  it('creates an optional weight only when its present arg is true', async () => {
    const make = (args: string) => page(root({ init: `a: { use: Linear, args: ${args} }`, forward: '- y: a(x)\n- l: y.mean()', steps: '- id: a\n- id: l' }));
    const params = (p: CompiledPage) => p.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name);
    const biased = await make('{ in_features: d, out_features: d }');
    expect(params(biased)).toEqual(['a.weight', 'a.bias']);
    biased.nodes.get('a.weight')!.set([[1, 0], [0, 1]]);
    biased.nodes.get('a.bias')!.set([10, 20]);
    expect(Array.from(biased.nodes.get('y')!.value.data)).toEqual([11, 22, 13, 24, 15, 26]);
    expect(params(await make('{ in_features: d, out_features: d, bias: False }'))).toEqual(['a.weight']);
  });

  it('rejects a shape or dtype name that binds from an input but is used nowhere else', async () => {
    const typo = SCALE.replace("x: { type: tensor, shape: ['...', n]", "x: { type: tensor, shape: ['...', nn]");
    const yaml = root({ init: 'a: { use: Scale, args: { n: d } }', extra: 'import: { Scale: ./scale.yaml }', forward: '- y: a(x)\n- l: y.mean()', steps: '- id: a\n- id: l' });
    await expect(page(yaml, { [SCALE_FILE]: typo })).rejects.toThrow(/'nn' takes its value from this input but is used nowhere else in Scale; if it should be an arg, check the spelling/);
    const prose = typo.replace("text: 'h = {{ h | matrix:1 }}'", "text: 'nn rows: {{ h | matrix:1 }}'");
    await expect(page(yaml, { [SCALE_FILE]: prose })).rejects.toThrow(/'nn' takes its value/);
    const shown = typo.replace("text: 'h = {{ h | matrix:1 }}'", "text: '{{ nn }} rows: {{ h | matrix:1 }}'");
    expect((await page(yaml, { [SCALE_FILE]: shown })).nodes.get('y')!.shape).toEqual([3, 2]);
    const anonymous = SCALE.replace("x: { type: tensor, shape: ['...', n]", "x: { type: tensor, shape: ['...', _]");
    expect((await page(yaml, { [SCALE_FILE]: anonymous })).nodes.get('y')!.shape).toEqual([3, 2]);
  });

  it('checks input shapes and keeps init to the args', async () => {
    const yaml = (inputs: string, init = '') => page(root({ inputs, init: init || undefined, forward: '- l: x.mean()', steps: '- id: l' }));
    await expect(yaml('x: { type: tensor, shape: [T, 3], dtype: fp32 }')).rejects.toThrow(/page\.yaml:5:6 data\.x: node 'x': expected shape \(3, 3\)/);
    await expect(yaml('x: { type: tensor, shape: [T, d], dtype: fp32 }', 'weights: { w: { shape: [T], init: ones } }')).rejects.toThrow(/name 'T' is not defined/);
  });
});

describe('forward', () => {
  it('names results by path, shares weights between calls and keeps each call apart', async () => {
    const p = await page(
      root({
        init: 's: { use: Scale, args: [{ n: d }] }',
        extra: 'import: { Scale: ./scale.yaml }',
        forward: '- y: s(x)\n- z: s(y)\n- l: z.mean()',
        steps: '- id: y\n- id: z\n- id: l',
      }),
      { [SCALE_FILE]: SCALE },
    );
    expect(['s.h', 'y', 's#2.h', 'z'].map((n) => p.nodes.get(n)?.name)).toEqual(['s.h', 'y', 's#2.h', 'z']);
    expect(p.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name)).toEqual(['s.weight']);
    expect(p.modules.map((m) => m.id)).toEqual(['s', 's#2']);
  });

  it('chains the first input through a module list and passes keywords to every layer', async () => {
    const LAYER = `
module: Layer
args: { n: dim }
inputs:
  x: { type: tensor, shape: [T, n], dtype: dtype }
  bias: { type: tensor, shape: [T, n], dtype: dtype }
outputs:
  out: { type: tensor, shape: [T, n], dtype: dtype }
init:
  weights:
    weight: { shape: [n], init: ones }
forward:
  - out: x * weight + bias
tex:
  out: 'o'
steps:
  - id: out
`;
    const p = await page(
      root({
        init: "layers: { use: 'list[Layer]', count: 3, args: [{ n: d }] }",
        extra: 'import: { Layer: ./layer.yaml }',
        forward: '- y: layers(x, bias=x)\n- l: y.mean()',
        steps: '- id: layers\n  loop: layer = layers[i]\n  defaults: { subtitle: "layer {{ i + 1 }}" }\n  rules:\n    - { at: -1, subtitle: last }\n- id: l',
      }),
      { 'mem:/pages/layer.yaml': LAYER },
    );
    expect(Array.from(p.nodes.get('y')!.value.data)).toEqual([4, 8, 12, 16, 20, 24]);
    expect(p.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name)).toEqual(['layers[0].weight', 'layers[1].weight', 'layers[2].weight']);
    expect(p.nodes.get('layers[1].out')!.tex).toBe('o^{[1]}');
    expect(p.modules.map((m) => m.subtitle)).toEqual(['layer 1', 'layer 2', 'last']);
  });

  it('returns named results and rejects a name defined twice', async () => {
    const yaml = (forward: string) => page(root({ init: 'nll: NLLLoss', extra: 'import: { NLLLoss: vizflow/nn/nll-loss }', data: 'x: [[-1, -2], [-3, -4]]\ny: [0, 1]', inputs: 'x: { type: tensor, shape: [T, d], dtype: fp32 }\ny: { type: tensor, shape: [T], dtype: int64 }', forward, steps: '- id: nll' }));
    const p = await yaml('- result: nll(logp=x, y=y)\n- l: result.loss\n- lt: result.lt');
    expect(p.nodes.get('l')!.value.data[0]).toBe(2.5);
    expect(p.nodes.get('lt')!.tex).toBe('\\mathbf{lt}');
    await expect(yaml('- result: nll(logp=x, y=y)\n- l: result.loss\n- l: result.lt')).rejects.toThrow(/name 'l' is already defined in M/);
  });
});

describe('coverage of forward results', () => {
  it('reports a result no step shows and accepts unshown and layout views of shown tensors', async () => {
    const yaml = (forward: string, steps: string, unshown?: string) => page(root({ forward, steps, unshown }));
    await expect(yaml('- y: x * 2\n- l: y.mean()', '- id: l')).rejects.toThrow(/'y' of M is not shown by any step/);
    await expect(yaml('- y: x * 2\n- l: y.mean()', '- id: l', '[y]')).resolves.toBeDefined();
    await expect(yaml('- y: x * 2\n- yt: y.transpose(0, 1)\n- l: yt.mean()', '- id: y\n- id: l')).resolves.toBeDefined();
  });
});

describe('steps', () => {
  it('patches nested module steps by path and appends text', async () => {
    const p = await page(
      root({
        init: 's: { use: Scale, args: [{ n: d }] }',
        extra: 'import: { Scale: ./scale.yaml }',
        forward: '- y: s(x)\n- l: y.mean()',
        steps: '- id: s\n  title: outer\n  edit:\n    double: { append: more, title: twice }\n- id: l',
      }),
      { [SCALE_FILE]: SCALE },
    );
    const titles = p.steps.forward.map((s) => s.title);
    expect(titles).toEqual(['twice', 'multiply', 'l']);
    expect(p.steps.forward[0].text).toMatch(/^h = \\begin\{pmatrix\} \{\{s\.h\[0, 0\]:1\}\} .*\\end\{pmatrix\}\nmore$/);
    expect(p.modules[0].title).toBe('outer');
    await expect(
      page(root({ init: 's: { use: Scale, args: [{ n: d }] }', extra: 'import: { Scale: ./scale.yaml }', forward: '- y: s(x)\n- l: y.mean()', steps: '- id: s\n  edit:\n    nope: { text: x }\n- id: l' }), { [SCALE_FILE]: SCALE }),
    ).rejects.toThrow(/there is no step 'nope'; the steps here are double, mul/);
  });

  it('marks the formula of a module call with the steps it names', async () => {
    const p = await page(root({ init: 's: { use: Scale, args: [{ n: d }] }', extra: 'import: { Scale: ./scale.yaml }', forward: '- y: s(x)\n- l: y.mean()', steps: '- id: s\n- id: l' }), { [SCALE_FILE]: SCALE });
    expect(p.steps.forward[0].formula?.tex).toBe('\\htmlData{vf-step=s0}{\\mathbf{y}=w\\odot}\\htmlData{vf-step=s1}{2\\mathbf{x}}');
    expect(p.steps.forward.map((s) => s.marks)).toEqual([['s1'], ['s0'], undefined]);
  });

  it('builds the overview and the inputs step and runs forward and backward', async () => {
    const p = await page(root({ forward: '- y: x * 2\n- l: y.mean()', steps: '- { id: overview, show: all(), title: all, text: "T is {{ T }}" }\n- { id: in, show: inputs() + parameters(), title: inputs }\n- id: y\n- id: l' }));
    expect(p.steps.forward[0]).toMatchObject({ title: 'all', text: 'T is 3' });
    const steps = createSteps(p.graph, p.steps);
    expect(steps.list.slice(0, 2).map((s) => s.kind)).toEqual(['overview', 'inputs']);
    expect(steps.list.filter((s) => s.dir === 'backward').length).toBeGreaterThan(0);
    steps.dispose();
  });
});

describe('layout', () => {
  it('places a tensor that several steps show once', async () => {
    const p = await page(root({ forward: '- y: x * 2\n- l: y.mean()', steps: '- id: y\n- { id: again, show: [y, l] }' }));
    const { blocks } = levelPlan(p.steps, p.levels).resolve({ folded: () => false, aligned: new Map() });
    expect(blocks.filter((b) => b.node === p.nodes.get('y')).length).toBe(1);
  });
});

describe('expressions', () => {
  it('follows Python semantics for lists, reflected operators and new axes', async () => {
    const forward = ['c: cat([x, x], dim=0)', 'p: x.permute([1, 0])', 'r: x.reshape([d, T])', 'v: x.view([T * d])', 'a: 1 - x', 'b: 1 / x', 'u: x[:, None]', 's: x[1, :]', 'l: c.mean() + p.mean() + r.mean() + v.mean() + a.mean() + b.mean() + u.mean() + s.mean()'];
    const p = await page(root({ forward: forward.map((line) => `- ${line}`).join('\n'), unshown: '[c, p, r, v, a, b, u, s]', steps: '- id: l' }));
    const value = (name: string) => p.nodes.get(name)!.value;
    expect(value('c').shape).toEqual([6, 2]);
    expect([value('p').shape, value('r').shape, value('v').shape, value('u').shape]).toEqual([[2, 3], [2, 3], [6], [3, 1, 2]]);
    expect(Array.from(value('a').data)).toEqual([0, -1, -2, -3, -4, -5]);
    expect(Array.from(value('b').data)).toEqual([1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6]);
    expect(Array.from(value('s').data)).toEqual([3, 4]);
    expect(p.graph.nodes.filter((n) => n.name.startsWith('s')).map((n) => n.name)).toEqual(['s']);
  });

  it('rejects arange with a zero step', async () => {
    await expect(page(root({ forward: '- t: arange(0, 4, 0)\n- l: x.mean()', steps: '- id: l' }))).rejects.toThrow(/arange: step must be nonzero/);
  });
});

describe('templates and tex', () => {
  it('interpolates scalars and formatted tensors and rejects a tensor without a format', async () => {
    const yaml = (text: string) => page(root({ forward: '- y: x * 2\n- l: y.mean()', steps: `- id: y\n  text: '${text}'\n- id: l` }));
    expect((await yaml('d={{ d }}, {{ y[0] | join: ", " }}')).steps.forward[0].text).toBe('d=2, {{y[0, 0]}}, {{y[0, 1]}}');
    expect((await yaml('$$\\viz{y}={{ y[:1] | matrix:2 }}$$')).steps.forward[0].text).toBe('$$\\mathbf{y}=\\begin{pmatrix} {{y[0, 0]:2}} & {{y[0, 1]:2}} \\end{pmatrix}$$');
    await expect(yaml('{{ y }}')).rejects.toThrow(/is not a single number; add a format/);
  });

  it('reads real sizes in root templates and prints integers as digits', async () => {
    const yaml = (text: string) => page(root({ config: '{ d: 2, real: { big: { name: Big, d: 4096, V: 151936 } } }', forward: '- y: x * 2\n- l: y.mean()', steps: `- id: y\n  text: "${text}"\n- id: l` }));
    expect((await yaml("{{ real['big'].name }}: {{ real['big'].V }}, {{ real['big'].d // d }}, {{ 0.00001 }}")).steps.forward[0].text).toBe('Big: 151936, 2048, 10^{-5}');
    await expect(yaml("{{ real['small'].d }}")).rejects.toThrow("real has no 'small'; it has big");
    await expect(yaml("{{ real['big'].T }}")).rejects.toThrow("real size 'big' has no 'T'; it has name, d, V");
  });

  it('wraps the blocks of a matrix to at most eight columns per line', async () => {
    const block = JSON.stringify([[1, 2, 3, 4]]);
    const yaml = (count: number) =>
      page(root({ config: '{ d: 4 }', inputs: 'x: { type: tensor, shape: [T, 1, d], dtype: fp32 }', data: `x: [${Array(count).fill(block).join(', ')}]`, forward: '- l: x.mean()', steps: "- id: l\n  text: '{{ x | matrix:0 }}'" }));
    const lines = (text: string) => /\\begin\{array\}\{l\}(.*)\\end\{array\}/.exec(text)?.[1].split('\\\\[1ex]').map((line) => (line.match(/\\begin\{pmatrix\}/g) ?? []).length);
    expect(lines((await yaml(3)).steps.forward[0].text ?? '')).toEqual([2, 1]);
    expect((await yaml(2)).steps.forward[0].text).not.toContain('array');
  });

  it('computes figure data from a template expression outside the graph', async () => {
    const yaml = (text: string) => page(root({ forward: '- y: x * 2\n- l: y.mean()', steps: `- id: y\n  text: '${text}'\n- id: l` }));
    const p = await yaml('{{ cos(arange(5)[:, None] * y[0]) | lines:shade=T }}');
    const tag = /^<vf-figure name="lines" program="([^"]+)" args="shade=3"><\/vf-figure>$/.exec(p.steps.forward[0].text ?? '');
    expect(tag).not.toBeNull();
    const value = runProgram(decodeProgram(tag![1]), (name) => p.nodes.get(name)!.value);
    expect(value.shape).toEqual([5, 2]);
    expect(value.get([4, 1])).toBeCloseTo(Math.cos(16), 12);
    expect(p.graph.nodes.map((n) => n.name)).toEqual(['x', 'y', 'l']);
    await expect(yaml('{{ y | lines:shade=T + 1 }}')).rejects.toThrow("lines shade must be an integer from 0 to 3, got 4 in '{{ y | lines:shade=T + 1 }}'");
    await expect(yaml('{{ y[0] | lines }}')).rejects.toThrow('lines needs a (N, K) tensor with at least 2 rows, got shape (2)');
    await expect(yaml('{{ tan(y) | lines }}')).rejects.toThrow("unknown function 'tan'");
  });

  it('evaluates buffer values with ops and broadcasting', async () => {
    const p = await page(root({ init: "buffers:\n  b: { shape: [2, d], dtype: fp32, value: 'exp(arange(2)[:, None] * arange(d))' }", forward: '- y: x * 2\n- l: y.mean()', steps: '- id: y\n- id: l' }));
    expect([...p.nodes.get('b')!.value.data]).toEqual([1, 1, 1, Math.E]);
  });

  it('expands preamble macros and takes tex of a child result from the outer module', async () => {
    const p = await page(
      root({
        init: 's: { use: Scale, args: [{ n: d }] }',
        extra: 'import: { Scale: ./scale.yaml }\npreamble: ./macros.sty',
        forward: '- y: s(x)\n- l: y.mean()',
        tex: "s.h: 'H'\ny: '\\pd{y}'",
        steps: '- id: s\n- id: l',
      }),
      { [SCALE_FILE]: SCALE, 'mem:/pages/macros.sty': '\\newcommand{\\pd}[1]{\\partial #1}' },
    );
    expect(p.nodes.get('s.h')!.tex).toBe('H');
    expect(p.nodes.get('y')!.tex).toBe('{\\partial y}');
  });
});

describe('page data', () => {
  it('computes data with tokens_to_ids and slices and re-evaluates it when the sequence changes', async () => {
    const p = await page(
      root({
        config: '{ d: 2, tokenizer: { vocab: [a, b, c] } }',
        data: "tokens: [a, b, c, a]\nids: { expr: 'tokens_to_ids(tokens[:-1])' }",
        inputs: 'tokens: { type: list, dtype: str }\nids: { type: tensor, shape: [T], dtype: int64 }',
        init: 'emb: { use: Embedding, args: { num_embeddings: 3, embedding_dim: d } }',
        forward: '- e: emb(ids)\n- l: e.mean()',
        steps: '- id: emb\n- id: l',
        extra: 'panels:\n  - { sequence: true, tokens: tokens, input: ids, label: ids }',
      }),
    );
    expect(Array.from(p.nodes.get('ids')!.value.data)).toEqual([0, 1, 2]);
    p.sequence!.set([2, 2, 1, 0]);
    expect(Array.from(p.nodes.get('ids')!.value.data)).toEqual([2, 2, 1]);
  });

  it('rejects config that is not an arg of the model', async () => {
    await expect(page(root({ config: '{ d: 2, e: 3 }', forward: '- l: x.mean()', steps: '- id: l' }))).rejects.toThrow(/config 'e' is not an arg of M/);
  });
});
