import { afterEach, describe, expect, it, vi } from 'vitest';
import { type CompiledPage, type PageFile, compilePage, loadPage, parseYaml } from '../../src/dsl';
import type { Node } from '../../src/core';
import type { LayoutEntry, LayoutTree } from '../../src/scene3d/auto';

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

const NORM = `
module: Norm
params:
  eps: { type: float, default: 1e-6 }
inputs:
  x: { shape: [T, n] }
outputs: [out]
weights:
  W: { shape: [n], init: ones }
body:
  - ms: 'mean(x ** 2, -1, keepdim=true)'
  - out: x * rsqrt(ms + eps) * W
steps:
  - { id: rms, show: ms, title: 'rms of $\\of{x}$' }
  - { id: scale, show: [W, out], title: scale, text: 'out = {{ out | matrix:2 }}' }
`;

function names(entries: LayoutEntry[]): unknown[] {
  return entries.map((entry) => {
    if (entry.kind === 'node') return entry.node.name;
    if (entry.kind === 'parallel') return { parallel: entry.branches.map((b) => ({ title: b.title, children: names(b.children) })) };
    return { title: (entry as LayoutTree).title, children: names(entry.children) };
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('parseYaml', () => {
  it('keeps numbers, strings and the YAML 1.2 meaning of on', () => {
    expect(parseYaml('a: 1e-6\nb: on\nc: [x, "y"]\n', 'f.yaml')).toEqual({ a: 1e-6, b: 'on', c: ['x', 'y'] });
  });

  it('reports syntax errors with file, line and column', () => {
    expect(() => parseYaml('a: 1\n  b: [\n', 'bad.yaml')).toThrow(/^bad\.yaml:\d+:\d+: /);
  });
});

describe('expressions', () => {
  it('evaluates precedence, scalars, transposes, ranges and integer division', async () => {
    const result = await page(`
title: t
dims: { T: 2, n: 4, H: 2 }
inputs:
  x: { value: [[1, 2, 3, 4], [5, 6, 7, 8]] }
weights:
  W: { shape: [n, n], init: ones }
body:
  - a: x @ W.T * 2 + 1
  - b: a ** 2 - 1e-1
  - c: -b
  - for: h
    in: 1..H
    body:
      - p[h]: narrow(a, 1, (h-1)*(n//H), n//H)
  - d: 'cat(p[1..H], dim=1)'
  - e: 'sum(d, -1, keepdim=true) / 4'
`);
    const a = result.nodes.get('a')!;
    expect([...a.value.data]).toEqual([21, 21, 21, 21, 53, 53, 53, 53]);
    expect(result.nodes.get('b')!.value.data[0]).toBeCloseTo(440.9, 12);
    expect(result.nodes.get('c')!.value.data[0]).toBeCloseTo(-440.9, 12);
    expect(result.nodes.get('p[2]')!.shape).toEqual([2, 2]);
    expect([...result.nodes.get('d')!.value.data]).toEqual([...a.value.data]);
    expect([...result.nodes.get('e')!.value.data]).toEqual([21, 53]);
    expect(result.nodes.get('a/1')!.op).toBe('matmul');
    expect(result.nodes.get('a/2')!.op).toBe('mul');
    expect(result.nodes.get('a')!.op).toBe('add');
    expect(result.nodes.get('c')!.op).toBe('mul');
  });

  it('reports parse errors and unknown names with line and column', async () => {
    await expect(page('title: t\ndims: { T: 2 }\ninputs: {}\nbody:\n  - a: x +* 2\n')).rejects.toThrow(/^mem:\/pages\/page\.yaml:5:11 body\[0\]\.a: unexpected '\*'/);
    await expect(page('title: t\ndims: { T: 2 }\ninputs:\n  x: { value: [1, 2] }\nbody:\n  - a: x + zz\n')).rejects.toThrow(/page\.yaml:6:12 body\[0\]\.a: name 'zz' is not defined in the page/);
  });

  it('rejects an item that only refers to an existing node', async () => {
    await expect(page('title: t\ndims: {}\ninputs:\n  x: { value: [1, 2] }\nbody:\n  - a: x\n')).rejects.toThrow(/'a' must compute a new node/);
  });
});

describe('modules', () => {
  const files = { 'mem:/pages/norm.yaml': NORM };

  it('binds dimensions by matching input shapes and checks repeated names', async () => {
    const mismatch = `
module: Same
inputs:
  a: { shape: [T, n] }
  b: { shape: [T, n] }
outputs: [c]
body:
  - c: a + b
`;
    await expect(page(`
title: t
import: { Same: ./same.yaml }
dims: {}
inputs:
  x: { value: [[1, 2], [3, 4]] }
  y: { value: [[1, 2, 3], [4, 5, 6]] }
body:
  - s: { use: Same, inputs: { a: x, b: y }, outputs: { c: z } }
`, { 'mem:/pages/same.yaml': mismatch })).rejects.toThrow(/inputs\.b: input has shape \(2, 3\), but dimension 1 of Same's pattern \(T, n\) is 2/);
  });

  it('keeps scopes closed', async () => {
    const leaky = `
module: Leaky
inputs:
  x: { shape: [T] }
outputs: [out]
body:
  - out: x + secret
`;
    await expect(page(`
title: t
import: { Leaky: ./leaky.yaml }
dims: {}
inputs:
  secret: { value: [1, 2] }
  x: { value: [3, 4] }
body:
  - l: { use: Leaky, inputs: { x: x }, outputs: { out: y } }
`, { 'mem:/pages/leaky.yaml': leaky })).rejects.toThrow(/^mem:\/pages\/leaky\.yaml:7:14 body\[0\]\.out: name 'secret' is not defined in 'l' \(Leaky\)/);
  });

  it('checks parameter types and defaults', async () => {
    const scaled = `
module: Scaled
params:
  k: { type: int }
  act: { type: op, values: [relu, gelu], default: relu }
inputs:
  x: { shape: [n] }
outputs: [out]
body:
  - out: act(x * k)
`;
    const files = { 'mem:/pages/scaled.yaml': scaled };
    const use = (params: string) => page(`
title: t
import: { Scaled: ./scaled.yaml }
dims: {}
inputs:
  x: { value: [-1, 2] }
body:
  - s: { use: Scaled, ${params}inputs: { x: x }, outputs: { out: y } }
`, files);
    await expect(use('')).rejects.toThrow(/parameter 'k' has no value and no default/);
    await expect(use('params: { k: 1.5 }, ')).rejects.toThrow(/parameter 'k' must be an integer, got 1.5/);
    await expect(use('params: { k: 2, act: silu }, ')).rejects.toThrow(/parameter 'act' must be one of relu, gelu, got 'silu'/);
    await expect(use('params: { k: 2, q: 1 }, ')).rejects.toThrow(/module Scaled has no parameter 'q'/);
    const ok = await use('params: { k: 2, act: gelu }, ');
    expect(ok.nodes.get('s.out')!.op).toBe('gelu');
    expect(ok.nodes.get('y')).toBe(ok.nodes.get('s.out'));
  });

  it('shares weights through bindings and creates its own otherwise', async () => {
    const result = await page(`
title: t
import: { Norm: ./norm.yaml }
dims: { T: 2, n: 3 }
inputs:
  x: { value: [[1, 2, 3], [4, 5, 6]] }
weights:
  G: { shape: [n], init: ones }
body:
  - a: { use: Norm, inputs: { x: x }, weights: { W: G }, outputs: { out: y1 } }
  - b: { use: Norm, inputs: { x: y1 }, weights: { W: G }, outputs: { out: y2 } }
  - c: { use: Norm, inputs: { x: y2 }, outputs: { out: y3 } }
`, files);
    expect(result.nodes.get('a.W')).toBe(result.nodes.get('G'));
    expect(result.nodes.get('b.W')).toBe(result.nodes.get('G'));
    expect(result.nodes.get('c.W')).not.toBe(result.nodes.get('G'));
    expect(result.graph.nodes.filter((n) => n.kind === 'param').map((n) => n.name)).toEqual(['G', 'c.W']);
    await expect(page(`
title: t
import: { Norm: ./norm.yaml }
dims: {}
inputs:
  x: { value: [[1, 2, 3]] }
weights:
  G: { shape: [2], init: ones }
body:
  - a: { use: Norm, inputs: { x: x }, weights: { W: G }, outputs: { out: y } }
`, files)).rejects.toThrow(/weight 'W' of Norm has shape \(3\), but 'G' has shape \(2\)/);
  });

  it('shares buffers between instances with the same parameters and dimensions', async () => {
    const rope = await page(`
title: t
import: { RoPE: vizflow/nn/rope }
dims: {}
inputs:
  a: { value: [[1, 2, 3, 4], [5, 6, 7, 8]] }
  b: { value: [[1, 0, 1, 0], [0, 1, 0, 1]] }
  c: { value: [[1, 2], [3, 4]] }
body:
  - ra: { use: RoPE, params: { dh: 4 }, inputs: { q: a, k: b }, outputs: { q_embed: a2, k_embed: b2 } }
  - rb: { use: RoPE, params: { dh: 4 }, inputs: { q: b, k: a }, outputs: { q_embed: b3, k_embed: a3 } }
  - rc: { use: RoPE, params: { dh: 4, theta: 100 }, inputs: { q: b, k: b }, outputs: { q_embed: b4, k_embed: b5 } }
  - rd: { use: RoPE, params: { dh: 2 }, inputs: { q: c, k: c }, outputs: { q_embed: c2, k_embed: c3 } }
`);
    const buffers = rope.graph.nodes.filter((n) => n.kind === 'buffer').map((n) => n.name);
    expect(buffers).toEqual(['ra.cos', 'ra.sin', 'rc.cos', 'rc.sin', 'rd.cos', 'rd.sin']);
    expect(rope.nodes.get('rb.cos')).toBe(rope.nodes.get('ra.cos'));
    const cos = rope.nodes.get('ra.cos')!;
    expect(cos.value.get([1, 1])).toBeCloseTo(Math.cos(1 / 100), 15);
    expect(rope.nodes.get('rc.cos')!.value.get([1, 1])).toBeCloseTo(Math.cos(1 / 10), 15);
  });

  it('detects circular imports', async () => {
    await expect(page('title: t\nimport: { A: ./a.yaml }\ndims: {}\ninputs: {}\nbody: []\n', {
      'mem:/pages/a.yaml': 'module: A\nimport: { B: ./b.yaml }\ninputs: {}\noutputs: []\nbody: []\n',
      'mem:/pages/b.yaml': 'module: B\nimport: { A: ./a.yaml }\ninputs: {}\noutputs: []\nbody: []\n',
    })).rejects.toThrow(/circular import: mem:\/pages\/page\.yaml -> mem:\/pages\/a\.yaml -> mem:\/pages\/b\.yaml -> mem:\/pages\/a\.yaml/);
  });
});

describe('tex', () => {
  it('derives tex from names, adds instance subscripts and warns on duplicates', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const result = await page(`
title: t
import: { Norm: ./norm.yaml }
dims: { T: 2, n: 2, H: 2 }
inputs:
  x0: { value: [[1, 2], [3, 4]] }
weights:
  Wq: { shape: [n, n], init: ones }
  W_q: { shape: [n, n], init: ones }
body:
  - q: x0 @ Wq.T
  - gate: q * 2
  - LSE: 'logsumexp(gate, -1, keepdim=true)'
  - n1: { use: Norm, inputs: { x: q }, outputs: { out: h } }
  - n2: { use: Norm, inputs: { x: h }, outputs: { out: h2 } }
  - for: i
    in: 1..H
    body:
      - k[i]: q * i
      - nk[i]: { use: Norm, inputs: { x: 'k[i]' } }
  - one: { use: Norm, inputs: { x: q }, suffix: '^{s}', tex: { ms: '\\mu' } }
`, { 'mem:/pages/norm.yaml': NORM });
    const tex = (path: string) => result.nodes.get(path)!.tex;
    expect(tex('x0')).toBe('\\mathbf{x}_{0}');
    expect(tex('Wq')).toBe('W_{q}');
    expect(tex('W_q')).toBe('W_{q}');
    expect(tex('q')).toBe('\\mathbf{q}');
    expect(tex('gate')).toBe('\\mathbf{gate}');
    expect(tex('LSE')).toBe('\\mathrm{LSE}');
    expect(tex('k[2]')).toBe('\\mathbf{k}^{(2)}');
    expect(tex('n1.ms')).toBe('\\mathbf{ms}_{1}');
    expect(tex('n1.W')).toBe('W_{1}');
    expect(tex('n2.ms')).toBe('\\mathbf{ms}_{2}');
    expect(tex('h')).toBe('\\mathbf{h}');
    expect(tex('nk[2].ms')).toBe('\\mathbf{ms}_{2}');
    expect(tex('one.ms')).toBe('\\mu');
    expect(tex('one.W')).toBe('W^{s}');
    expect(result.nodes.get('n1.ms')!.elem).toBe('\\mathrm{ms}_{1}');
    expect(result.nodes.get('n1.out')!.tex).toBe('\\mathbf{h}');
    expect(result.nodes.get('n1.out/1')!.tex).toBe('\\mathbf{ms}_{1}+10^{-6}');
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/'n1\.ms', 'nk\[1\]\.ms' share the tex '\\mathbf\{ms\}_\{1\}'/));
  });

  it('merges subscripts, substitutes \\of and leaves tex with \\of alone', async () => {
    const files = { 'mem:/pages/norm.yaml': NORM, 'mem:/pages/act.yaml': `
module: Act
params:
  act: { type: op, default: silu }
  eps: { type: float, default: 2e-5 }
inputs:
  x: { shape: [n] }
outputs: [y]
weights:
  W_q: { shape: [n], init: ones }
body:
  - y: { op: act(x * W_q), tex: '\\of{act}(\\of{x})' }
steps:
  - { id: s, show: y, title: '$\\of{act.def}$ and $\\of{eps}$' }
` };
    const result = await page(`
title: t
import: { Act: ./act.yaml }
dims: {}
inputs:
  v: { value: [1, 2] }
body:
  - a1: { use: Act, inputs: { x: v } }
  - a2: { use: Act, inputs: { x: v } }
`, files);
    expect(result.nodes.get('a1.W_q')!.tex).toBe('W_{q,1}');
    expect(result.nodes.get('a2.y')!.tex).toBe('\\mathrm{SiLU}(\\mathbf{v})');
    expect(result.steps.forward[0].title).toBe('$\\mathrm{SiLU}(v)=v\\,\\sigma(v)$ and $2\\times 10^{-5}$');
  });
});

describe('steps', () => {
  const files = { 'mem:/pages/norm.yaml': NORM };

  it('inserts module steps at the instance and generates steps for uncovered nodes', async () => {
    const result = await page(`
title: t
import: { Norm: ./norm.yaml }
dims: {}
inputs:
  x: { value: [[1, 2], [3, 4]] }
weights:
  W: { shape: [2, 2], init: ones }
body:
  - a: x @ W.T
  - n: { use: Norm, inputs: { x: a }, outputs: { out: h }, steps: { rms: { append: 'from $\\of{a}$' }, scale: { title: new } } }
  - b: h + 1
  - c: b * 2
steps:
  - { id: last, show: c, title: last }
`, files);
    const forward = result.steps.forward;
    expect(forward.map((s) => s.title)).toEqual(['a', 'rms of $\\mathbf{a}$', 'new', 'b', 'last']);
    expect((forward[0].show as Node[]).map((n) => n.name)).toEqual(['W', 'W.T', 'a']);
    expect(forward[1].text).toBe('from $\\mathbf{a}$');
    expect((forward[1].show as Node[]).map((n) => n.name)).toEqual(['n.ms/1', 'n.ms']);
    expect(forward[2].text).toBe('out = \\begin{pmatrix} {{n.out[0, 0]:2}} & {{n.out[0, 1]:2}} \\\\ {{n.out[1, 0]:2}} & {{n.out[1, 1]:2}} \\end{pmatrix}');
  });

  it('places a step after the first node it shows, even inside a loop', async () => {
    const result = await page(`
title: t
dims: { H: 2 }
inputs:
  x: { value: [1, 2] }
body:
  - for: h
    in: 1..H
    body:
      - p[h]: x * h
      - q[h]: p[h] + 1
steps:
  - { id: split, show: 'p[1..H]', title: split }
`);
    expect(result.steps.forward.map((s) => s.title)).toEqual(['split', 'q[1]']);
    expect((result.steps.forward[1].show as Node[]).map((n) => n.name)).toEqual(['q[1]', 'q[2]']);
  });

  it('merges the steps that the iterations of a loop share, including the steps of instances', async () => {
    const result = await page(`
title: t
import: { Norm: ./norm.yaml }
dims: { H: 3 }
inputs:
  x: { value: [[1, 2]] }
body:
  - for: h
    in: 1..H
    body:
      - p[h]: x * h
      - n[h]:
          use: Norm
          inputs: { x: 'p[h]' }
          steps: { rms: { append: 'iteration \\of{h}' } }
      - g[h]:
          body:
            - r[h]: n[h].out + 1
          steps:
            - { id: plus, show: 'r[h]', active: 'n[h].out', title: 'plus \\of{h}' }
  - y: 'cat(r[1..H], dim=1)'
`, { 'mem:/pages/norm.yaml': NORM });
    const forward = result.steps.forward;
    expect(forward.map((s) => s.title)).toEqual(['p[1]', 'rms of $\\mathbf{p}^{(1)}$', 'scale', 'plus 1', 'y']);
    const shown = (i: number) => (forward[i].show as Node[]).map((n) => n.name);
    expect(shown(0)).toEqual(['p[1]', 'p[2]', 'p[3]']);
    expect(shown(1)).toEqual(['n[1].ms/1', 'n[1].ms', 'n[2].ms/1', 'n[2].ms', 'n[3].ms/1', 'n[3].ms']);
    expect(forward[1].text).toBe('iteration 1');
    expect(shown(3)).toEqual(['r[1]', 'r[2]', 'r[3]']);
    expect((forward[3].active as Node[]).map((n) => n.name)).toEqual(['n[1].out', 'n[2].out', 'n[3].out']);
  });

  it('applies a patch for a later iteration to the merged step', async () => {
    const result = await page(`
title: t
import: { Norm: ./norm.yaml }
dims: { H: 2 }
modules:
  Two:
    params: { H: { type: dim } }
    inputs: { x: { shape: [T, n] } }
    outputs: [o]
    body:
      - for: h
        in: 1..H
        body:
          - n[h]: { use: Norm, inputs: { x: x } }
      - o: n[1].out + n[2].out
inputs:
  x: { value: [[1, 2]] }
body:
  - two: { use: Two, params: { H: H }, inputs: { x: x }, steps: { 'n[2].scale': { title: second } } }
`, { 'mem:/pages/norm.yaml': NORM });
    expect(result.steps.forward.map((s) => s.title)).toEqual(['rms of $\\mathbf{x}$', 'second', 'o']);
  });

  it('rejects patches for steps that do not exist', async () => {
    await expect(page(`
title: t
import: { Norm: ./norm.yaml }
dims: {}
inputs:
  x: { value: [[1, 2]] }
body:
  - n: { use: Norm, inputs: { x: x }, steps: { nope: { title: x } } }
`, files)).rejects.toThrow(/page\.yaml:8:\d+ body\[0\]\.n\.steps\.nope: step 'nope' does not exist in 'n'/);
  });

  it('rewrites template names to graph names and attaches backward text', async () => {
    const result = await page(`
title: t
import: { Norm: ./norm.yaml }
dims: {}
vocab: [a, b, c]
inputs:
  ids: { tokens: [a, c] }
weights:
  E: { shape: [3, 2], init: ones }
body:
  - e: 'embedding(ids, E)'
  - n: { use: Norm, inputs: { x: e }, outputs: { out: h } }
  - l: mean(h)
steps:
  - { id: s, show: l, title: s, text: '{{n.ms[0]:3}} {{ h[b, 1] }} {{ids[b]:token}} {{ h.grad[0, 0] }} {{l}}' }
backward:
  h:
    title: back to h
    text: '{{ h.grad | matrix:2 }}'
panels:
  - loss: { sgd: { params: all, lr: 0.1 }, text: '{{ids[b]:token}}' }
  - { tokens: ids, columns: [h, { node: e, grad: true }] }
  - { matrix: E.T, highlight: ids }
`, files);
    expect(result.steps.forward.find((s) => s.title === 's')!.text).toBe('{{n.ms[0]:3}} {{ n.out[b, 1] }} {{ids[b]:token}} {{ n.out.grad[0, 0] }} {{l}}');
    expect(result.steps.backward).toBe('reverse');
    const h = result.nodes.get('h')!;
    const entry = result.steps.forward.find((s) => (Array.isArray(s.show) ? s.show : [s.show]).includes(h))!.backward!;
    expect(entry.title).toBe('back to h');
    expect(entry.text).toContain('{{n.out.grad[1, 1]:2}}');
    const [loss, tokens, matrix] = result.panels as [{ loss: { sgd: { params: unknown[] } } }, { tokens: unknown; columns: unknown[] }, { matrix: unknown; highlight: unknown }];
    expect(loss.loss.sgd.params).toEqual([result.nodes.get('E'), result.nodes.get('n.W')]);
    expect(tokens.tokens).toBe(result.nodes.get('ids'));
    expect(tokens.columns).toEqual([result.nodes.get('h'), { node: result.nodes.get('e'), grad: true }]);
    expect(matrix.matrix).toBe(result.nodes.get('E')!.T);
  });
});

describe('page structure', () => {
  it('builds the layout tree, modules and real dimensions', async () => {
    const files = { 'mem:/pages/norm.yaml': NORM };
    const result = await page(`
title: t
import: { Norm: ./norm.yaml }
dims: { T: 2, d: 2, H: 2 }
real: { name: Big, d: 64 }
inputs: {}
weights:
  x: { shape: [T, d], init: ones }
  Wq: { shape: ['H*d', d], init: ones }
body:
  - q: x @ Wq.T
  - for: h
    in: 1..H
    brace: head $\\of{h}$
    body:
      - p[h]: narrow(q, 1, (h-1)*d, d)
      - n[h]: { use: Norm, inputs: { x: 'p[h]' } }
  - g:
      subtitle: grouped
      body:
        - s: 'cat(p[1..H], dim=1)'
`, files);
    expect(names(result.tree.children)).toEqual([
      'x',
      'Wq',
      'q',
      {
        parallel: [
          { title: 'head $1$', children: ['p[1]', { title: 'n', children: ['n[1].ms/1', 'n[1].ms', 'n[1].out/1', 'n[1].out/2', 'n[1].out/3', 'n[1].W', 'n[1].out'] }] },
          { title: 'head $2$', children: ['p[2]', { title: 'n', children: ['n[2].ms/1', 'n[2].ms', 'n[2].out/1', 'n[2].out/2', 'n[2].out/3', 'n[2].W', 'n[2].out'] }] },
        ],
      },
      { title: 'g', children: ['s'] },
    ]);
    expect(result.modules.map((m) => [m.title, m.subtitle, m.nodes.length])).toEqual([
      ['head $1$', '', 7],
      ['n', '', 6],
      ['head $2$', '', 7],
      ['n', '', 6],
      ['g', 'grouped', 1],
    ]);
    expect(result.modules.map((m) => [m.id, m.parent])).toEqual([['h[1]', null], ['n[1]', 'h[1]'], ['h[2]', null], ['n[2]', 'h[2]'], ['g', null]]);
    expect(result.modules[1].inputs.map((n) => n.name)).toEqual(['p[1]']);
    expect(result.modules[1].outputs.map((n) => n.name)).toEqual(['n[1].out']);
    expect(result.modules[1].params.map((n) => n.name)).toEqual(['n[1].W']);
    expect(result.nodes.get('Wq')!.realShape).toEqual([128, 64]);
    expect(result.nodes.get('n[1].W')!.realShape).toEqual([64]);
    expect(result.graph.dims['4:128']).toBe(4);
    expect(result.page.title).toBe('t');
  });

  it('compiles a JS object with a base URL and package modules', async () => {
    const spec: PageFile = {
      title: 'js',
      import: { RMSNorm: 'vizflow/nn/rmsnorm' },
      dims: { T: 1, d: 2 },
      inputs: { x: { value: [[3, 4]] } },
      body: [{ n: { use: 'RMSNorm', inputs: { x: 'x' }, outputs: { out: 'y' } } }],
    };
    const result = await compilePage(spec);
    const r = 1 / Math.sqrt(12.5 + 1e-6);
    expect([...result.nodes.get('y')!.value.data]).toEqual([3 * r, 4 * r]);
    expect(result.steps.forward.map((s) => s.title)).toEqual(['RMSNorm：逐元素平方', '均方根的倒数', '归一化', '乘上权重 γ']);
    await expect(compilePage({ ...spec, import: { A: './a.yaml' } })).rejects.toThrow(/cannot resolve '\.\/a\.yaml': the page has no base URL/);
  });
});

describe('beside', () => {
  const files = { 'mem:/pages/norm.yaml': NORM };
  const head = `
title: t
import: { Norm: ./norm.yaml }
dims: { T: 2, d: 2, H: 2 }
inputs:
  x: { value: [[1, 2], [3, 4]] }
  y: { value: [[5, 6], [7, 8]] }
body:
`;
  const lanes = (grpBeside: string, forBeside: string) => `${head}  - na: { use: Norm, inputs: { x: x }, outputs: { out: a } }
  - grp:
      ${grpBeside}
      body:
        - s: exp(y)
  - for: h
    in: 1..H
    name: heads
    ${forBeside}
    body:
      - p[h]: y * h
      - for: j
        in: 1..h
        body:
          - 'r[h][j]': p[h] * j
  - z: a + s
`;

  it('turns an item and the items beside it into one parallel entry with a lane each', async () => {
    const result = await page(lanes('beside: na', 'beside: grp'), files);
    const norm = ['na.ms/1', 'na.ms', 'na.out/1', 'na.out/2', 'na.out/3', 'na.W', 'na.out'];
    const heads = {
      children: [
        {
          parallel: [
            { children: ['p[1]', { parallel: [{ children: ['r[1][1]'] }] }] },
            { children: ['p[2]', { parallel: [{ children: ['r[2][1]'] }, { children: ['r[2][2]'] }] }] },
          ],
        },
      ],
    };
    expect(names(result.tree.children)).toEqual(['x', 'y', { parallel: [{ title: 'na', children: norm }, { title: 'grp', children: ['s'] }, heads] }, 'z']);
    const plain = await page(lanes('subtitle: grouped', 'brace: head'), files);
    expect(plain.tree.children.map((entry) => entry.kind)).toEqual(['node', 'node', 'module', 'module', 'parallel', 'node']);
    expect(result.steps.forward.map((s) => s.title)).toEqual(plain.steps.forward.map((s) => s.title));
    expect(result.graph.findLoss()).toBe(plain.graph.findLoss());
  });

  it('accepts any member of the group before it as the target', async () => {
    const result = await page(lanes('beside: na', 'beside: na'), files);
    expect(result.tree.children[2]).toMatchObject({ kind: 'parallel' });
    expect((result.tree.children[2] as { branches: unknown[] }).branches).toHaveLength(3);
  });

  it('rejects a target that is unknown or not right before the item', async () => {
    await expect(page(lanes('beside: nope', 'brace: head'), files)).rejects.toThrow(
      /page\.yaml:11:15 body\[1\]\.grp\.beside: beside 'nope': no instance, group or named for item 'nope' comes before this one in the same body/,
    );
    const gap = `${head}  - na: { use: Norm, inputs: { x: x } }
  - b: exp(x)
  - nb: { use: Norm, inputs: { x: y }, beside: na }
`;
    await expect(page(gap, files)).rejects.toThrow(/page\.yaml:11:\d+ body\[2\]\.nb\.beside: beside 'na': 'na' is not the item right before this one/);
    await expect(page(`${head}  - na: { use: Norm, inputs: { x: x } }\n  - b: { op: exp(x), beside: na }\n`, files)).rejects.toThrow(
      /body\[1\]\.b\.beside: beside applies to an instance, a group or a for item, not to a single node/,
    );
  });

  it('rejects an item that uses a result of an item in the same beside group', async () => {
    const uses = `${head}  - na: { use: Norm, inputs: { x: x }, outputs: { out: a } }
  - grp:
      body:
        - s: exp(y)
  - nb: { use: Norm, inputs: { x: a }, beside: grp }
`;
    await expect(page(uses.replace('      body:', '      beside: na\n      body:'), files)).rejects.toThrow(
      /page\.yaml:14:48 body\[2\]\.nb\.beside: 'nb' is beside 'grp' but uses 'na\.out', which 'na' in the same beside group computes; .* Put it after the beside group/,
    );
  });
});
