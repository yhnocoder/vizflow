// @vitest-environment jsdom
import katex from 'katex';
import { beforeAll, describe, expect, it } from 'vitest';
import { type Node, moduleJacobian } from '../../src/core';
import type { CompiledPage } from '../../src/dsl';
import { elemTexOf } from '../../src/explain/elem';
import { type Steps, createSteps } from '../../src/steps';
import { referenceModel } from '../fixtures/qwen3/reference.js';
import { loadExample } from './load';

interface RefTensor { v: Float64Array; g: Float64Array; R: number; C: number; leaf?: boolean }
interface RefOp { id: string; mod: string | null; ins: string[]; outs: string[] }
interface RefModule { parent: string | null; t: string; s: string; io: [string, string][]; fixed?: boolean }
interface Reference {
  TS: Record<string, RefTensor>;
  OPS: RefOp[];
  MODS: Record<string, RefModule>;
  ALIAS: Record<string, string>;
  st: { loss: number; J: Record<string, number[][][]> };
  P: { seq: number[] };
  CF: Record<string, () => [string, string]>;
  CB: Record<string, () => [string, string]>;
  OPL: Record<string, string>;
  OPLB: Record<string, string>;
  MODF: Record<string, [string, string]>;
}

const ATTN = 'layer.self_attn';
const ROPE = `${ATTN}.apply_rotary_pos_emb`;
const GQA = `${ATTN}.eager_attention_forward`;
const NORMS: Record<string, string> = { q: `${ATTN}.q_norm`, k: `${ATTN}.k_norm` };
const RMS: Record<string, string> = { '1': 'layer.input_layernorm', '2': 'layer.post_attention_layernorm', '': 'norm' };

const TENSORS: Record<string, string> = {
  E: 'E',
  G1: `${RMS['1']}.weight`,
  Wq: `${ATTN}.Wq.T`,
  Wk: `${ATTN}.Wk.T`,
  Wv: `${ATTN}.Wv.T`,
  Gq: `${NORMS.q}.weight`,
  Gk: `${NORMS.k}.weight`,
  Wo: `${ATTN}.Wo.T`,
  G2: `${RMS['2']}.weight`,
  Wg: 'layer.mlp.Wg.T',
  Wu: 'layer.mlp.Wu.T',
  Wd: 'layer.mlp.Wd.T',
  G: 'norm.weight',
  W: 'W.T',
  cos: `${ROPE}.cos`,
  sin: `${ROPE}.sin`,
  M: `${ATTN}.M`,
  e: 'e',
  x0: 'x0',
  ...Object.fromEntries(Object.entries(RMS).flatMap(([i, path]) => [
    [`sq${i}`, `${path}.sq`],
    [`ms${i}`, `${path}.ms`],
    [`r${i}`, `${path}.r`],
  ])),
  ah1: `${RMS['1']}.xh`,
  n1: `${RMS['1']}.out`,
  ...Object.fromEntries(['q', 'k', 'v'].map((k) => [k, `${ATTN}.${k}`])),
  ...Object.fromEntries(Object.entries(NORMS).flatMap(([x, path]) => [
    [`${x}sq`, `${path}.sq`],
    [`${x}ms`, `${path}.ms`],
    [`${x}rr`, `${path}.r`],
    [`${x}ah`, `${path}.xh`],
    [`${x}n`, `${path}.out`],
  ])),
  ...Object.fromEntries(['qc', 'kc', 'qrh', 'krh', 'qs', 'ks'].map((k) => [k, `${ROPE}.${k}`])),
  qr: `${ROPE}.q_embed`,
  kr: `${ROPE}.k_embed`,
  ...Object.fromEntries([0, 1].flatMap((h) => [
    [`kp${h}`, `${GQA}.kp[${h + 1}]`],
    [`V${h}`, `${GQA}.V[${h + 1}]`],
    [`kT${h}`, `${GQA}.kT[${h + 1}]`],
    [`S${h}`, `${GQA}.S[${h + 1}]`],
    [`Ss${h}`, `${GQA}.Ss[${h + 1}]`],
    [`Sm${h}`, `${GQA}.Sm[${h + 1}]`],
    [`Se${h}`, `${GQA}.ex[${h + 1}]`],
    [`Z${h}`, `${GQA}.Z[${h + 1}]`],
    [`P${h}`, `${GQA}.P[${h + 1}]`],
    [`O${h}`, `${GQA}.o[${h + 1}]`],
  ])),
  O: `${GQA}.o`,
  ao: `${ATTN}.att`,
  a: 'layer.a',
  ah2: `${RMS['2']}.xh`,
  n: `${RMS['2']}.out`,
  gate: 'layer.mlp.gate',
  up: 'layer.mlp.up',
  s: 'layer.mlp.a',
  m: 'layer.mlp.m',
  down: 'layer.mlp.down',
  x: 'layer.x_out',
  xh: 'norm.xh',
  h: 'norm.out',
  z: 'z',
  L: 'LSE',
  lp: 'logp',
  lt: 'lt',
  loss: 'loss',
};

const MODULES: Record<string, string> = {
  pre: 'layers',
  dec: 'layer',
  rms1: RMS['1'],
  attn: ATTN,
  qkn: `${ATTN}.qk_norm`,
  rope: ROPE,
  rms2: RMS['2'],
  mlp: 'layer.mlp',
  rms: 'norm',
};

const BUFFERS = new Set(['cos', 'sin', 'M']);
const MATH_UNICODE = /[\u0370-\u03FF\u2070-\u209F\u2190-\u22FF\u0300-\u036F\u2032\u00B7\u00D7\u00F7\u2113\u1D00-\u1DBF\u2212\u00B2\u00B3\u00B9]/;
const TITLE_CHANGES: Record<string, string> = { b_r1n: '传到γ1和x0', b_r2n: '传到γ2和a', b_fh: '传到γ和x' };
const REF_BROKEN = new Set(['b_r1sq', 'b_r2sq', 'b_fsq']);
const LABEL_CHANGES: Record<string, string> = {
  qkrs: '$\\mathrm{ms}_{q},\\ r_{q},\\ \\mathrm{ms}_{k},\\ r_{k}$',
  b_qkrs: '$\\partial\\mathrm{ms}_{q},\\ \\partial\\mathbf{q}^{2},\\ \\partial\\mathrm{ms}_{k},\\ \\partial\\mathbf{k}^{2}$',
  b_qkah: '$\\partial r_{q},\\ \\partial\\mathbf{q},\\ \\partial r_{k},\\ \\partial\\mathbf{k}$',
  b_qkn: '$\\partial\\boldsymbol{\\gamma}_{q},\\ \\partial\\hat{\\mathbf{q}},\\ \\partial\\boldsymbol{\\gamma}_{k},\\ \\partial\\hat{\\mathbf{k}}$',
};

const GLAB: Record<string, string> = {
  ov: '总览', in: '输入', emb: 'embed_tokens', pre: '⋯ 第 1 到 N−1 层', dec: 'Qwen3DecoderLayer', rms1: 'input_layernorm', attn: 'self_attn',
  res1: '残差', rms2: 'post_attention_layernorm', mlp: 'mlp', res2: '残差', rms: 'model.norm', lm: 'lm_head', ls: 'log_softmax', nll: 'nll_loss',
};
const TOP_OPS: Record<string, string> = { emb: 'emb', lm: 'lm', lse: 'ls', lp: 'ls', nll: 'nll', mean: 'nll' };

interface RefStep { id: string; d: 'f' | 'b'; fm?: string; mod: string | null }

function refSteps(ref: Reference): RefStep[] {
  const label = (m: string | null) => (m === null ? null : ref.MODS[m].t);
  const steps: RefStep[] = [{ id: 'f0', d: 'f', mod: null }, { id: 'in', d: 'f', mod: null }];
  let last = false;
  for (const o of ref.OPS) {
    if (o.mod === 'pre') {
      if (!last) steps.push({ id: 'F_pre', d: 'f', fm: 'pre', mod: label('pre') });
      last = true;
      continue;
    }
    last = false;
    steps.push({ id: o.id, d: 'f', mod: label(o.mod) });
  }
  for (let i = ref.OPS.length - 1; i >= 0; i--) {
    const o = ref.OPS[i];
    steps.push(o.mod === 'pre' ? { id: 'B_pre', d: 'b', fm: 'pre', mod: label('pre') } : { id: `b_${o.id}`, d: 'b', mod: label(o.mod) });
  }
  steps.push({ id: 'ball', d: 'b', mod: null });
  return steps;
}

function groupOf(ref: Reference, s: RefStep): string {
  if (s.id === 'f0' || s.id === 'ball') return 'ov';
  if (s.id === 'in') return 'in';
  const oid = s.fm ? null : s.d === 'f' ? s.id : s.id.slice(2);
  const o = oid === null ? undefined : ref.OPS.find((x) => x.id === oid);
  let m = s.fm ?? o?.mod ?? null;
  if (m === null) return TOP_OPS[oid as string] ?? 'ov';
  if (m === 'pre') return 'pre';
  if (m === 'dec') return o ? o.id : 'dec';
  while (ref.MODS[m].parent && ref.MODS[m].parent !== 'dec') m = ref.MODS[m].parent as string;
  return m;
}

function mathText(tex: string): string {
  let mathml: string;
  try {
    mathml = katex.renderToString(tex, { output: 'mathml', throwOnError: true });
  } catch {
    mathml = katex.renderToString(tex.replace(/_([0-9A-Za-z])_/g, '_$1'), { output: 'mathml', throwOnError: true });
  }
  const text = mathml.replace(/<annotation[\s\S]*?<\/annotation>/g, '').replace(/<[^>]+>/g, '');
  return text.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#x27;/g, "'");
}

function flatHeads(text: string): string {
  return text.replace(/\\left\[\\,([\s\S]*?)\\,\\right\]/g, (_m, inner: string) => {
    const blocks = [...inner.matchAll(/\\begin\{pmatrix\}([\s\S]*?)\\end\{pmatrix\}_\{h=\d+\}/g)].map((b) => b[1].split('\\\\'));
    return `\\begin{pmatrix}${blocks[0].map((_, r) => blocks.map((rows) => rows[r]).join('&')).join('\\\\')}\\end{pmatrix}`;
  });
}

function visible(text: string): string {
  const rendered = text
    .replace(/\$\$([\s\S]+?)\$\$/g, (_m, tex: string) => mathText(tex))
    .replace(/\$([^$]+?)\$/g, (_m, tex: string) => mathText(tex))
    .replace(/<[^>]+>/g, '');
  return rendered
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .replace(/[⊤]/g, 'T')
    .replace(/[⋅]/g, '·')
    .replace(/-/g, '−')
    .replace(/[\s{}^_ˉ‾¯~˜√\u2061-\u2063]/g, '');
}

function refLabel(ref: Reference, s: RefStep): string {
  if (s.id === 'f0' || s.id === 'ball') return '总览';
  if (s.id === 'in') return '输入';
  if (s.fm !== undefined) return ref.MODF[s.fm][s.d === 'f' ? 0 : 1];
  return s.d === 'f' ? ref.OPL[s.id] : ref.OPLB[s.id.slice(2)];
}

function node(page: CompiledPage, name: string): Node {
  const found = page.nodes.get(name) ?? page.graph.node(name);
  if (found === undefined) throw new Error(`no node '${name}'`);
  return found;
}

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, what: string): void {
  expect(actual.length, what).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const want = expected[i];
    if (want === -Infinity) expect(actual[i], `${what}[${i}]`).toBe(-Infinity);
    else expect(Math.abs(actual[i] - want), `${what}[${i}]: ${actual[i]} vs ${want}`).toBeLessThan(1e-12);
  }
}

describe('qwen3-dsl.yaml against the reference page', () => {
  let page: CompiledPage;
  let steps: Steps;
  let ref: Reference;

  beforeAll(async () => {
    page = await loadExample('qwen3-dsl.yaml');
    steps = createSteps(page.graph, page.steps);
    ref = referenceModel() as Reference;
  });

  it('gives every attention node the Qwen3-8B shape, counting all 32 query heads and 8 kv heads', () => {
    const real = (name: string) => node(page, name).realShape;
    expect(real(`${ATTN}.q/1`)).toEqual([3, 4096]);
    expect(real(`${ATTN}.q`)).toEqual([3, 32, 128]);
    expect(real(`${ATTN}.k`)).toEqual([3, 8, 128]);
    expect(real(`${NORMS.q}.sq`)).toEqual([3, 32, 128]);
    expect(real(`${NORMS.k}.sq`)).toEqual([3, 8, 128]);
    expect(real(`${NORMS.q}.r`)).toEqual([3, 32, 1]);
    expect(real(`${ROPE}.cos`)).toEqual([3, 1, 128]);
    expect(real(`${GQA}.kT[1]`)).toEqual([128, 3]);
    expect(real(`${GQA}.S[1]`)).toEqual([3, 3]);
    expect(real(`${GQA}.P[2]`)).toEqual([3, 3]);
    expect(real(`${GQA}.o[1]`)).toEqual([3, 128]);
    expect(real(`${GQA}.o`)).toEqual([3, 4096]);
    expect(real(`${ATTN}.att`)).toEqual([3, 4096]);
  });

  it('computes the reference loss with theta = 10', () => {
    const loss = page.graph.findLoss()!.value.data[0];
    expect(loss.toFixed(3)).toBe('2.777');
    expect(Math.abs(loss - ref.st.loss)).toBeLessThan(1e-12);
  });

  it('matches every reference tensor value and gradient', () => {
    const covered = Object.keys(ref.TS).filter((k) => !(k in ref.ALIAS));
    expect(covered.sort()).toEqual(Object.keys(TENSORS).sort());
    for (const [key, name] of Object.entries(TENSORS)) {
      const t = ref.TS[key];
      const n = node(page, name);
      expectClose(n.value.data, t.v, `${key} -> ${name} value`);
      if (BUFFERS.has(key)) continue;
      expectClose(n.grad!.data, t.g, `${key} -> ${name} grad`);
    }
  });

  it('has one node per reference op output, op by op', () => {
    const ops = page.graph.nodes.filter((n) => n.kind === 'op' && !n.name.endsWith('.T') && !n.name.endsWith('/1'));
    const outs = ref.OPS.flatMap((o) => o.outs).map((k) => TENSORS[k]);
    expect(ops.map((n) => n.name).sort()).toEqual([...new Set(outs)].sort());
  });

  it('lists 48 forward and 47 backward steps with the reference labels, titles and texts', () => {
    const expected = refSteps(ref);
    expect(steps.list.filter((s) => s.dir === 'forward')).toHaveLength(48);
    expect(steps.list.filter((s) => s.dir === 'backward')).toHaveLength(47);
    expect(steps.list).toHaveLength(expected.length);
    steps.list.forEach((info, i) => {
      const want = expected[i];
      expect(info.dir[0], `step ${i}`).toBe(want.d);
      const { title, body } = steps.text(i);
      if (REF_BROKEN.has(want.id)) {
        expect(title).toBe('传到输入');
        return;
      }
      const [refTitle, refBody] = (want.d === 'f' ? ref.CF : ref.CB)[want.id]();
      expect(visible(title), `step ${i} ${want.id}`).toBe(TITLE_CHANGES[want.id] ?? visible(refTitle));
      expect(visible(flatHeads(body)), `step ${i} ${want.id}`).toBe(visible(refBody));
    });
  });

  it('labels steps so that they read like the reference OPL, OPLB and MODF tables', () => {
    const expected = refSteps(ref);
    steps.list.forEach((info, i) => expect(visible(info.label), `step ${i} ${expected[i].id}: ${info.label}`).toBe(visible(LABEL_CHANGES[expected[i].id] ?? refLabel(ref, expected[i]))));
    expect(steps.list.map((s) => s.label).slice(2, 8)).toEqual(['$\\mathbf{e}=E[x]$', '$\\mathbf{x}_0$', '$\\mathbf{x}_0^{2}$', '$\\mathrm{ms}_{1},\\ r_{1}$', '$\\hat{\\mathbf{x}}_{0}$', '$\\mathbf{n}_{1}$']);
    expect(steps.list.map((s) => s.kind).filter((k) => k !== 'op')).toEqual(['overview', 'inputs', 'module', 'module', 'overview']);
  });

  it('groups steps into the reference segments and sub-modules', () => {
    const expected = refSteps(ref);
    steps.list.forEach((info, i) => {
      const want = expected[i];
      const segment = GLAB[groupOf(ref, want)];
      const sub = want.mod !== null && want.mod !== segment && want.mod !== 'Qwen3DecoderLayer' ? want.mod : undefined;
      expect([visible(info.segment), info.sub === undefined ? undefined : visible(info.sub)], `step ${i} ${want.id}`).toEqual([visible(segment), sub === undefined ? undefined : visible(sub)]);
    });
  });

  it('describes the module tree like the reference MODS', () => {
    const byId = new Map(page.modules.map((m) => [m.id, m]));
    expect(page.modules.map((m) => m.id).sort()).toEqual(Object.values(MODULES).sort());
    for (const [key, mod] of Object.entries(ref.MODS)) {
      const m = byId.get(MODULES[key])!;
      expect(visible(m.title), key).toBe(visible(mod.t));
      expect(m.parent, key).toBe(mod.parent === null ? null : MODULES[mod.parent]);
      expect(m.fixed, key).toBe(mod.fixed === true);
      expect(m.folded, key).toBe(key === 'pre');
      const alias = (k: string) => ref.ALIAS[k] ?? k;
      expect(m.inputs.map((n) => n.name), key).toEqual(mod.io.map(([i]) => TENSORS[alias(i)]));
      expect(m.outputs.map((n) => n.name), key).toEqual(mod.io.map(([, o]) => TENSORS[alias(o)]));
    }
    expect(byId.get(ATTN)!.subtitle).toBe('Qwen3Attention');
    expect(byId.get('layer')!.subtitle).toBe('第 N 层');
    expect(byId.get('layer.mlp')!.subtitle).toBe('Qwen3MLP');
  });

  it('writes labels, titles, segments and module names in LaTeX rather than unicode math', () => {
    const s = createSteps(page.graph, page.steps);
    const texts = (list: readonly { label: string; segment: string; sub?: string }[]) => list.flatMap((x, i) => [x.label, x.segment, x.sub ?? '', s.text(i).title]);
    const all = [...texts(s.list), ...page.modules.flatMap((m) => [m.title, m.subtitle])];
    s.setAllFolded(true);
    all.push(...texts(s.list));
    for (const text of all) expect(text, text).not.toMatch(MATH_UNICODE);
    s.dispose();
  });

  it('renders every step text, label and node TeX', () => {
    const renders = (tex: string, what: string) => expect(() => katex.renderToString(tex, { throwOnError: true }), `${what}: ${tex}`).not.toThrow();
    const math = (text: string) => [...text.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)].map((m) => m[1] ?? m[2]);
    steps.list.forEach((info, i) => {
      const { title, body } = steps.text(i);
      for (const tex of [...math(title), ...math(body), ...math(info.label)]) renders(tex, `step ${i}`);
    });
    for (const n of page.graph.nodes) {
      renders(n.tex, n.name);
      renders(elemTexOf(n, n.shape.map(() => 0)), `${n.name} elem`);
    }
  });

  it('reads and replaces the token sequence', async () => {
    const fresh = await loadExample('qwen3-dsl.yaml');
    const seq = fresh.sequence!;
    expect(seq.tokens).toEqual(['on', 'mat', 'the', 'sat', 'cat']);
    expect(seq.length).toBe(4);
    expect(seq.get()).toEqual(ref.P.seq);
    let notified = 0;
    fresh.graph.subscribe(() => notified++);
    const next = [4, 1, 0, 3];
    seq.set(next);
    expect(notified).toBeGreaterThan(0);
    expect(seq.get()).toEqual(next);
    expect(Array.from(node(fresh, 'ids').value.data)).toEqual([4, 1, 0]);
    expect(Array.from(node(fresh, 'y').value.data)).toEqual([1, 0, 3]);
    const other = referenceModel(next) as Reference;
    expect(Math.abs(fresh.graph.findLoss()!.value.data[0] - other.st.loss)).toBeLessThan(1e-12);
    expectClose(node(fresh, 'E').grad!.data, other.TS.E.g, 'E grad after set');
    expect(() => seq.set([0, 1, 2])).toThrow(/expected 4 token ids/);
    expect(() => seq.set([0, 1, 2, 5])).toThrow(/between 0 and 4/);
  });
  it('folds a module into one forward and one backward step with the reference texts', async () => {
    for (const [key, id] of Object.entries(MODULES)) {
      if (key === 'pre') continue;
      const s = createSteps(page.graph, page.steps);
      expect(s.folded(id), id).toBe(false);
      s.setFolded(id, true);
      expect(s.folded(id), id).toBe(true);
      const own = s.list.filter((x) => x.kind === 'module' && x.module === id);
      expect(own.map((x) => [x.dir, visible(x.label)]), id).toEqual([['forward', visible(ref.MODF[key][0])], ['backward', visible(ref.MODF[key][1])]]);
      const inside = page.modules.find((m) => m.id === id)!.nodes;
      expect(s.list.some((x) => x.dir === 'forward' && x.kind === 'op' && x.nodes.some((n) => inside.includes(n) && n.kind === 'op')), id).toBe(false);
      for (const info of own) {
        const i = s.list.indexOf(info);
        const [title, body] = (info.dir === 'forward' ? ref.CF[`F_${key}`] : ref.CB[`B_${key}`])();
        expect(visible(s.text(i).title), `${id} ${info.dir}`).toBe(visible(title));
        expect(visible(flatHeads(s.text(i).body)), `${id} ${info.dir}`).toBe(visible(body));
      }
      s.dispose();
    }
  });

  it('keeps fixed modules folded and moves the current step to the folded one', () => {
    const s = createSteps(page.graph, page.steps);
    expect(s.folded('layers')).toBe(true);
    s.setFolded('layers', false);
    expect(s.folded('layers')).toBe(true);
    expect(() => s.setFolded('nope', true)).toThrow();
    const qkv = s.list.findIndex((x) => x.kind === 'op' && x.nodes.some((n) => n.name === `${ATTN}.q`));
    s.go(qkv);
    let notified = 0;
    s.subscribe(() => notified++);
    s.setFolded(ATTN, true);
    expect(notified).toBeGreaterThan(0);
    expect(s.list.filter((x) => x.dir === 'forward')).toHaveLength(48 - 21 + 1);
    expect(visible(s.list[s.current].label)).toBe(visible('att = self_attn(n₁)'));
    s.setAllFolded(true);
    expect(s.list.map((x) => visible(x.label))).toEqual([
      '总览', '输入', 'e = E[x]', 'x₀', 'x = layer(x₀)', 'h', 'z', 'LSE', 'log p', 'ℓₜ', 'ℓ',
      '∂ℓₜ', '∂log p', '∂z, ∂LSE', '∂z', '∂W, ∂h', '∂γ, ∂x', '∂参数, ∂x₀', '∂e', '∂E', '总览',
    ].map(visible));
    expect(visible(s.list[s.current].label)).toBe(visible('x = layer(x₀)'));
    s.setAllFolded(false);
    expect(s.list).toHaveLength(95);
    expect(s.folded('layers')).toBe(true);
    s.dispose();
  });

  it('computes module Jacobians like the reference', () => {
    for (const [key, mod] of Object.entries(ref.MODS)) {
      const m = page.modules.find((x) => x.id === MODULES[key])!;
      mod.io.forEach((_, q) => {
        const J = moduleJacobian(page.graph, m, m.inputs[q], m.outputs[q]);
        const want = ref.st.J[key][q];
        expect(J.length, key).toBe(want.length);
        J.forEach((row, i) => expectClose(row, want[i], `${key} J[${i}]`));
      });
    }
  });
});
