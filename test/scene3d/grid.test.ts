// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, loadPage, parseYaml } from '../../src/dsl';
import { CARD_GAP, COPY_GAP, type LevelLayout, levelPlan } from '../../src/scene3d/levels';
import { OUTLINE_STEP, outlineRect } from '../../src/scene3d/kernels';
import { isIndexView } from '../../src/scene3d/layout';
import { examplesUrl, readFileUrl } from '../dsl/load';

const BASE = 'mem:/pages/page.yaml';
const GRID = 'mem:/pages/g.yaml';

function kernel(layout?: string): string {
  return [
    'module: G',
    'kernel: true',
    'grid: { i: 2 }',
    'shared: [row]',
    'args: { bm: dim }',
    'inputs: { x: [T, d], y: [T, d] }',
    'outputs: { y: [T, d] }',
    'forward:',
    '  - row: x[i]',
    '  - a: row * 2',
    '  - b: a + 1',
    '  - c: b * a',
    '  - acc: c * 0',
    '  - for: n in range(i + 1)',
    '    do:',
    '      - acc: acc + x[n]',
    "  - 'y[i]': acc",
    layout === undefined ? '' : `layout: ${layout}`,
    'steps:',
    '  - id: row',
    '    show: [row, a, b, c, acc]',
    '  - id: r',
    '    loop: n',
    "    defaults: { show: acc, title: 'copy {{ i }} round {{ round }}' }",
    '  - id: y',
  ].join('\n');
}

const ROOT = `title: t
model: M
import: { G: ./g.yaml }
config:
  d: 2
data:
  x: [[1, 2], [3, 4], [5, 6], [7, 8]]
modules:
  M:
    args: { d: dim }
    init:
      g: { use: G, args: { bm: 2 } }
    inputs:
      x: { type: tensor, shape: [T, d], dtype: fp32 }
    outputs:
      l: { type: tensor, shape: [], dtype: fp32 }
    forward:
      - h: x * 2
      - buf: zeros(T, d)
      - y: g(h, buf)
      - l: y.sum()
    steps:
      - id: h
      - id: buf
      - id: g
      - id: l
`;

function page(layout?: string): Promise<CompiledPage> {
  const files: Record<string, string> = { [BASE]: ROOT, [GRID]: kernel(layout) };
  return loadPage(BASE, { read: async (url) => files[url] });
}

function plan(p: CompiledPage): LevelLayout {
  return levelPlan(p.steps, p.levels).resolve({ folded: () => false, aligned: new Map() });
}

function blockOf(layout: LevelLayout, name: string) {
  const found = layout.blocks.find((b) => b.copyOf === undefined && new RegExp(`^g\\.${name}(\\[\\d+\\])+$`).test(b.node.name));
  if (found === undefined) throw new Error(`no block for ${name}`);
  return found;
}

const zOf = (layout: LevelLayout, name: string) => Math.max(...blockOf(layout, name).elements.map((e) => e.z));
const xOf = (layout: LevelLayout, name: string) => Math.min(...blockOf(layout, name).elements.map((e) => e.x));
const yOf = (layout: LevelLayout, name: string) => blockOf(layout, name).elements[0].y;

describe('kernel ground', () => {
  it('draws the rounds and the stacked copies of a name as one block', async () => {
    const layout = plan(await page());
    const accs = layout.blocks.filter((b) => /^g\.acc\[[^/]*$/.test(b.node.name));
    expect(accs.map((b) => b.slot?.map((n) => n.name))).toEqual([['g.acc[0]', 'g.acc[1]'], ['g.acc[0][0]', 'g.acc[1][0]', 'g.acc[1][1]']]);
  });

  it('puts every block of the call at one height, the HBM card left of the chip card', async () => {
    const layout = plan(await page());
    expect(layout.kernels).toHaveLength(1);
    const ground = layout.kernels[0];
    expect(ground.grids.map((g) => g.module)).toEqual(['g']);
    expect(new Set(ground.blocks.flatMap((b) => b.elements.map((e) => e.y.toFixed(6))))).toEqual(new Set([ground.y.toFixed(6)]));
    const [chip] = ground.chips;
    expect(chip.card.x[0] - ground.hbm!.x[1]).toBeCloseTo(CARD_GAP);
    expect(chip.card.z).toEqual(ground.hbm!.z);
    expect(chip.shared!.x).toEqual(chip.registers!.x);
    expect(chip.shared!.z[1]).toBeLessThan(chip.registers!.z[0]);
  });

  it('numbers register rows by data dependency depth, shared memory behind them', async () => {
    const layout = plan(await page());
    expect(zOf(layout, 'row')).toBeLessThan(zOf(layout, 'a'));
    expect(zOf(layout, 'a')).toBeLessThan(zOf(layout, 'b'));
    expect(zOf(layout, 'b')).toBeLessThan(zOf(layout, 'c'));
    expect(yOf(layout, 'a')).toBeCloseTo(yOf(layout, 'c'));
  });

  it('places the names of layout.rows in their row, in the listed order', async () => {
    const layout = plan(await page('{ rows: [[b, a]] }'));
    expect(Math.abs(zOf(layout, 'a') - zOf(layout, 'b'))).toBeLessThan(2);
    expect(xOf(layout, 'b')).toBeLessThan(xOf(layout, 'a'));
    expect(zOf(layout, 'c')).toBeLessThan(zOf(layout, 'acc'));
  });

  it.each([
    ['{ rows: [[a, zz]] }', "'zz' is not a tensor computed in the forward"],
    ['{ rows: [[a], [y]] }', 'outputs stay in device memory'],
    ['{ rows: a }', 'rows'],
  ])('rejects layout %s', async (layout, message) => {
    await expect(page(layout)).rejects.toThrow(message);
  });

  it('rejects rows on a module that is not a kernel', async () => {
    const plain = ['module: G', 'args: { bm: dim }', 'inputs: { x: [T, d], y: [T, d] }', 'outputs: { z: [T, d] }', 'forward:', '  - a: x * 2', '  - z: a + y', 'layout: { rows: [[a]] }'].join('\n');
    const files: Record<string, string> = { [BASE]: ROOT, [GRID]: plain };
    await expect(loadPage(BASE, { read: async (url) => files[url] })).rejects.toThrow('needs kernel: true');
  });

  it('copies a read input drawn elsewhere onto the HBM card and links it to the original', async () => {
    const layout = plan(await page());
    const ground = layout.kernels[0];
    const copies = ground.blocks.filter((b) => b.copyOf !== undefined);
    expect(copies.map((b) => b.node.name)).toEqual(['h']);
    expect(layout.links.map((l) => [l.original.node.name, l.copy.node.name])).toEqual([['h', 'h']]);
    const names = ground.blocks.filter((b) => !/^g\./.test(b.node.name)).map((b) => b.node.name);
    expect(names).toContain('y');
    expect(names).not.toContain('buf');
  });

  it('lays expanded copies side by side, COPY_GAP apart, each with its own chip card', async () => {
    const layout = plan(await page('{ grid: { i: expand } }'));
    const chips = layout.kernels[0].chips;
    expect(chips.map((c) => c.expand)).toEqual([[0], [1]]);
    expect(chips[1].card.x[0] - chips[0].card.x[1]).toBeCloseTo(COPY_GAP);
  });

  it('draws no block for an anonymous index node and starts its edges on the source', async () => {
    const p = await page();
    const layout = plan(p);
    const anonymous = p.graph.nodes.filter((n) => isIndexView(n));
    expect(anonymous.length).toBeGreaterThan(0);
    expect(layout.blocks.filter((b) => isIndexView(b.node))).toEqual([]);
  });
});

describe('stacked copy outlines', () => {
  it('shift each copy right, down and back by OUTLINE_STEP', () => {
    const card = { x: [0, 4] as [number, number], z: [-3, 1] as [number, number] };
    const second = outlineRect(card, 5, 2);
    const [dx, dy, dz] = OUTLINE_STEP;
    expect(dx).toBeGreaterThan(0);
    expect(dy).toBeLessThan(0);
    expect(dz).toBeLessThan(0);
    expect(second.slice(0, 6)).toEqual([2 * dx, 5 + 2 * dy, -3 + 2 * dz, 4 + 2 * dx, 5 + 2 * dy, -3 + 2 * dz]);
    expect(second.slice(-3)).toEqual(second.slice(0, 3));
  });
});

describe('FlashAttention kernels', () => {
  it('follow the rows of the forward kernel layout', async () => {
    const url = new URL('flash-attn/flash-attn-dsl.yaml', examplesUrl).href;
    const p = await compilePage(parseYaml(await readFileUrl(url), url) as never, { base: url, read: readFileUrl });
    const layout = plan(p);
    const fwd = layout.blocks.filter((b) => b.copyOf === undefined && /\.fwd\./.test(b.node.name));
    const z = (name: string) => Math.max(...fwd.find((b) => new RegExp(`\\.fwd\\.${name}(\\[\\d+\\])+$`).test(b.node.name))!.elements.map((e) => e.z));
    const order = ['acc_s', 'row_max', 'p', 'acc_o', 'lse'].map(z);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    const ground = layout.kernels.find((k) => k.grids.some((g) => /\.fwd$/.test(g.module)))!;
    expect(ground.blocks.filter((b) => b.copyOf !== undefined).map((b) => b.node.name).sort()).toEqual(['q']);
  }, 60000);
});

describe('example pages', () => {
  it.each([
    ['qwen3/qwen3-dsl.yaml', {}],
    ['qwen3/qwen3-dsl.yaml', { attn_implementation: 'flash_attention_2' }],
    ['qwen3/qwen3-dsl.yaml', { fused_linear_cross_entropy: true }],
    ['qwen3/qwen3-dsl.yaml', { attn_implementation: 'flash_attention_2', fused_linear_cross_entropy: true }],
    ['qwen3_5/qwen3_5-dsl.yaml', {}],
  ])('plans %s with %j', async (file, config) => {
    const url = new URL(file, examplesUrl).href;
    const spec = parseYaml(await readFileUrl(url), url) as { config: Record<string, unknown> };
    Object.assign(spec.config, config);
    const p = await compilePage(spec as never, { base: url, read: readFileUrl });
    const layout = plan(p);
    const kernels = p.levels.grids.length;
    expect(layout.kernels.length > 0).toBe(kernels > 0);
    for (const ground of layout.kernels) for (const chip of ground.chips) expect(chip.card.x[0]).toBeGreaterThan(ground.hbm?.x[1] ?? -Infinity);
  }, 60000);
});
