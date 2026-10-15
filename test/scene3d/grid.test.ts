// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, loadPage, parseYaml } from '../../src/dsl';
import { BOX_MARGIN, CHIP_GAP, COPY_GAP, type LevelLayout, levelPlan } from '../../src/scene3d/levels';
import { STACK_CORNER, STACK_DX, STACK_DZ, stackEdge } from '../../src/scene3d/kernels';
import { isIndexView } from '../../src/scene3d/layout';
import { examplesUrl, readFileUrl } from '../dsl/load';

const BASE = 'mem:/pages/page.yaml';
const GRID = 'mem:/pages/g.yaml';

function kernel(layout?: string): string {
  return [
    'module: G',
    'kernel: true',
    'args: { bm: dim }',
    'grid: { i: 2 }',
    'shared: [row]',
    'inputs: { x: [T, d], y: [T, d] }',
    'outputs: { y: [T, d] }',
    'forward:',
    '  - row: x[i]',
    '  - acc: row * 0',
    '  - for: n in range(i + 1)',
    '    do:',
    '      - acc: acc + x[n]',
    "  - 'y[i]': acc",
    layout === undefined ? '' : `layout: ${layout}`,
    'steps:',
    '  - id: row',
    '    show: [row, acc]',
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
      - buf: zeros(T, d)
      - y: g(x, buf)
      - l: y.sum()
    steps:
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

function xsOf(layout: LevelLayout, name: RegExp): number[] {
  return layout.blocks.filter((b) => name.test(b.node.name)).flatMap((b) => b.elements.map((e) => e.x));
}

describe('grid kernels in the level layout', () => {
  it('draws the rounds and the stacked copies of a name as one block', async () => {
    const layout = plan(await page());
    const accs = layout.blocks.filter((b) => /^g\.acc\[[^/]*$/.test(b.node.name));
    expect(accs.map((b) => b.slot?.map((n) => n.name))).toEqual([['g.acc[0]', 'g.acc[1]'], ['g.acc[0][0]', 'g.acc[1][0]', 'g.acc[1][1]']]);
  });

  it('puts shared memory in a column left of the registers and frames both', async () => {
    const layout = plan(await page());
    expect(layout.kernels).toHaveLength(1);
    const region = layout.kernels[0];
    expect(region.grids.map((g) => g.module)).toEqual(['g']);
    const shared = xsOf(layout, /^g\.row\[[^/]*$/);
    const registers = xsOf(layout, /^g\.acc\[/);
    expect(Math.min(...registers) - Math.max(...shared)).toBeCloseTo(CHIP_GAP);
    expect(region.shared!.x[1]).toBeCloseTo(Math.max(...shared) + BOX_MARGIN.right);
    expect(region.registers!.x[0]).toBeCloseTo(Math.min(...registers) - BOX_MARGIN.left);
    expect(region.frame.x).toEqual([region.shared!.x[0], region.registers!.x[1]]);
    expect([...region.grids[0].shared, ...region.grids[0].registers].some((n) => n.name === 'x' || n.name === 'y')).toBe(false);
  });

  it('lays expanded copies side by side, COPY_GAP apart, each with its own frame', async () => {
    const layout = plan(await page('{ grid: { i: expand } }'));
    expect(layout.kernels.map((k) => k.expand)).toEqual([[0], [1]]);
    const [a, b] = layout.kernels;
    expect(b.frame.x[0] - a.frame.x[1]).toBeCloseTo(COPY_GAP);
  });

  it('draws no block for an anonymous index node and starts its edges on the source', async () => {
    const p = await page();
    const layout = plan(p);
    const anonymous = p.graph.nodes.filter((n) => isIndexView(n));
    expect(anonymous.length).toBeGreaterThan(0);
    expect(layout.blocks.filter((b) => isIndexView(b.node))).toEqual([]);
    expect(anonymous.every((n) => layout.sourceOf.get(n)?.name === 'x')).toBe(true);
  });
});

describe('stacked outlines', () => {
  it('draw the top edge and the top STACK_CORNER of the right edge, shifted back and right', () => {
    const box = { x: [0, 4] as [number, number], y: [0, 10] as [number, number], z: [-1, 1] as [number, number] };
    expect(stackEdge(box, 2)).toEqual([2 * STACK_DX, 10, 1 - 2 * STACK_DZ, 4 + 2 * STACK_DX, 10, 1 - 2 * STACK_DZ, 4 + 2 * STACK_DX, 10 - STACK_CORNER, 1 - 2 * STACK_DZ]);
  });

  it('stop at the bottom of a region lower than STACK_CORNER', () => {
    const box = { x: [0, 4] as [number, number], y: [0, 1] as [number, number], z: [0, 0] as [number, number] };
    expect(stackEdge(box, 1)[7]).toBe(0);
  });
});

describe('FlashAttention forward kernel', () => {
  it('draws the row vectors as discs beside the tensors of the same round', async () => {
    const url = new URL('qwen3/qwen3-dsl.yaml', examplesUrl).href;
    const spec = parseYaml(await readFileUrl(url), url) as { config: Record<string, unknown> };
    Object.assign(spec.config, { attn_implementation: 'flash_attention_2' });
    const layout = plan(await compilePage(spec as never, { base: url, read: readFileUrl }));
    const kernel = layout.blocks.filter((b) => /^model\.layers\[0\]\..*\.fwd\./.test(b.node.name));
    const named = (name: string) => kernel.find((b) => new RegExp(`\\.fwd\\.${name}(\\[\\d+\\])+$`).test(b.node.name))!;
    for (const name of ['prev_max', 'row_max', 'scores_scale', 'row_sum_scaled', 'row_sum']) expect(named(name).shape).toBe('disc');
    const y = (name: string) => named(name).elements[0].y;
    expect(y('prev_max')).toBeCloseTo(y('cols'));
    expect(y('prev_max')).toBeLessThan(y('row_max'));
    const level = (b: (typeof kernel)[number]) => b.elements[0].y.toFixed(3);
    const discs = kernel.filter((b) => b.shape === 'disc');
    expect(new Set(discs.map(level)).size).toBeLessThan(discs.length);
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
    const kernels = p.levels.grids.filter((g) => g.shared.size + g.registers.size > 0).length;
    expect(layout.kernels.length > 0).toBe(kernels > 0);
  }, 60000);
});
