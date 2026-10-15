// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { type CompiledPage, compilePage, loadPage, parseYaml } from '../../src/dsl';
import { BOX_MARGIN, CHIP_GAP, COPY_GAP, type LevelLayout, levelPlan } from '../../src/scene3d/levels';
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
