import { describe, expect, it } from 'vitest';
import { realSizeOf } from '../../src/core';
import { type CompiledPage, loadPage } from '../../src/dsl';
import { createSteps } from '../../src/steps';

const BASE = 'mem:/pages/page.yaml';
const GRID = 'mem:/pages/g.yaml';

interface Kernel {
  grid?: string;
  kernel?: boolean;
  shared?: string;
  inputs?: string;
  outputs?: string;
  forward: string;
  steps?: string;
  layout?: string;
}

function kernel(spec: Kernel): string {
  return [
    'module: G',
    spec.kernel === false ? '' : 'kernel: true',
    'args: { bm: dim }',
    `grid: ${spec.grid ?? '{ i: T // bm, h: 2 }'}`,
    spec.shared === undefined ? '' : `shared: ${spec.shared}`,
    `inputs: ${spec.inputs ?? '{ x: [T, _], y: [T, 2] }'}`,
    `outputs: ${spec.outputs ?? '{ y: [T, 2] }'}`,
    'forward:',
    spec.forward.replace(/^/gm, '  '),
    spec.layout === undefined ? '' : `layout: ${spec.layout}`,
    'steps:',
    (spec.steps ?? (spec.forward.startsWith('- tile') ? '- id: tile\n- id: y' : '- id: y')).replace(/^/gm, '  '),
  ].join('\n');
}

function root(forward = '- buf: zeros(T, 2)\n- y: g(x, buf)\n- l: y.sum()', steps = '- id: buf\n- id: g\n- id: l'): string {
  return `title: t
model: M
import: { G: ./g.yaml }
config:
  d: 2
  real:
    big: { name: Big, T: 8, d: 2 }
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
${forward.replace(/^/gm, '      ')}
    steps:
${steps.replace(/^/gm, '      ')}
`;
}

function page(spec: Kernel, forward?: string, steps?: string): Promise<CompiledPage> {
  const files: Record<string, string> = { [BASE]: root(forward, steps), [GRID]: kernel(spec) };
  return loadPage(BASE, {
    read: async (url) => {
      const text = files[url];
      if (text === undefined) throw new Error(`no file ${url}`);
      return text;
    },
  });
}

const TILES = "- tile: x[i * bm:(i + 1) * bm] * (h + 1)\n- 'y[i * bm:(i + 1) * bm, h]': tile.sum(-1)";

const ROUNDS = {
  grid: '{ i: 2 }',
  inputs: '{ x: [T, d], y: [T, d] }',
  outputs: '{ y: [T, d] }',
  forward: "- acc: x[0] * 0\n- for: n in range(i + 1)\n  do:\n    - acc: acc + x[n]\n- 'y[i]': acc",
  steps: "- id: acc\n- id: r\n  loop: n\n  defaults: { show: acc, title: 'copy {{ i }} round {{ round }} n {{ n }}' }\n- id: y",
};
const ROUNDS_ROOT = '- buf: zeros(T, d)\n- y: g(x, buf)\n- l: y.sum()';

describe('grid modules', () => {
  it('runs the forward once per copy, names copies by index and writes all tiles with one scatter_tiles', async () => {
    const p = await page({ forward: TILES });
    expect([...p.nodes.keys()].filter((name) => /^g\.tile\[[^/]*$/.test(name))).toEqual(['g.tile[0][0]', 'g.tile[0][1]', 'g.tile[1][0]', 'g.tile[1][1]']);
    expect(p.nodes.get('g.tile[1][0]')!.tex).toBe(p.nodes.get('g.tile[0][1]')!.tex);
    const scatters = p.graph.nodes.filter((n) => n.op === 'scatter_tiles');
    expect(scatters).toHaveLength(1);
    expect(p.nodes.get('y')).toBe(scatters[0]);
    expect(Array.from(scatters[0].value.data)).toEqual([3, 6, 7, 14, 11, 22, 15, 30]);
    const tile = p.graph.node('g.tile[1][1]')!;
    const sizes = p.graph.realSizes();
    const shapes = p.graph.realShapes();
    expect(tile.copies.drawn).toBe(4);
    expect(tile.copies.real.map((d) => realSizeOf(d, sizes, shapes) as number).reduce((a, b) => a * b, 1)).toBe(8);
    expect(p.steps.grids).toEqual([{ module: 'g', names: ['i', 'h'], counts: [2, 2] }]);
    const grid = p.levels.grids.find((entry) => entry.module === 'g')!;
    expect(grid.copyOf.get(tile)).toEqual([1, 1]);
    expect(grid.real.map((d) => realSizeOf(d, sizes, shapes))).toEqual([4, 2]);
  });

  it('rejects overlapping writes, and adds overlapping tiles with +=', async () => {
    await expect(page({ grid: '{ i: 2 }', forward: "- 'y[0:2, 0]': x[0:2, 0]" })).rejects.toThrow(/scatter_tiles: tiles 0 and 1 write the same element; overlapping tiles need accumulate/);
    const p = await page({ grid: '{ i: 2 }', forward: "- 'y[0:2, 0] +=': x[i * 2:i * 2 + 2, 1]" }, '- buf: ones(T, 2)\n- y: g(x, buf)\n- l: y.sum()');
    expect(Array.from(p.nodes.get('y')!.value.data)).toEqual([1 + 2 + 6, 1, 1 + 4 + 8, 1, 1, 1, 1, 1]);
    await expect(page({ grid: '{ i: 2 }', forward: "- 'y[i, 0] +=': x[i, 0]\n- 'y[i, 1]': x[i, 1]" })).rejects.toThrow(/'y' is written with both = and \+=/);
    const twice = await page({ grid: '{ i: 2 }', forward: "- 'y[i, 0]': x[i, 0]\n- 'y[i, 1]': x[i, 1] * 2" });
    expect(Array.from(twice.nodes.get('y')!.value.data)).toEqual([1, 4, 3, 8, 0, 0, 0, 0]);
    expect(twice.graph.nodes.map((n) => n.name).filter((name) => /#2$/.test(name))).toEqual(['y[0]#2', 'y[1]#2']);
  });

  it('rejects a read of an input after the copy writes it, and outputs that are not written inputs', async () => {
    await expect(page({ grid: '{ i: 2 }', forward: "- 'y[i, 0]': x[i, 0]\n- z: y[i, 1] * 2\n- 'y[i, 1]': z" })).rejects.toThrow(/'y' is read after this copy writes it; the copies write y together after the grid finishes/);
    await expect(page({ grid: '{ i: 2 }', outputs: '{ y: [T, 2], z: [T, 2] }', forward: "- 'y[i]': x[i]" })).rejects.toThrow(/output 'z' of G is not an input that the grid writes/);
    await expect(page({ grid: '{ i: 2 }', inputs: '{ x: [T, _], y: [T, 2], w: [T, 2] }', forward: "- 'y[i]': x[i]\n- 'w[i]': x[i]" }, '- buf: zeros(T, 2)\n- y: g(x, buf, buf)\n- l: y.sum()')).rejects.toThrow(/the grid writes input 'w', so 'w' must also be an output of G/);
  });

  it('compiles += outside a grid to a scatter of the old value plus the new one', async () => {
    const p = await page({ grid: '{ i: 2 }', forward: "- 'y[i]': x[i]" }, "- buf: zeros(T, 2)\n- y: g(x, buf)\n- s: x * 1\n- for: k in range(2)\n  do:\n    - 's[k] +=': x[k + 2]\n- l: y.sum() + s.sum()", '- id: buf\n- id: g\n- id: s\n- id: add\n  loop: k\n  defaults: { show: s }\n- id: l');
    expect(Array.from(p.nodes.get('s[1]')!.value.data)).toEqual([6, 8, 10, 12, 5, 6, 7, 8]);
  });

  it('makes one loop step per round, lights only the copies that have that round and keeps the steps of idle copies dark', async () => {
    const p = await page(ROUNDS, ROUNDS_ROOT);
    const rounds = p.steps.forward.filter((s) => s.grid === 'g' && s.copies?.some((c) => c.title.startsWith('copy')));
    expect(rounds).toHaveLength(2);
    expect(rounds.map((s) => (Array.isArray(s.show) ? s.show : [s.show]).map((n) => n.name).filter((name) => !name.includes('/')))).toEqual([['g.acc[0][0]', 'g.acc[1][0]'], ['g.acc[1][1]']]);
    expect(rounds.map((s) => s.copies!.map((c) => c.title))).toEqual([['copy 0 round 0 n 0', 'copy 1 round 0 n 0'], ['copy 1 round 1 n 1']]);
    await expect(page({ ...ROUNDS, steps: '- id: r\n  loop: q\n  defaults: { show: acc }\n- id: y' }, ROUNDS_ROOT)).rejects.toThrow(/loop 'q' is not a loop of the forward of G/);
    await expect(page({ ...ROUNDS, steps: '- id: r\n  loop: n in range(i + 1)\n  defaults: { show: acc }\n- id: y' }, ROUNDS_ROOT)).rejects.toThrow(/is no longer written with range; write loop: n/);
  });

  it('evaluates step texts for every copy and follows the front copy', async () => {
    const p = await page(ROUNDS, ROUNDS_ROOT);
    const steps = createSteps(p.graph, p.steps);
    const titles = () => steps.list.filter((s) => s.copy !== undefined && s.title.startsWith('copy')).map((s) => [s.title, s.copy!.index, s.copy!.front]);
    expect(steps.front('g')).toEqual([1]);
    expect(titles()).toEqual([
      ['copy 1 round 0 n 0', [1], true],
      ['copy 1 round 1 n 1', [1], true],
    ]);
    let changes = 0;
    const stop = steps.subscribe(() => changes++);
    steps.setFront('g', [0]);
    expect(changes).toBeGreaterThan(0);
    expect(titles()).toEqual([
      ['copy 0 round 0 n 0', [0], true],
      ['copy 1 round 1 n 1', [1], false],
    ]);
    expect(() => steps.setFront('g', [2])).toThrow(/setFront: \[2\] is not a copy of 'g'/);
    expect(() => steps.front('h')).toThrow(/front: 'h' is not a grid module call/);
    stop();
    steps.dispose();
  });

  it('checks shared and layout.grid', async () => {
    await expect(page({ forward: TILES, kernel: false, shared: '[tile]' })).rejects.toThrow(/shared lists the tensors a kernel keeps in shared memory, so it needs kernel: true; G is not a kernel/);
    await expect(page({ forward: TILES, shared: '[y]' })).rejects.toThrow(/'y' is an output of G; outputs stay in device memory/);
    await expect(page({ forward: TILES, shared: '[tiles]' })).rejects.toThrow(/'tiles' is not a tensor computed in the forward of G/);
    const p = await page({ forward: TILES, shared: '[tile]' });
    const grid = p.levels.grids.find((entry) => entry.module === 'g')!;
    expect([...grid.shared].map((n) => n.name).filter((name) => !name.includes('/'))).toEqual(['g.tile[0][0]', 'g.tile[0][1]', 'g.tile[1][0]', 'g.tile[1][1]']);
    expect([...grid.registers].some((n) => grid.shared.has(n))).toBe(false);
    await expect(page({ forward: TILES, layout: '{ grid: { j: expand } }' })).rejects.toThrow(/layout.grid names 'j', which is not a dimension of the grid of G/);
    await expect(page({ forward: TILES, layout: '{ grid: { i: spread } }' })).rejects.toThrow(/layout.grid 'i' must be expand or stack/);
  });

  it('rejects layout.grid in a module without a grid', async () => {
    const files: Record<string, string> = {
      [BASE]: root('- buf: zeros(T, 2)\n- y: g(x, buf)\n- l: y.sum()'),
      [GRID]: 'module: G\nargs: { bm: dim }\ninputs: { x: [T, _], y: [T, 2] }\noutputs: { z: [T, 2] }\nforward:\n  - z: y * 2\nlayout: { grid: { i: expand } }\nsteps:\n  - id: z',
    };
    await expect(loadPage(BASE, { read: async (url) => files[url] })).rejects.toThrow(/layout.grid arranges the copies of a grid module; G has no grid/);
  });

  it('groups the rounds and the stacked copies of a name into one slot, and lays expanded copies side by side', async () => {
    const stacked = await page(ROUNDS, ROUNDS_ROOT);
    const names = (nodes: readonly { name: string }[]) => nodes.map((n) => n.name);
    const accs = stacked.levels.slots.map(names).filter((group) => /^g\.acc[^/]*$/.test(group[0]));
    expect(accs).toEqual([['g.acc[0]', 'g.acc[1]'], ['g.acc[0][0]', 'g.acc[1][0]', 'g.acc[1][1]']]);
    const drawn = stacked.levels.lines.flat().map((n) => n.name);
    expect(drawn).toContain('g.acc[0][0]');
    expect(drawn).not.toContain('g.acc[1][0]');
    expect(drawn).not.toContain('g.acc[1][1]');
    const expanded = await page({ ...ROUNDS, layout: '{ grid: { i: expand } }' }, ROUNDS_ROOT);
    expect(expanded.levels.slots.map(names).filter((group) => /^g\.acc[^/]*$/.test(group[0]))).toEqual([['g.acc[1][0]', 'g.acc[1][1]']]);
    const levels = expanded.levels.lines.map((level) => level.map((n) => n.name));
    expect(levels.find((level) => level.includes('g.acc[0][0]'))).toContain('g.acc[1][0]');
  });
});
