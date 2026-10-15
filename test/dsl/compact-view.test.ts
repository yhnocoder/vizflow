import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, loadPage } from '../../src/dsl';
import { positionsOf } from '../../src/explain';
import { INPUT_TITLE, type ModuleTile, OUTPUT_TITLE, type Steps, createSteps, tileLeaves, tileStates } from '../../src/steps';
import { autoLayout } from '../../src/scene3d/auto';
import { compactBlocks } from '../../src/scene3d/compactlayout';
import { resolveLayout, targetColumn } from '../../src/scene3d/layout';
import { forEachIndex } from '../../src/tensor';
import { examplesUrl, loadExample, readFileUrl } from './load';

const PAGE = 'qwen3/qwen3-infer-dsl.yaml';

function loadEdited(name: string, edit: (text: string) => string): Promise<CompiledPage> {
  return loadPage(new URL(name, examplesUrl).href, { read: async (url) => (url.endsWith(name) ? edit(await readFileUrl(url)) : readFileUrl(url)) });
}

function titles(tiles: readonly ModuleTile[]): string[] {
  return tiles.map((tile) => tile.title);
}

function cacheNames(tile: ModuleTile): string[] {
  return tile.caches.map((decl) => decl.nodes[0].name);
}

function compactSteps(steps: Steps, row: number): { id: string; lit: string[]; done: string[]; sample: boolean }[] {
  const leaves = tileLeaves(steps.tiles);
  const name = (tile: ModuleTile) => `${steps.tiles.find((top) => top.children.includes(tile))?.title ?? ''}/${tile.title}`;
  return steps.list.flatMap((info, i) => {
    if (info.id !== `s:${row}` && !info.id.startsWith(`d:${row}:`)) return [];
    steps.go(i);
    const compact = steps.state().generation?.compact;
    if (compact === undefined) return [];
    const states = tileStates(leaves, compact);
    return [{ id: info.id, lit: leaves.filter((_, k) => states[k] === 'lit').map(name), done: leaves.filter((_, k) => states[k] === 'done').map(name), sample: compact.sample }];
  });
}

describe('module tiles of the compact decode view', () => {
  let page: CompiledPage;
  let steps: Steps;

  beforeAll(async () => {
    page = await loadExample(PAGE);
    steps = createSteps(page.graph, page.modes!.infer);
  });

  it('turns on with generate.compact_view and rejects other values', async () => {
    expect(page.compactView?.infer).toBe(page.modes!.infer.infer);
    await expect(loadEdited(PAGE, (text) => text.replace('compact_view: modules', 'compact_view: layers'))).rejects.toThrow(/compact_view must be 'graph' or 'modules'/);
    const graph = await loadEdited(PAGE, (text) => text.replace('compact_view: modules', 'compact_view: graph'));
    expect(graph.compactView).toBeUndefined();
  });

  it('marks the groups that come from a step loop', () => {
    const loops = steps.groups.filter((group) => group.loop === true).map((group) => group.id);
    expect(loops).toEqual(['model.layers[0]', 'model.layers[1]']);
  });

  it('makes one tile per top-level module, merges the steps before and after the layers, and cuts each layer at its own steps', () => {
    expect(titles(steps.tiles)).toEqual([INPUT_TITLE, 'layers[0]', 'layers[1]', OUTPUT_TITLE]);
    expect(steps.tiles[0].names).toEqual(['embedding', 'position_and_mask']);
    expect(steps.tiles[3].names).toEqual(['model.norm', 'lm_head', '转成 float32', 'log_softmax']);
    const layer = steps.tiles[1];
    expect(layer.subtitle).toBe('Qwen3DecoderLayer');
    expect(titles(layer.children)).toEqual(['self_attn', 'mlp']);
    expect(layer.children.map((tile) => tile.names)).toEqual([
      ['input_layernorm', 'self_attn', '残差相加'],
      ['post_attention_layernorm', 'mlp', '残差相加'],
    ]);
    expect(steps.tiles.filter((tile) => tile.children.length === 0 && tile.caches.length > 0)).toEqual([]);
    expect(cacheNames(layer.children[0])).toEqual(['model.layers[0].self_attn.kr']);
    expect(cacheNames(layer.children[1])).toEqual([]);
    expect(cacheNames(steps.tiles[2].children[0])).toEqual(['model.layers[1].self_attn.kr']);
  });

  it('lights the tiles each compact step of decode 2 runs, after the ones it already ran', () => {
    expect(compactSteps(steps, 3)).toEqual([
      { id: 'd:3:0', lit: ['/输入'], done: [], sample: false },
      { id: 'd:3:1', lit: ['layers[0]/self_attn'], done: ['/输入'], sample: false },
      { id: 'd:3:2', lit: ['layers[0]/mlp'], done: ['/输入', 'layers[0]/self_attn'], sample: false },
      { id: 'd:3:3', lit: ['layers[1]/self_attn'], done: ['/输入', 'layers[0]/self_attn', 'layers[0]/mlp'], sample: false },
      { id: 'd:3:4', lit: ['layers[1]/mlp', '/输出'], done: ['/输入', 'layers[0]/self_attn', 'layers[0]/mlp', 'layers[1]/self_attn'], sample: false },
      expect.objectContaining({ id: 's:3', lit: [], sample: true }),
    ]);
    const last = compactSteps(steps, 3).at(-1)!;
    expect(last.done).toHaveLength(tileLeaves(steps.tiles).length);
  });

  it('leaves decode 1 and earlier without a compact step, and reports the cache rows read at each position', () => {
    expect(compactSteps(steps, 2)).toEqual([]);
    const at = steps.list.findIndex((info) => info.id === 'd:3:1');
    steps.go(at);
    const generation = steps.state().generation!;
    expect(generation.position).toBe(3);
    expect(generation.caches.map((c) => [c.kind, c.read])).toEqual([
      ['rows', [0, 1, 2]],
      ['rows', [0, 1, 2]],
    ]);
  });

  it('places state and window caches of a linear attention layer on its tile without model-specific rules', async () => {
    const q35 = await loadExample('qwen3_5/qwen3_5-infer-dsl.yaml');
    expect(q35.compactView).toBeUndefined();
    const s = createSteps(q35.graph, q35.modes!.infer);
    const layer0 = s.tiles.find((tile) => tile.title === 'layers[0]')!;
    const linear = layer0.children.find((tile) => tile.title === 'linear_attn')!;
    expect(linear.caches.map((decl) => decl.nodes[0].name).sort()).toEqual(['model.layers[0].linear_attn.mixed_t', 'model.layers[0].linear_attn.recurrent_gated_delta_rule.last_recurrent_state[0]']);
    const at = s.list.findIndex((info) => info.id === 'd:4:1');
    s.go(at);
    const generation = s.state().generation!;
    const kinds = Object.fromEntries(generation.caches.map((c) => [c.kind, c]));
    expect(kinds.state.nodes.map((n) => n.name)).toEqual(['model.layers[0].linear_attn.recurrent_gated_delta_rule.last_recurrent_state[4]']);
    expect(kinds.state.previous.map((n) => n.name)).toEqual(['model.layers[0].linear_attn.recurrent_gated_delta_rule.last_recurrent_state[3]']);
    expect(kinds.window.read).toEqual([1, 2, 3]);
    expect(compactSteps(s, 4).find((step) => step.id === 'd:4:1')?.lit).toEqual(['layers[0]/linear_attn']);
    expect(titles(s.tiles)).toEqual([INPUT_TITLE, 'layers[0]', 'layers[1]', OUTPUT_TITLE]);
    expect(titles(layer0.children)).toEqual(['linear_attn', 'mlp']);
  });

  it('reveals the current state and hides cache rows outside the window in a compact step', async () => {
    const q35 = await loadExample('qwen3_5/qwen3_5-infer-dsl.yaml');
    const s = createSteps(q35.graph, q35.modes!.infer);
    s.go(s.list.findIndex((info) => info.id === 'd:4:1'));
    const generation = s.state().generation!;
    expect(generation.compact).toBeDefined();
    const state = generation.caches.find((c) => c.kind === 'state')!;
    expect([...state.nodes, ...state.previous].every((n) => s.state().revealed(n))).toBe(true);
    const window = generation.caches.find((c) => c.kind === 'window')!;
    const node = window.decl.nodes[0];
    const positions = positionsOf(q35.graph, q35.modes!.infer.infer!.input.shape[0]);
    const hidden = new Map<number, boolean>();
    forEachIndex(node.shape, (index, offset) => {
      const h = positions.horizon(node, offset);
      if (h !== undefined && h <= generation.position) hidden.set(h, (hidden.get(h) ?? true) && s.state().row?.(node, index) === 'hidden');
    });
    expect([...hidden].filter(([, all]) => all).map(([h]) => h).sort()).toEqual(Array.from({ length: window.read[0] }, (_, h) => h));
  });

  it('cuts DeepSeek V3 layers the same way', async () => {
    const ds = await loadExample('deepseek_v3/deepseek_v3-infer-dsl.yaml');
    const s = createSteps(ds.graph, ds.modes!.infer);
    expect(titles(s.tiles)).toEqual([INPUT_TITLE, 'layers[0]', 'layers[1]', OUTPUT_TITLE]);
    expect(titles(s.tiles[1].children)).toEqual(['self_attn', 'mlp']);
    expect(s.tiles[1].children[0].caches).toHaveLength(ds.modes!.infer.infer!.caches.length / 2);
  });

  it('moves the cache blocks next to their module tile and hides every other block', () => {
    const { layout, levels } = autoLayout(page.tree, page.steps, page.levels);
    const resolved = resolveLayout(layout, { visible: steps.visible, aligned: new Map(), targetOf: targetColumn, focus: 0, folded: () => false, plan: levels });
    steps.go(steps.list.findIndex((info) => info.id === 'd:3:1'));
    const generation = steps.state().generation!;
    const positions = positionsOf(page.graph, page.compactView!.infer.input.shape[0]);
    const plan = compactBlocks(resolved, steps.tiles, generation, undefined, {
      tileWidth: 8,
      tileDepth: 2,
      up: [0, 0.8, -0.6],
      side: [1, 0, 0],
      tokens: ['a', 'b', 'c', 'd', 'e'],
      horizon: (node, offset) => positions.horizon(node, offset),
    });
    const shown = plan.layout.blocks.filter((block) => block.hidden !== true);
    expect(shown.map((block) => block.node.name).sort()).toEqual(['model.layers[0].self_attn.kr', 'model.layers[0].self_attn.v_linear', 'model.layers[1].self_attn.kr', 'model.layers[1].self_attn.v_linear']);
    expect(plan.layout.blocks).toHaveLength(resolved.blocks.length);
    expect(plan.tiles.map((tile) => tile.tile?.title ?? '')).toEqual([INPUT_TITLE, 'self_attn', 'mlp', 'self_attn', 'mlp', OUTPUT_TITLE, '']);
    expect(plan.tiles.map((tile) => tile.state)).toEqual(['done', 'lit', 'idle', 'idle', 'idle', 'idle', 'idle']);
    const ys = plan.tiles.map((tile) => tile.y);
    expect(ys.every((y, i) => i === 0 || y > ys[i - 1])).toBe(true);
    for (const board of plan.boards) {
      expect(board.y).toBe(plan.tiles[board.leaf].y);
      expect(board.x[0]).toBeGreaterThan(8);
      expect(board.block.elements.every((e) => e.y === board.y)).toBe(true);
    }
    expect(plan.boards.map((board) => board.leaf)).toEqual([1, 1, 3, 3]);
  });
});
