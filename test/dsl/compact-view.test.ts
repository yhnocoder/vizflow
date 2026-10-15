import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, loadPage } from '../../src/dsl';
import { type ModuleTile, type Steps, createSteps, tileLeaves, tileStates } from '../../src/steps';
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

  it('makes one tile per top-level step of the model and splits loop iterations into their child calls', () => {
    expect(titles(steps.tiles)).toEqual(['embedding', 'position_and_mask', 'layers[0]', 'layers[1]', 'model.norm', 'lm_head', '转成 float32', 'log_softmax']);
    const layer = steps.tiles[2];
    expect(layer.subtitle).toBe('Qwen3DecoderLayer');
    expect(titles(layer.children)).toEqual(['input_layernorm', 'self_attn', '残差相加', 'post_attention_layernorm', 'mlp', '残差相加']);
    expect(steps.tiles.filter((tile) => tile.children.length === 0 && tile.caches.length > 0)).toEqual([]);
    expect(cacheNames(layer.children[1])).toEqual(['model.layers[0].self_attn.kr']);
    expect(cacheNames(steps.tiles[3].children[1])).toEqual(['model.layers[1].self_attn.kr']);
  });

  it('lights the tiles each compact step of decode 2 runs, after the ones it already ran', () => {
    expect(compactSteps(steps, 3)).toEqual([
      { id: 'd:3:0', lit: ['/embedding', '/position_and_mask', 'layers[0]/input_layernorm'], done: [], sample: false },
      { id: 'd:3:1', lit: ['layers[0]/self_attn'], done: ['/embedding', '/position_and_mask', 'layers[0]/input_layernorm'], sample: false },
      {
        id: 'd:3:2',
        lit: ['layers[0]/残差相加', 'layers[0]/post_attention_layernorm', 'layers[0]/mlp', 'layers[0]/残差相加', 'layers[1]/input_layernorm'],
        done: ['/embedding', '/position_and_mask', 'layers[0]/input_layernorm', 'layers[0]/self_attn'],
        sample: false,
      },
      expect.objectContaining({ id: 'd:3:3', lit: ['layers[1]/self_attn'] }),
      expect.objectContaining({ id: 'd:3:4', lit: ['layers[1]/残差相加', 'layers[1]/post_attention_layernorm', 'layers[1]/mlp', 'layers[1]/残差相加', '/model.norm', '/lm_head', '/转成 float32', '/log_softmax'] }),
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
  });
});
