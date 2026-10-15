import { beforeAll, describe, expect, it } from 'vitest';
import { type CompiledPage, loadPage } from '../../src/dsl';
import { simulateVllmPages } from '../../src/sim';
import { PAGE_REQUEST, engineStepAt, kvScenario, pageTokens, slotOf, slotSources, touchedPositions, writtenPositions } from '../../src/steps';
import { examplesUrl, loadExample, readFileUrl } from './load';

const PAGE = 'qwen3/qwen3-kv-scene-dsl.yaml';

function loadEdited(edit: (text: string) => string): Promise<CompiledPage> {
  return loadPage(new URL(PAGE, examplesUrl).href, { read: async (url) => (url.endsWith(PAGE) ? edit(await readFileUrl(url)) : readFileUrl(url)) });
}

describe('qwen3-kv-scene-dsl.yaml, the vLLM block pool drawn in the scene', () => {
  let scene: CompiledPage;
  let panel: CompiledPage;

  beforeAll(async () => {
    scene = await loadExample(PAGE);
    panel = await loadExample('qwen3/qwen3-infer-dsl.yaml');
  });

  it('moves the kv_blocks spec with scene: true out of the panels and into the scene', () => {
    const hasPanel = (items: CompiledPage['panels']) => items.some((item) => typeof item === 'object' && 'kvBlocks' in item);
    expect(hasPanel(scene.modePanels!.infer)).toBe(false);
    expect(hasPanel(scene.modePanels!.train)).toBe(false);
    expect(scene.kvPool).toMatchObject({ blockSize: 2, numBlocks: 8, others: [{ id: 'B', prompt: ['the', 'cat', 'on', 'mat'], arrive: 1 }] });
    expect(scene.kvPool!.infer).toBe(scene.modes!.infer.infer);
    expect(hasPanel(panel.modePanels!.infer)).toBe(true);
    expect(panel.kvPool).toBeUndefined();
  });

  it('builds the same scenario as the panel page, with the page request first', () => {
    const fromPanel = panel.modePanels!.infer.find((item) => typeof item === 'object' && 'kvBlocks' in item) as { kvBlocks: NonNullable<CompiledPage['kvPool']> };
    expect(kvScenario(scene.kvPool!)).toEqual(kvScenario(fromPanel.kvBlocks));
    const tokens = pageTokens(scene.kvPool!.infer);
    const scenario = kvScenario(scene.kvPool!);
    expect(scenario.requests[0]).toEqual({ id: PAGE_REQUEST, prompt: tokens.slice(0, 2), outputs: tokens.slice(2), arrive: 0 });
    expect(scenario.steps).toBe(tokens.length - 2 + 1 + 1);
  });

  it('maps each computed position to slot block_table[t // block_size], offset t % block_size, with the first request owning a shared slot', () => {
    const steps = simulateVllmPages(kvScenario(scene.kvPool!));
    expect(engineStepAt(steps, 1)).toBe(0);
    expect(engineStepAt(steps, 2)).toBe(1);
    expect(engineStepAt(steps, 3)).toBe(2);
    const decode1 = steps[1];
    const sources = slotSources(decode1, 2);
    expect(sources[1]).toEqual([{ request: 'A', position: 0 }, { request: 'A', position: 1 }]);
    expect(sources[2]).toEqual([{ request: 'A', position: 2 }, null]);
    expect(sources[3]).toEqual([{ request: 'B', position: 2 }, { request: 'B', position: 3 }]);
    expect(sources[0]).toEqual([null, null]);
    const a = decode1.requests.find((r) => r.id === 'A')!;
    const b = decode1.requests.find((r) => r.id === 'B')!;
    expect(slotOf(a.table, 2, 2)).toEqual({ block: 2, offset: 0 });
    expect(writtenPositions(a)).toEqual([2, 3]);
    expect(writtenPositions(b)).toEqual([2, 4]);
    expect(touchedPositions(steps[0], b)).toEqual([0, 4]);
    expect(touchedPositions(steps[0], a)).toEqual([2, 3]);
  });

  it('rejects a scene option that is not a boolean, and panel text on a pool drawn in the scene', async () => {
    await expect(loadEdited((text) => text.replace('scene: true', 'scene: 1'))).rejects.toThrow(/scene must be true or false/);
    await expect(loadEdited((text) => text.replace('      scene: true\n    mode: infer\n', '      scene: true\n    mode: infer\n    title: 块池\n'))).rejects.toThrow(/unknown key 'title'/);
  });
});
