// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import type { Block } from '../../src/scene3d/layout';
import { levelPlan } from '../../src/scene3d/levels';
import { texWidth } from '../../src/scene3d/columns';
import { loadExample } from '../dsl/load';

const PAGES = ['qwen3/qwen3-dsl.yaml', 'qwen3/qwen3-nested.yaml', 'qwen3_5/qwen3_5-nested.yaml', 'qwen3/qwen3-mini.yaml'];

type Box = { x: [number, number]; z: [number, number] };

function boxes(block: Block): Box[] {
  const plates = block.slabs?.map((slab) => slab.plate) ?? block.chunks ?? (block.plate === undefined ? [] : [block.plate]);
  if (plates.length > 0) return plates.map((p) => ({ x: [p.x - p.w / 2, p.x + p.w / 2], z: [p.z - p.d / 2, p.z + p.d / 2] }));
  return block.elements.map((e) => ({ x: [e.x - block.footprint[0] / 2, e.x + block.footprint[0] / 2], z: [e.z - block.footprint[1] / 2, e.z + block.footprint[1] / 2] }));
}

function meets(a: [number, number], b: [number, number]): boolean {
  return a[0] < b[1] - 1e-6 && b[0] < a[1] - 1e-6;
}

describe('column assignment on the example pages', () => {
  for (const name of PAGES) {
    it(`draws ${name} without two blocks of one level overlapping`, async () => {
      const page = await loadExample(name);
      const layout = levelPlan(page.steps, page.levels).resolve({ folded: () => false, aligned: new Map() });
      const byY = new Map<number, Block[]>();
      for (const block of layout.blocks) {
        if (block.copyOf !== undefined) continue;
        const y = block.elements[0].y;
        byY.set(y, [...(byY.get(y) ?? []), block]);
      }
      const clashes: string[] = [];
      for (const block of layout.blocks) {
        if (block.copyOf !== undefined) continue;
        const rows = new Map<number, number[]>();
        for (const e of block.elements) rows.set(e.z, [...(rows.get(e.z) ?? []), e.x]);
        for (const xs of rows.values()) {
          const sorted = [...xs].sort((a, b) => a - b);
          if (sorted.some((x, i) => i > 0 && x - sorted[i - 1] < block.footprint[0] - 1e-6)) clashes.push(`${block.node.name} overlaps itself [${sorted.map((v) => v.toFixed(2))}]`);
        }
      }
      for (const blocks of byY.values()) {
        for (let i = 0; i < blocks.length; i++) {
          for (let j = i + 1; j < blocks.length; j++) {
            const hit = boxes(blocks[i]).find((a) => boxes(blocks[j]).some((b) => meets(a.x, b.x) && meets(a.z, b.z)));
            if (hit !== undefined) clashes.push(`${blocks[i].node.name} [${hit.x.map((v) => v.toFixed(2))}] and ${blocks[j].node.name}`);
          }
        }
      }
      expect(clashes).toEqual([]);
    }, 120000);
  }
});

describe('texWidth', () => {
  it('counts letters, treats commands as one glyph and scripts as smaller', () => {
    expect(texWidth('t')).toBeCloseTo(0.3);
    expect(texWidth('\\mathbf{h}')).toBeCloseTo(0.3);
    expect(texWidth('\\boldsymbol{\\gamma}')).toBeCloseTo(0.3);
    expect(texWidth('\\cos(t\\,\\boldsymbol{\\omega})')).toBeCloseTo(0.3 * 7);
    expect(texWidth('\\mathbf{k}^{[0]}_{\\mathrm{fp32}}')).toBeCloseTo(0.3 * (1 + 0.6 * 7));
  });
});
