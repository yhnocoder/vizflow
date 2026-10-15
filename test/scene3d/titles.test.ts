// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import type { Block } from '../../src/scene3d/layout';
import { levelPlan } from '../../src/scene3d/levels';
import { texWidth } from '../../src/scene3d/columns';
import { loadExample } from '../dsl/load';

const MINI = ['', '-gap-2', '-gap-labels', '-lines', '-prepare-side', '-prepare-start', '-prepare-use', '-span-members'].map((v) => `qwen3/qwen3-mini${v}.yaml`);
const PAGES = [...MINI, 'qwen3/qwen3-nested.yaml', 'qwen3_5/qwen3_5-nested.yaml'];

type Range = [number, number];
type Rect = { name: string; x: Range; depth: Range };

function boxes(block: Block): { x: Range; z: Range }[] {
  const plates = block.slabs?.map((slab) => slab.plate) ?? block.chunks ?? (block.plate === undefined ? [] : [block.plate]);
  if (plates.length > 0) return plates.map((p) => ({ x: [p.x - p.w / 2, p.x + p.w / 2], z: [p.z - p.d / 2, p.z + p.d / 2] }));
  return block.elements.map((e) => ({ x: [e.x - block.footprint[0] / 2, e.x + block.footprint[0] / 2], z: [e.z - block.footprint[1] / 2, e.z + block.footprint[1] / 2] }));
}

function depth(block: Block): Range {
  const zs = boxes(block).flatMap((b) => b.z);
  return [Math.min(...zs), Math.max(...zs)];
}

function blockRects(block: Block): Rect[] {
  return boxes(block).map((b) => ({ name: block.node.name, x: b.x, depth: depth(block) }));
}

function titleRect(block: Block): Rect | undefined {
  if (block.untitled === true) return undefined;
  const w = texWidth(block.node.tex);
  const { x, anchor } = block.title;
  const span: Range = anchor === 'left' ? [x - w, x] : anchor === 'right' ? [x, x + w] : [x - w / 2, x + w / 2];
  return { name: `title of ${block.node.name}`, x: span, depth: depth(block) };
}

function meets(a: Range, b: Range): boolean {
  return a[0] < b[1] - 1e-6 && b[0] < a[1] - 1e-6;
}

describe('titles on the example pages', () => {
  for (const name of PAGES) {
    it(`draws ${name} without a title overlapping another block or title of its level`, async () => {
      const page = await loadExample(name);
      const layout = levelPlan(page.steps, page.levels).resolve({ folded: () => false, aligned: new Map() });
      const byY = new Map<number, Block[]>();
      for (const block of layout.blocks) {
        if (block.copyOf !== undefined) continue;
        const y = block.elements[0].y;
        byY.set(y, [...(byY.get(y) ?? []), block]);
      }
      const clashes: string[] = [];
      for (const blocks of byY.values()) {
        for (const block of blocks) {
          const title = titleRect(block);
          if (title === undefined) continue;
          for (const other of blocks) {
            if (other === block) continue;
            const rects = [...blockRects(other), titleRect(other)].filter((r): r is Rect => r !== undefined);
            const hit = rects.find((r) => meets(title.x, r.x) && meets(title.depth, r.depth));
            if (hit !== undefined) clashes.push(`${title.name} [${title.x.map((v) => v.toFixed(2))}] meets ${hit.name} [${hit.x.map((v) => v.toFixed(2))}]`);
          }
        }
      }
      expect(clashes).toEqual([]);
    }, 120000);
  }
});
