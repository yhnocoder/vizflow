import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { graph } from '../../src/core';
import type { SceneController } from '../../src/scene3d/controller';
import { buildEdges } from '../../src/scene3d/edges';
import type { Block } from '../../src/scene3d/layout';
import type { Renderable } from '../../src/scene3d/renderable';

function block(node: Node, extra: Partial<Block> = {}): Block {
  const elements: Block['elements'] = [];
  const [rows, cols] = node.shape;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) elements.push({ index: [r, c], row: r, col: c, x: c, y: 0, z: r });
  return { node, elements, ...extra } as Block;
}

describe('buildEdges', () => {
  it('links each element of a view copy to the original element at the same offset', () => {
    const g = graph({ dims: {} });
    const x = g.param('x', [2, 3], [[1, 2, 3], [4, 5, 6]]);
    const v = g.view(x, 3, 2);
    const original = block(x);
    const copy = block(v, { key: 'copy', copyOf: x });
    const c = {
      lod: false,
      frontRowOnly: false,
      boards: [],
      renderables: [{ block: original, far: false }, { block: copy, far: false }] as Renderable[],
      explainer: { edges: () => [] },
      levelLayout: { io: [], links: [{ copy, original }], sourceOf: new Map() },
      state: { dir: 'forward', index: 1, revealed: () => true },
    } as unknown as SceneController;
    const links = buildEdges(c).filter((e) => e.link === true);
    expect(links.map((e) => e.fromNode)).toEqual(Array(6).fill(x));
    expect(links.map((e) => ('index' in e.edge.from ? e.edge.from.index : []))).toEqual([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]);
    expect(links.map((e) => e.edge.id)).toEqual(['link:copy:0,0', 'link:copy:0,1', 'link:copy:1,0', 'link:copy:1,1', 'link:copy:2,0', 'link:copy:2,1']);
  });
});
