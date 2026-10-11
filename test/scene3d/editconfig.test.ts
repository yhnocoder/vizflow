import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import '../../src/explain';
import { editableLayout, layoutCode, literal, moveTarget, readPosition, writePosition } from '../../src/scene3d/editconfig';
import { type Block, type LayoutItem, board, classAlignment, node, resolveLayout, stack, targetColumn } from '../../src/scene3d/layout';

function build() {
  const g = graph({ dims: { d: 4, V: 5 } });
  const h = g.input('h', [1, 'd'], [0.8, -1.1, 0.5, 1.4]);
  const W = g.param('W', ['V', 'd'], [
    [0.9, -0.6, 0.3, 0.6],
    [0.2, -0.8, 0.6, 0.2],
    [-0.4, 0.2, 0.9, 0.3],
    [0.5, 0.7, -0.6, -0.2],
    [-0.7, 0.4, -0.3, -0.5],
  ]);
  const t = g.label('t', [0], { tokens: ['a', 'b', 'c', 'd', 'e'] });
  const z = g.matmul(h, W.T, { name: 'z' });
  const L = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
  const logp = g.sub(z, L, { name: 'log p' });
  const loss = g.nll_loss(logp, t, { name: 'loss' });
  const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
  return { g, h, W, z, L, logp, loss, ctx };
}

const blockOf = (blocks: Block[], name: string) => blocks.find((b) => b.node.name === name)!;

function evaluate(code: string, g: unknown): { layout: LayoutItem[]; camera?: object; labels?: object } {
  return new Function('g', 'board', 'stack', 'node', `return {\n${code}\n};`)(g, board, stack, node);
}

describe('stack offsets', () => {
  it('moves the named layer and the layers built on it, and leaves earlier layers in place', () => {
    const { h, W, z, L, logp, loss, ctx } = build();
    const plain = resolveLayout([board(h, W.T, z), stack(z, L, logp, loss)], ctx).blocks;
    const moved = resolveLayout([board(h, W.T, z), stack(z, L, logp, loss, { offsets: { 'log p': [1, 0.5, -2] } })], ctx).blocks;
    const shift = (a: Block, b: Block) => a.elements.map((e, i) => [b.elements[i].x - e.x, b.elements[i].y - e.y, b.elements[i].z - e.z]);
    for (const d of shift(blockOf(plain, 'log p'), blockOf(moved, 'log p'))) {
      expect(d[0]).toBeCloseTo(1, 12);
      expect(d[1]).toBeCloseTo(0.5, 12);
      expect(d[2]).toBeCloseTo(-2, 12);
    }
    const [lossShift] = shift(blockOf(plain, 'loss'), blockOf(moved, 'loss'));
    expect(lossShift[0]).toBeCloseTo(1, 12);
    expect(lossShift[2]).toBeCloseTo(-2, 12);
    expect(shift(blockOf(plain, 'LSE'), blockOf(moved, 'LSE'))).toEqual([[0, 0, 0]]);
    expect(blockOf(moved, 'log p').plate!.x - blockOf(plain, 'log p').plate!.x).toBeCloseTo(1, 12);
    expect(blockOf(moved, 'log p').axis!.z - blockOf(plain, 'log p').axis!.z).toBeCloseTo(-2, 12);
    expect(blockOf(moved, 'log p').title.y - blockOf(plain, 'log p').title.y).toBeCloseTo(0.5, 12);
  });
});

describe('drag targets', () => {
  it('writes board blocks to the board at and later stack layers to offsets', () => {
    const { g, h, W, z, L, logp, loss } = build();
    const layout = [board(h, W.T, z), stack(z, L, logp, loss)];
    expect(moveTarget(layout, W.T, g.nodes)).toEqual({ item: layout[0], key: 'at' });
    expect(moveTarget(layout, z, g.nodes)).toEqual({ item: layout[0], key: 'at' });
    expect(moveTarget(layout, L, g.nodes)).toEqual({ item: layout[1], key: 'offsets' });
    expect(moveTarget(layout, logp, g.nodes)).toEqual({ item: layout[1], key: 'offsets' });
  });

  it('writes the first layer of a stack that starts from at, and node templates, to at', () => {
    const { g, z, L, logp, loss } = build();
    const layout = [stack(z, logp), node(loss, { at: [1, 2, 3] }), node(L)];
    expect(moveTarget(layout, z, g.nodes)).toEqual({ item: layout[0], key: 'at' });
    expect(moveTarget(layout, logp, g.nodes)).toEqual({ item: layout[0], key: 'offsets' });
    expect(moveTarget(layout, loss, g.nodes)).toEqual({ item: layout[1], key: 'at' });
    const hidden = g.nodes.filter((n) => n !== z);
    expect(moveTarget(layout, logp, hidden)).toEqual({ item: layout[0], key: 'at' });
  });

  it('reads and writes positions rounded to two decimals without touching the page layout', () => {
    const { g, h, W, z, L, logp, loss } = build();
    const page = [board(h, W.T, z, { at: [1, 0, 0] }), stack(z, L, logp, loss)];
    const layout = editableLayout(page);
    const boardTarget = moveTarget(layout, h, g.nodes)!;
    expect(readPosition(boardTarget, h)).toEqual([1, 0, 0]);
    writePosition(boardTarget, h, [1.23456, 0, -0.004]);
    expect(layout[0].options).toEqual({ at: [1.23, 0, 0] });
    const stackTarget = moveTarget(layout, logp, g.nodes)!;
    expect(readPosition(stackTarget, logp)).toEqual([0, 0, 0]);
    writePosition(stackTarget, logp, [0.5, 0, -1.25]);
    expect(layout[1].options).toEqual({ offsets: { 'log p': [0.5, 0, -1.25] } });
    expect(page[0].options).toEqual({ at: [1, 0, 0] });
    expect(page[1].options).toEqual({});
  });
});

describe('layoutCode', () => {
  it('writes the layout, camera and label offsets as code that evaluates back to the same options', () => {
    const { g, h, W, z, L, logp, loss } = build();
    const layout = editableLayout([
      board(h, W.T, z, { scales: { h: { forward: 0.45 }, 'W.T': { forward: 0.6, backward: 0.8 } } }),
      stack(z, L, logp, { level: 3.9, scalars: 'row' }),
      node(loss, { at: [0, 12, 0] }),
    ]);
    writePosition(moveTarget(layout, h, g.nodes)!, h, [0.5, 0, -1]);
    (layout[0].options as { gap?: number }).gap = 1.2;
    writePosition(moveTarget(layout, logp, g.nodes)!, logp, [1, 0, -0.5]);
    const camera = { direction: [0.5, 0.8, 1], polar: [0.3, 1.45] };
    const labels: Record<string, [number, number]> = { 'log p': [12, -8] };
    const code = layoutCode({ layout, camera, labels });
    expect(code).toBe(
      [
        'layout: [',
        "  board(g.node('h'), g.node('W.T'), g.node('z'), { scales: { h: { forward: 0.45 }, 'W.T': { forward: 0.6, backward: 0.8 } }, at: [0.5, 0, -1], gap: 1.2 }),",
        "  stack(g.node('z'), g.node('LSE'), g.node('log p'), { level: 3.9, scalars: 'row', offsets: { 'log p': [1, 0, -0.5] } }),",
        "  node(g.node('loss'), { at: [0, 12, 0] }),",
        '],',
        'camera: { direction: [0.5, 0.8, 1], polar: [0.3, 1.45] },',
        "labels: { 'log p': [12, -8] },",
      ].join('\n'),
    );
    const parsed = evaluate(code, g);
    expect(parsed.camera).toEqual(camera);
    expect(parsed.labels).toEqual(labels);
    expect(parsed.layout.map((item) => item.template)).toEqual(['board', 'stack', 'node']);
    parsed.layout.forEach((item, i) => {
      expect(item.options).toEqual(layout[i].options);
      expect(item).toEqual(layout[i]);
    });
    expect(layoutCode({ layout: parsed.layout, camera: parsed.camera, labels: parsed.labels as Record<string, [number, number]> })).toBe(code);
  });

  it('leaves out empty options, camera and labels', () => {
    const { g, h, W, z } = build();
    const code = layoutCode({ layout: [board(h, W.T, z)], camera: {}, labels: {} });
    expect(code).toBe("layout: [\n  board(g.node('h'), g.node('W.T'), g.node('z')),\n],");
    expect(evaluate(code, g).layout[0]).toEqual(board(h, W.T, z));
  });

  it('quotes keys and strings that are not identifiers', () => {
    expect(literal({ "it's": [-0, 1.5], ok: 'a\\b' })).toBe("{ 'it\\'s': [0, 1.5], ok: 'a\\\\b' }");
  });
});
