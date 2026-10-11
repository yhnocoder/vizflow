import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import '../../src/explain';
import {
  AXIS_OFFSET,
  BATCH_ROW_SPACING,
  BOARD_COLUMN_SPACING,
  SCALAR_NODE_RISE,
  SELECT_NODE_RISE,
  SIDE_NODE_OFFSET,
  STACK_BACK_OFFSET,
  STACK_LEVEL,
  type Block,
  board,
  classAlignment,
  elementAt,
  node,
  resolveLayout,
  stack,
  targetColumn,
} from '../../src/scene3d/layout';

function build(targets: number[] = [0], rows = 1) {
  const g = graph({ dims: { d: 4, V: 5 } });
  const hValues = Array.from({ length: rows }, (_, r) => [0.8 + r, -1.1, 0.5, 1.4]);
  const h = g.input('h', [rows, 'd'], hValues);
  const W = g.param('W', ['V', 'd'], [
    [0.9, -0.6, 0.3, 0.6],
    [0.2, -0.8, 0.6, 0.2],
    [-0.4, 0.2, 0.9, 0.3],
    [0.5, 0.7, -0.6, -0.2],
    [-0.7, 0.4, -0.3, -0.5],
  ]);
  const t = g.label('t', targets, { tokens: ['a', 'b', 'c', 'd', 'e'] });
  const z = g.matmul(h, W.T, { name: 'z' });
  const L = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
  const logp = g.sub(z, L, { name: 'log p' });
  const loss = g.nll_loss(logp, t, { name: 'loss' });
  const aligned = classAlignment(g.nodes);
  const ctx = { visible: g.nodes, aligned, targetOf: targetColumn, focus: 0 };
  return { g, h, W, t, z, L, logp, loss, aligned, ctx };
}

const blockOf = (blocks: Block[], name: string) => blocks.find((b) => b.node.name === name)!;

describe('board layout', () => {
  it('places b, out and a as the reference page does', () => {
    const { h, W, z, ctx } = build();
    const { blocks } = resolveLayout([board(h, W.T, z)], ctx);
    const b = blockOf(blocks, 'W.T');
    const out = blockOf(blocks, 'z');
    const a = blockOf(blocks, 'h');
    expect(b.shape).toBe('tile');
    expect(elementAt(b, [0, 0])).toMatchObject({ x: -2 * BOARD_COLUMN_SPACING, y: 0, z: 2.9 - 1.3 - 3 * 0.7 });
    expect(elementAt(b, [3, 4]).x).toBeCloseTo(2 * BOARD_COLUMN_SPACING, 12);
    expect(elementAt(b, [3, 4]).z).toBeCloseTo(1.6, 12);
    expect(elementAt(out, [0, 2])).toMatchObject({ x: 0, y: 0, z: 2.9 });
    expect(elementAt(a, [0, 0]).x).toBeCloseTo(-2.9 - 4.3, 12);
    expect(elementAt(a, [0, 3]).x).toBeCloseTo(-7.2 + 3 * 0.95, 12);
    expect(elementAt(a, [0, 3]).z).toBe(2.9);
    expect(a.footprint).toEqual([0.6, 0.6]);
    expect(b.footprint).toEqual([0.58, 0.4]);
    expect(out.footprint).toEqual([0.62, 0.62]);
    expect(b.plate).toBeDefined();
    expect(a.plate).toBeDefined();
    expect(out.plate).toBeDefined();
  });

  it('applies gap, scale and at', () => {
    const { h, W, z, ctx } = build();
    const { blocks } = resolveLayout([board(h, W.T, z, { gap: 1.2, scale: 2, at: [1, 0, 3] })], ctx);
    const b = blockOf(blocks, 'W.T');
    const out = blockOf(blocks, 'z');
    expect(elementAt(b, [0, 1]).x).toBeCloseTo(1 + -1 * BOARD_COLUMN_SPACING * 2, 12);
    expect(elementAt(b, [2, 0]).z).toBeCloseTo(3 + 2.9 * 2 - 1.3 * 2 - 1 * 1.2 * 2, 12);
    expect(elementAt(out, [0, 0]).z).toBeCloseTo(3 + 2.9 * 2, 12);
  });

  it('rejects options that board does not read', () => {
    const { h, W, z } = build();
    expect(() => board(h, W.T, z, { offsets: { z: [0, 0, 1] } } as never)).toThrow("board: unknown option 'offsets'");
  });

  it('puts batch row 0 of out and a in front and later rows behind it', () => {
    const { h, W, z, ctx } = build([0, 1], 2);
    const { blocks } = resolveLayout([board(h, W.T, z)], ctx);
    expect(elementAt(blockOf(blocks, 'z'), [0, 0]).z).toBeCloseTo(2.9 + BATCH_ROW_SPACING, 12);
    expect(elementAt(blockOf(blocks, 'z'), [1, 0]).z).toBeCloseTo(2.9, 12);
    expect(elementAt(blockOf(blocks, 'h'), [0, 2]).z).toBeCloseTo(2.9 + BATCH_ROW_SPACING, 12);
    expect(elementAt(blockOf(blocks, 'h'), [1, 2]).z).toBeCloseTo(2.9, 12);
    expect(blockOf(blocks, 'h').classLabels[0].z).toBeGreaterThan(2.9 + BATCH_ROW_SPACING);
    expect(blockOf(blocks, 'z').batched).toBe(true);
    expect(blockOf(blocks, 'W.T').batched).toBe(false);
  });

  it('marks class-aligned blocks with a target frame and token labels', () => {
    const { h, W, z, t, aligned, ctx } = build();
    expect(aligned.get(z)).toBe(t);
    expect(aligned.get(W.T)).toBe(t);
    expect(aligned.has(h)).toBe(false);
    const { blocks } = resolveLayout([board(h, W.T, z)], ctx);
    expect(blockOf(blocks, 'W.T').frameHalf).toBeDefined();
    expect(blockOf(blocks, 'z').frameHalf).toBeDefined();
    expect(blockOf(blocks, 'h').frameHalf).toBeUndefined();
    expect(blockOf(blocks, 'W.T').classLabels.map((l) => l.text)).toEqual(['a', 'b', 'c', 'd', 'e']);
    expect(blockOf(blocks, 'W.T').rowLabels.map((l) => l.text)).toEqual(['k=1', 'k=2', 'k=3', 'k=4']);
    expect(blockOf(blocks, 'W.T').rowLabels[0].x).toBeLessThan(blockOf(blocks, 'W.T').plate!.x - blockOf(blocks, 'W.T').plate!.w / 2);
    expect(blockOf(blocks, 'h').classLabels[0].z).toBeGreaterThan(2.9 + 0.3);
    expect(blockOf(blocks, 'h').classLabels.map((l) => l.text)).toEqual(['k=1', 'k=2', 'k=3', 'k=4']);
  });

  it('reuses an input that is already placed and shared by several boards', () => {
    const { g, h, W, z, ctx } = build();
    const U = g.param('U', [3, 'd'], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]);
    const u = g.matmul(h, U.T, { name: 'u' });
    const all = { ...ctx, visible: g.nodes };
    const { blocks, boards } = resolveLayout([stack(h, { at: [-10, 0, 0] }), board(h, W.T, z, { at: [0, 3, 0] }), board(h, U.T, u, { at: [10, 3, 0] })], all);
    expect(blocks.filter((b) => b.node === h)).toHaveLength(1);
    expect(boards.map((b) => b.sharedA)).toEqual([true, true]);
    expect(boards[0].a).toBe(blockOf(blocks, 'h'));
    expect(elementAt(blockOf(blocks, 'h'), [0, 0]).y).toBe(0);
    expect(elementAt(blockOf(blocks, 'u'), [0, 0]).y).toBe(3);
  });
});

describe('stack layout', () => {
  it('starts from the board output and stacks LSE, log p and loss', () => {
    const { h, W, z, L, logp, loss, ctx } = build([2]);
    const { blocks } = resolveLayout([board(h, W.T, z), stack(z, L, logp, loss)], ctx);
    expect(blocks.filter((b) => b.node === z)).toHaveLength(1);
    const side = blockOf(blocks, 'LSE');
    expect(side.shape).toBe('disc');
    expect(side.elements[0]).toMatchObject({ x: 2 * BOARD_COLUMN_SPACING + SIDE_NODE_OFFSET, y: STACK_LEVEL / 2, z: 2.9 - STACK_BACK_OFFSET / 2 });
    const plane = blockOf(blocks, 'log p');
    expect(plane.shape).toBe('bar');
    expect(plane.elements.map((e) => e.x)).toEqual(blockOf(blocks, 'z').elements.map((e) => e.x));
    expect(plane.elements[0].y).toBe(STACK_LEVEL);
    expect(plane.elements[0].z).toBeCloseTo(2.9 - STACK_BACK_OFFSET, 12);
    expect(plane.axis).toEqual({ x: -2 * BOARD_COLUMN_SPACING - AXIS_OFFSET, z: 2.9 - STACK_BACK_OFFSET - 0.5 });
    expect(plane.classLabels[2].text).toBe('c');
    const box = blockOf(blocks, 'loss');
    expect(box.shape).toBe('box');
    expect(box.elements[0]).toMatchObject({ x: 0, y: STACK_LEVEL + SELECT_NODE_RISE });
    expect(box.elements[0].z).toBeCloseTo(2.9 - STACK_BACK_OFFSET, 12);
    expect(box.extent).toHaveLength(5);
  });

  it('moves the select node with the target column', () => {
    const { h, W, z, L, logp, loss, t, ctx } = build([2]);
    const layout = [board(h, W.T, z), stack(z, L, logp, loss)];
    const before = blockOf(resolveLayout(layout, ctx).blocks, 'loss').elements[0].x;
    t.set([4]);
    const after = blockOf(resolveLayout(layout, ctx).blocks, 'loss').elements[0].x;
    expect(before).toBe(0);
    expect(after).toBeCloseTo(2 * BOARD_COLUMN_SPACING, 12);
  });

  it('places a stack without a board around at and honours level', () => {
    const g = graph({ dims: { V: 3 } });
    const x = g.input('x', ['V'], [1, 2, 3]);
    const t = g.label('t', [0], { tokens: ['a', 'b', 'c'] });
    const p = g.softmax(x, -1, { name: 'p' });
    const y = g.gather(p, 0, t, { name: 'y' });
    const logy = g.log(y, { name: 'log y' });
    const l = g.neg(logy, { name: 'loss' });
    const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
    const { blocks } = resolveLayout([stack(x, p, y, logy, l, { at: [1, 0, 2], level: 3 })], ctx);
    expect(blockOf(blocks, 'x').elements.map((e) => e.x)).toEqual([1 - BOARD_COLUMN_SPACING, 1, 1 + BOARD_COLUMN_SPACING]);
    expect(blockOf(blocks, 'x').elements[0].z).toBe(2);
    expect(blockOf(blocks, 'p').elements[0].y).toBe(3);
    expect(blockOf(blocks, 'y')).toMatchObject({ shape: 'disc' });
    expect(blockOf(blocks, 'y').elements[0]).toMatchObject({ x: 1 - BOARD_COLUMN_SPACING, y: 3 + SELECT_NODE_RISE });
    expect(blockOf(blocks, 'log y').elements[0].y).toBeCloseTo(3 + SELECT_NODE_RISE + SCALAR_NODE_RISE, 12);
    expect(blockOf(blocks, 'loss')).toMatchObject({ shape: 'box' });
    expect(blockOf(blocks, 'loss').elements[0].y).toBeCloseTo(3 + SELECT_NODE_RISE + 2 * SCALAR_NODE_RISE, 12);
    expect(blockOf(blocks, 'y').title.anchor).toBe('right');
    expect(blockOf(blocks, 'log y').title.anchor).toBe('right');
    expect(blockOf(blocks, 'loss').title.anchor).toBe('right');
  });

  it('centers a planar layer with a different column count over the previous layer', () => {
    const g = graph({ dims: { d: 4, V: 3 } });
    const x = g.input('x', [1, 'd'], [[1, 2, 3, 4]]);
    const W = g.param('W', ['V', 'd'], [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
    ]);
    const z = g.matmul(x, W.T, { name: 'z' });
    const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
    const { blocks } = resolveLayout([stack(x, z, { at: [0, 0, 0], offsets: { x: [5, 0, 0] } })], ctx);
    expect(blockOf(blocks, 'x').elements.map((e) => e.x)[0]).toBeCloseTo(5 - 1.5 * BOARD_COLUMN_SPACING, 12);
    expect(blockOf(blocks, 'z').elements.map((e) => e.x)).toEqual([5 - BOARD_COLUMN_SPACING, 5, 5 + BOARD_COLUMN_SPACING]);
  });

  it('moves each planar layer and the side node back by the back option', () => {
    const g = graph({ dims: { V: 3 } });
    const x = g.input('x', ['V'], [1, 2, 3]);
    const u = g.exp(x, { name: 'u' });
    const S = g.sum(u, -1, { keepdim: true, name: 'S' });
    const p = g.div(u, S, { name: 'p' });
    const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
    const { blocks } = resolveLayout([stack(x, u, S, p, { at: [0, 0, 2.6], back: 1.7 })], ctx);
    expect(blockOf(blocks, 'x').elements[0].z).toBe(2.6);
    expect(blockOf(blocks, 'u').elements[0].z).toBeCloseTo(0.9, 12);
    expect(blockOf(blocks, 'S').elements[0].z).toBeCloseTo(0.05, 12);
    expect(blockOf(blocks, 'p').elements[0].z).toBeCloseTo(-0.8, 12);
  });

  it('keeps each batch row of a select chain on its own target column and centers the collapsed loss', () => {
    const g = graph({ dims: { V: 3 } });
    const x = g.input('x', [2, 'V'], [
      [1, 2, 3],
      [3, 1, 2],
    ]);
    const t = g.label('t', [[2], [0]], { classes: 3 });
    const p = g.softmax(x, -1, { name: 'p' });
    const y = g.gather(p, 1, t, { name: 'y' });
    const logy = g.log(y, { name: 'log y' });
    const lb = g.neg(logy, { name: 'l_b' });
    const loss = g.mean(lb, { name: 'loss' });
    const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
    const layout = [stack(x, p, y, logy, lb, loss)];
    const columns = (name: string) => blockOf(resolveLayout(layout, ctx).blocks, name).elements.map((e) => e.x);
    const xs = blockOf(resolveLayout(layout, ctx).blocks, 'p').elements.filter((e) => e.row === 0).map((e) => e.x);
    for (const name of ['y', 'log y', 'l_b']) expect(columns(name)).toEqual([xs[2], xs[0]]);
    expect(columns('loss')).toEqual([xs[1]]);
    t.set([[1], [2]]);
    for (const name of ['y', 'log y', 'l_b']) expect(columns(name)).toEqual([xs[1], xs[2]]);
    expect(blockOf(resolveLayout(layout, ctx).blocks, 'l_b').extent.map((p) => p[0])).toEqual(xs);
    for (const name of ['y', 'log y', 'l_b', 'loss']) expect(blockOf(resolveLayout(layout, ctx).blocks, name).title.anchor).toBe('left');
  });

  it('skips nodes that are not visible and reads scales from options', () => {
    const { h, W, z, L, logp, loss, aligned } = build();
    const ctx = { visible: [h, W.T, z, L], aligned, targetOf: targetColumn, focus: 0 };
    const { blocks } = resolveLayout([board(h, W.T, z), stack(z, L, logp, loss, { scales: { LSE: { forward: 2 } } })], ctx);
    expect(blocks.map((b) => b.node.name)).toEqual(['h', 'W.T', 'z', 'LSE']);
    expect(blockOf(blocks, 'LSE').scales).toEqual({ forward: 2 });
  });

  it('places an explicit node at the given position', () => {
    const { h, W, z, L, ctx } = build();
    const { blocks } = resolveLayout([board(h, W.T, z), node(L, { at: [4, 1, 2] })], ctx);
    expect(blockOf(blocks, 'LSE').elements[0]).toMatchObject({ x: 4, y: 1, z: 2 });
  });

  it('rejects tensors with more than two dimensions', () => {
    const g = graph();
    const x = g.input('x', [2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    const ctx = { visible: g.nodes, aligned: new Map(), targetOf: targetColumn, focus: 0 };
    expect(() => resolveLayout([stack(x)], ctx)).toThrow(/x.*\(2, 2, 2\)/);
  });
});

describe('targetColumn', () => {
  it('reads the label row for scalar, vector and matrix labels', () => {
    const g = graph();
    expect(targetColumn(g.label('a', 2), 0)).toBe(2);
    expect(targetColumn(g.label('b', [1]), 0)).toBe(1);
    expect(targetColumn(g.label('c', [1, 3]), 1)).toBe(3);
    expect(targetColumn(g.label('d', [[0], [4]]), 1)).toBe(4);
  });

  it('rejects the node template for tensors with more than one element per row', () => {
    const g = graph();
    const x = g.input('x', [3], [1, 2, 3]);
    expect(() => node(x)).toThrow("node: 'x' has shape (3); use stack for tensors with more than one element per row");
    expect(node(g.input('s', [], 1)).template).toBe('node');
    expect(node(g.input('b', [2, 1], [1, 2])).template).toBe('node');
  });
});
