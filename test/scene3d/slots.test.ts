import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { graph } from '../../src/core';
import { copyTotal, frontIn, stackedCopies } from '../../src/scene3d/kernels';
import type { LevelGrid } from '../../src/scene3d/levels';
import type { SceneModule } from '../../src/scene3d/modules';
import { busyCopies, copyIndices, currentGrid, moduleHeading, sameStack, shownMember } from '../../src/scene3d/slots';
import { copyName, copyNote } from '../../src/steps';

function nodes(names: string[]): Node[] {
  const g = graph({ dims: {} });
  return names.map((name) => g.input(name, [2], [1, 2]));
}

function stateOf(active: Node[], revealed: Node[]) {
  return { active: (n: Node) => active.includes(n), revealed: (n: Node) => revealed.includes(n) || active.includes(n) };
}

function gridOf(copyOf: Map<Node, number[]>, extra: Partial<LevelGrid> = {}): LevelGrid {
  return { module: 'g', names: ['m', 'h'], counts: [2, 2], real: [2, 2], arrangement: ['stack', 'stack'], copyOf, shared: new Set(), registers: new Set(), inputs: [], outputs: [], rows: [], ...extra };
}

describe('shownMember', () => {
  const [r0, r1, r2] = nodes(['acc[0]', 'acc[1]', 'acc[2]']);
  const all = () => true;

  it('shows the round the step lights', () => {
    expect(shownMember([r0, r1, r2], all, stateOf([r1], [r0]))).toBe(r1);
  });

  it('shows the last round the copy has reached when no round is lit', () => {
    expect(shownMember([r0, r1, r2], all, stateOf([], [r0, r1]))).toBe(r1);
  });

  it('shows the first round before any round appears', () => {
    expect(shownMember([r0, r1, r2], all, stateOf([], []))).toBe(r0);
  });

  it('keeps to the members of the front copy', () => {
    const [front0, front1, back0, back1] = nodes(['a[0][1]', 'a[1][1]', 'a[0][0]', 'a[1][0]']);
    const inFront = (n: Node) => n === front0 || n === front1;
    expect(shownMember([back0, front0, back1, front1], inFront, stateOf([back1], [back0, front0]))).toBe(front0);
  });

  it('shows nothing when the front copy has no member in the slot', () => {
    const [a, b] = nodes(['x[0][0]', 'x[1][0]']);
    expect(shownMember([a, b], () => false, stateOf([a], [a, b]))).toBeUndefined();
  });
});

describe('stacked copies', () => {
  const [a, b, c, d] = nodes(['t[0][0]', 't[0][1]', 't[1][0]', 't[1][1]']);
  const copyOf = new Map([
    [a, [0, 0]],
    [b, [0, 1]],
    [c, [1, 0]],
    [d, [1, 1]],
  ]);

  it('lists every copy in grid order and finds the busy ones from the lit nodes', () => {
    const grid = gridOf(copyOf);
    expect(copyIndices(grid.counts)).toEqual([[0, 0], [0, 1], [1, 0], [1, 1]]);
    expect(busyCopies(grid, (n) => n === b || n === d)).toEqual([[0, 1], [1, 1]]);
    expect(busyCopies(grid, () => false)).toEqual([]);
    const single = gridOf(new Map(), { names: [], counts: [], real: [], arrangement: [], registers: new Set([a]) });
    expect(busyCopies(single, (n) => n === a)).toEqual([[]]);
    expect(busyCopies(single, () => false)).toEqual([]);
  });

  it('draws every stacked copy behind the front copy and keeps expanded copies apart', () => {
    const grid = gridOf(copyOf);
    expect(stackedCopies(grid, [1, 1], [])).toEqual([[0, 0], [0, 1], [1, 0]]);
    const mixed = gridOf(copyOf, { arrangement: ['expand', 'stack'] });
    expect(frontIn(mixed, [1, 1], [0])).toEqual([0, 1]);
    expect(stackedCopies(mixed, [0, 1], [0])).toEqual([[0, 0]]);
    expect(sameStack(mixed, [0, 1], [1, 1])).toBe(true);
    expect(sameStack(mixed, [0, 0], [1, 1])).toBe(false);
  });

  it('counts the copies at toy size and at real size', () => {
    const grid = gridOf(copyOf, { real: ['M', 4] });
    expect(copyTotal(grid, undefined, [])).toEqual({ drawn: 4, real: undefined });
    expect(copyTotal(grid, { M: 512 }, [])).toEqual({ drawn: 4, real: 2048 });
  });

  it('picks the call whose copies the step lights among calls that share blocks', () => {
    const first = gridOf(new Map([[a, [0, 0]]]), { module: 'k' });
    const second = gridOf(new Map([[b, [0, 0]]]), { module: 'k#2' });
    expect(currentGrid([first, second], stateOf([a], [a, b]))).toBe(first);
    expect(currentGrid([first, second], stateOf([], [a, b]))).toBe(second);
    expect(currentGrid([first, second], stateOf([], []))).toBe(first);
  });

  it('names a copy in grid order and notes when the front copy is idle', () => {
    expect(copyName({ names: ['m', 'h'] }, [2, 1])).toBe('m=2 · h=1');
    expect(copyNote({ names: ['h', 'm'] }, { index: [0, 0], front: false })).toBe('这一步按 h=0 · m=0 说明，最前面的副本这一步空闲');
    expect(copyNote({ names: ['h', 'm'] }, { index: [0, 0], front: true })).toBeUndefined();
  });
});

describe('moduleHeading', () => {
  const [r0, r1] = nodes(['s[0]', 's[1]']);
  const module: SceneModule = {
    title: 'round 1',
    nodes: [r0, r1],
    rounds: [
      { title: '第 1 轮', subtitle: 'a', nodes: [r0] },
      { title: '第 2 轮', nodes: [r1], grid: 'g', copies: [{ index: [0], title: '第 2 轮：m=0' }, { index: [1], title: '第 2 轮：m=1', subtitle: 'b' }] },
    ],
  };

  it('takes the lit round, else the last revealed round, else the first', () => {
    expect(moduleHeading(module, stateOf([r0], [r0, r1]), () => undefined).title).toBe('第 1 轮');
    expect(moduleHeading(module, stateOf([], []), () => undefined)).toMatchObject({ title: '第 1 轮', subtitle: 'a' });
    expect(moduleHeading(module, stateOf([], [r0, r1]), () => undefined).title).toBe('第 2 轮');
  });

  it('takes the title of the front copy', () => {
    expect(moduleHeading(module, stateOf([r1], [r1]), () => [1])).toEqual({ title: '第 2 轮：m=1', subtitle: 'b' });
    expect(moduleHeading(module, stateOf([r1], [r1]), () => [0]).title).toBe('第 2 轮：m=0');
  });
});
