import type { Node } from '../autograd';
import type { LevelGrid } from './levels';
import type { SceneModule } from './modules';

export interface CopyRef {
  grid: LevelGrid;
  index: readonly number[];
}

export interface MemberState {
  active(node: Node): boolean;
  revealed(node: Node): boolean;
}

export function copiesOf(grids: readonly LevelGrid[]): Map<Node, CopyRef> {
  const out = new Map<Node, CopyRef>();
  for (const grid of grids) for (const [node, index] of grid.copyOf) out.set(node, { grid, index });
  return out;
}

export function sameStack(grid: LevelGrid, a: readonly number[], b: readonly number[]): boolean {
  return grid.arrangement.every((arrangement, d) => arrangement === 'expand' || a[d] === b[d]);
}

export function sameCopy(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, d) => v === b[d]);
}

export function shownMember(members: readonly Node[], inFront: (node: Node) => boolean, state: MemberState): Node | undefined {
  const own = members.filter(inFront);
  if (own.length === 0) return undefined;
  return last(own, (n) => state.active(n)) ?? last(own, (n) => state.revealed(n)) ?? own[0];
}

function last<T>(items: readonly T[], test: (item: T) => boolean): T | undefined {
  for (let i = items.length - 1; i >= 0; i--) if (test(items[i])) return items[i];
  return undefined;
}


export function copyIndices(counts: readonly number[]): number[][] {
  let out: number[][] = [[]];
  for (const count of counts) out = out.flatMap((prefix) => Array.from({ length: count }, (_, k) => [...prefix, k]));
  return out;
}

export function busyCopies(grid: LevelGrid, active: (node: Node) => boolean): number[][] {
  if (grid.names.length === 0) return [...grid.shared, ...grid.registers].some(active) ? [[]] : [];
  const busy = new Map<string, number[]>();
  for (const [node, index] of grid.copyOf) {
    const key = index.join(',');
    if (!busy.has(key) && active(node)) busy.set(key, [...index]);
  }
  return [...busy.values()];
}

export function moduleHeading(module: SceneModule, state: MemberState, front: (grid: string) => readonly number[] | undefined): { title: string; subtitle?: string } {
  const rounds = module.rounds;
  if (rounds === undefined) return module;
  const round = last(rounds, (r) => r.nodes.some((n) => state.active(n))) ?? last(rounds, (r) => r.nodes.some((n) => state.revealed(n))) ?? rounds[0];
  const index = round.grid === undefined ? undefined : front(round.grid);
  const copy = index === undefined ? undefined : round.copies?.find((c) => sameCopy(c.index, index));
  if (copy !== undefined) return { title: copy.title, subtitle: copy.subtitle ?? round.subtitle };
  return round;
}

export function currentGrid(grids: readonly LevelGrid[], state: MemberState): LevelGrid {
  const touched = (grid: LevelGrid, test: (node: Node) => boolean) => [...grid.copyOf.keys()].some(test);
  return last(grids, (g) => touched(g, (n) => state.active(n))) ?? last(grids, (g) => touched(g, (n) => state.revealed(n))) ?? grids[0];
}
