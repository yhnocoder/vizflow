import type { Node } from '../autograd';
import type { CacheDecl } from './cache';
import type { ForwardStep, InferSpec, StepGroup } from './steps';

export interface ModuleTile {
  id: string;
  title: string;
  subtitle?: string;
  steps: ForwardStep[];
  caches: CacheDecl[];
  children: ModuleTile[];
}

export interface CompactStep {
  steps: ReadonlySet<ForwardStep>;
  done: ReadonlySet<ForwardStep>;
  sample: boolean;
}

export interface CompactViewSpec {
  infer: InferSpec;
}

export type TileState = 'lit' | 'done' | 'idle';

function shownBy(step: ForwardStep): Node[] {
  return Array.isArray(step.show) ? step.show : [step.show];
}

function tileAt(list: ModuleTile[], id: string, make: () => Omit<ModuleTile, 'steps' | 'caches' | 'children'>): ModuleTile {
  const last = list[list.length - 1];
  if (last?.id === id) return last;
  const tile: ModuleTile = { ...make(), steps: [], caches: [], children: [] };
  list.push(tile);
  return tile;
}

export function tileLeaves(tiles: readonly ModuleTile[]): ModuleTile[] {
  return tiles.flatMap((tile) => (tile.children.length > 0 ? tile.children : [tile]));
}

export function moduleTiles(steps: readonly ForwardStep[], groups: readonly StepGroup[], caches: readonly CacheDecl[]): ModuleTile[] {
  const byId = new Map(groups.map((group) => [group.id, group]));
  const stepTile = (step: ForwardStep, index: number) => ({ id: `step:${index}`, title: step.title });
  const groupTile = (id: string) => {
    const group = byId.get(id);
    const title = group?.title ?? id;
    if (group?.loop !== true) return { id, title };
    return { id, title: id.slice(id.lastIndexOf('.') + 1), subtitle: title };
  };
  const tiles: ModuleTile[] = [];
  steps.forEach((step, index) => {
    const [outer, inner] = step.groups ?? [];
    const top = tileAt(tiles, outer ?? `step:${index}`, () => (outer === undefined ? stepTile(step, index) : groupTile(outer)));
    top.steps.push(step);
    if (outer === undefined || byId.get(outer)?.loop !== true) return;
    tileAt(top.children, inner ?? `step:${index}`, () => (inner === undefined ? stepTile(step, index) : groupTile(inner))).steps.push(step);
  });
  const leaves = tileLeaves(tiles);
  for (const decl of caches) {
    const holds = (nodes: Node[]) => leaves.find((tile) => tile.steps.some((step) => shownBy(step).some((n) => nodes.includes(n))));
    (holds(decl.nodes) ?? holds(decl.members))?.caches.push(decl);
  }
  return tiles;
}

export function tileStates(leaves: readonly ModuleTile[], compact: CompactStep): TileState[] {
  return leaves.map((tile) => {
    if (tile.steps.some((step) => compact.steps.has(step))) return 'lit';
    return tile.steps.some((step) => compact.done.has(step)) ? 'done' : 'idle';
  });
}
