import type { Node } from '../autograd';
import type { CacheDecl } from './cache';
import type { ForwardStep, InferSpec, StepGroup } from './steps';

export const INPUT_TITLE = '输入';
export const OUTPUT_TITLE = '输出';

export interface ModuleTile {
  title: string;
  subtitle?: string;
  names: string[];
  steps: ForwardStep[];
  caches: CacheDecl[];
  children: ModuleTile[];
}

export interface CompactStep {
  steps: ReadonlySet<ForwardStep>;
  row: ReadonlySet<ForwardStep>;
  done: ReadonlySet<ForwardStep>;
  sample: boolean;
}

export interface CompactViewSpec {
  infer: InferSpec;
}

export type TileState = 'lit' | 'done' | 'idle';

interface Part {
  key: string;
  name: string;
  own: boolean;
  steps: ForwardStep[];
}

function shows(steps: readonly ForwardStep[], nodes: readonly Node[]): boolean {
  return steps.some((step) => (Array.isArray(step.show) ? step.show : [step.show]).some((n) => nodes.includes(n)));
}

export function tileLeaves(tiles: readonly ModuleTile[]): ModuleTile[] {
  return tiles.flatMap((tile) => (tile.children.length > 0 ? tile.children : [tile]));
}

export function moduleTiles(steps: readonly ForwardStep[], groups: readonly StepGroup[], caches: readonly CacheDecl[]): ModuleTile[] {
  const byId = new Map(groups.map((group) => [group.id, group]));
  const position = new Map(steps.map((step, i) => [step, i]));
  const loopOf = (step: ForwardStep) => {
    const outer = step.groups?.[0];
    return outer !== undefined && byId.get(outer)?.loop === true ? outer : undefined;
  };
  const partsOf = (run: readonly ForwardStep[], depth: number): Part[] => {
    const parts: Part[] = [];
    for (const step of run) {
      const group = step.groups?.[depth];
      const key = group ?? `step:${position.get(step)}`;
      const last = parts[parts.length - 1];
      if (last?.key === key) last.steps.push(step);
      else parts.push({ key, name: (group === undefined ? undefined : byId.get(group)?.title) ?? step.title, own: group === undefined, steps: [step] });
    }
    return parts;
  };
  const leaf = (parts: Part[], title?: string): ModuleTile => {
    const largest = parts.reduce((best, part) => (part.steps.length > best.steps.length ? part : best));
    return { title: title ?? largest.name, names: parts.map((part) => part.name), steps: parts.flatMap((part) => part.steps), caches: [], children: [] };
  };
  const segments: { loop: string | undefined; steps: ForwardStep[] }[] = [];
  for (const step of steps) {
    const loop = loopOf(step);
    const last = segments[segments.length - 1];
    if (last !== undefined && last.loop === loop) last.steps.push(step);
    else segments.push({ loop, steps: [step] });
  }
  const tiles = segments.map((segment, k): ModuleTile => {
    if (segment.loop === undefined) return leaf(partsOf(segment.steps, 0), k === 0 ? INPUT_TITLE : k === segments.length - 1 ? OUTPUT_TITLE : undefined);
    const parts = partsOf(segment.steps, 1);
    const cuts: Part[][] = [];
    let open: Part[] = [];
    for (const part of parts) {
      open.push(part);
      if (part.own) {
        cuts.push(open);
        open = [];
      }
    }
    if (open.length > 0) cuts.push(open);
    const bodies = parts.some((part) => part.own) ? cuts : parts.map((part) => [part]);
    const id = segment.loop;
    return { title: id.slice(id.lastIndexOf('.') + 1), subtitle: byId.get(id)!.title, names: [], steps: segment.steps, caches: [], children: bodies.map((members) => leaf(members)) };
  });
  const leaves = tileLeaves(tiles);
  for (const decl of caches) {
    const holder = leaves.find((tile) => shows(tile.steps, decl.nodes)) ?? leaves.find((tile) => shows(tile.steps, decl.members));
    holder?.caches.push(decl);
  }
  return tiles;
}

export function tileStates(leaves: readonly ModuleTile[], compact: CompactStep): TileState[] {
  return leaves.map((tile) => {
    const ran = tile.steps.filter((step) => compact.row.has(step));
    if (ran.length === 0) return 'idle';
    const half = ran.length / 2;
    if (ran.filter((step) => compact.steps.has(step)).length > half) return 'lit';
    return ran.filter((step) => compact.done.has(step)).length > half ? 'done' : 'idle';
  });
}
