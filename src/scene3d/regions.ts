import type { Node } from '../autograd';
import type { ScatterTilesAttrs } from '../autograd/ops/index_ops';
import { explainOf } from '../explain';
import { forEachIndex } from '../tensor';
import type { ThemeColors } from '../theme';
import type { SceneController } from './controller';
import type { Block, Element } from './layout';
import { FRAME_HALF } from './levels';
import { LineBatch, type Polyline } from './lines';
import { type Renderable, elementOf, renderableIn, renderableOf } from './renderable';
import { sameStack } from './slots';

export const REGION_LIFT = 0.02;
export const REGION_OPACITY = 0.9;
export const REGION_OTHER_OPACITY = 0.6;

export interface Taken {
  source: Node;
  indices: number[][];
}

const MAX_HOPS = 32;

export function takenFrom(target: Node, drawn: (n: Node) => boolean): Taken | undefined {
  const keys = new Set<string>();
  let indices: number[][] = [];
  forEachIndex(target.shape, (index) => indices.push([...index]));
  let n = target;
  for (let hop = 0; hop < MAX_HOPS; hop++) {
    if (n !== target && drawn(n)) {
      const size = n.shape.reduce((p, d) => p * d, 1);
      return indices.length < size ? { source: n, indices } : undefined;
    }
    if (n.kind !== 'op' || n.inputs.length === 0) return undefined;
    const def = explainOf(n.op as string);
    if (def?.selected === undefined) return undefined;
    const ctx = { node: n, inputs: n.inputs.map((i) => i.value), out: n.value, attrs: n.attrs };
    const next: number[][] = [];
    keys.clear();
    for (const index of indices) {
      const picks = def.selected(ctx, index);
      if (picks.length !== 1 || picks[0].arg !== 0) return undefined;
      const key = picks[0].index.join(',');
      if (keys.has(key)) continue;
      keys.add(key);
      next.push([...picks[0].index]);
    }
    indices = next;
    n = n.inputs[0];
  }
  return undefined;
}

export function tileRegions(n: Node): { tile: Node; indices: number[][] }[] {
  if (n.kind !== 'op' || n.op !== 'scatter_tiles') return [];
  const { regions } = n.attrs as unknown as ScatterTilesAttrs;
  return regions.map((region, k) => {
    const indices: number[][] = [];
    forEachIndex(n.shape, (index) => {
      if (region.every((entry, d) => entry === null || (typeof entry === 'number' ? index[d] === entry : index[d] >= entry[0] && index[d] < entry[1]))) indices.push([...index]);
    });
    return { tile: n.inputs[k + 1], indices };
  });
}

export function frameRects(block: Block, elements: readonly Element[]): number[][] {
  if (elements.length === 0) return [];
  const [hx, hz] = FRAME_HALF;
  const xs = elements.map((e) => e.x);
  const zs = elements.map((e) => e.z);
  const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
  const y = elements[0].y + REGION_LIFT;
  const rect = (a0: number, a1: number, b0: number, b1: number) => [a0 - hx, y, b0 - hz, a1 + hx, y, b0 - hz, a1 + hx, y, b1 + hz, a0 - hx, y, b1 + hz, a0 - hx, y, b0 - hz];
  const eps = 1e-6;
  const inside = block.elements.filter((e) => e.x >= x0 - eps && e.x <= x1 + eps && e.z >= z0 - eps && e.z <= z1 + eps).length;
  if (inside === new Set(elements).size) return [rect(x0, x1, z0, z1)];
  return [...new Set(elements)].map((e) => rect(e.x, e.x, e.z, e.z));
}

export class RegionFrames {
  private readonly lines: LineBatch;
  private readonly taken = new Map<Node, Taken | null>();
  private known: object | undefined;

  constructor(private readonly c: SceneController) {
    this.lines = new LineBatch(c.theme.pos);
    c.scene.add(this.lines.object);
  }

  private takenOf(n: Node): Taken | undefined {
    const c = this.c;
    if (this.known !== c.levelLayout) {
      this.taken.clear();
      this.known = c.levelLayout;
    }
    const known = this.taken.get(n);
    if (known !== undefined) return known ?? undefined;
    const views = c.levelLayout?.sourceOf;
    const found = takenFrom(n, (m) => views?.has(m) !== true && renderableOf(c, m) !== undefined);
    this.taken.set(n, found ?? null);
    return found;
  }

  private inFront(n: Node): boolean {
    const c = this.c;
    const copy = c.copyOf(n);
    return copy === undefined || sameStack(copy.grid, copy.index, c.frontOf(copy.grid));
  }

  update(): void {
    const c = this.c;
    const polylines: Polyline[] = [];
    const push = (r: Renderable, source: Node, indices: readonly number[][], front: boolean) => {
      if (r.node !== source || r.far) return;
      const elements = indices.map((index) => elementOf(r.block, source, index));
      for (const points of frameRects(r.block, elements)) polylines.push({ points, opacity: front ? REGION_OPACITY : REGION_OTHER_OPACITY, dashed: !front });
    };
    if (!c.lod) {
      for (const n of c.stepNodes) {
        if (!c.state.active(n)) continue;
        const taken = this.takenOf(n);
        const sourceR = taken === undefined ? undefined : renderableIn(c, taken.source, n);
        if (taken !== undefined && sourceR !== undefined) push(sourceR, taken.source, taken.indices, this.inFront(n));
        const tiles = tileRegions(n);
        const outR = tiles.length === 0 ? undefined : renderableOf(c, n);
        if (outR !== undefined) for (const { tile, indices } of tiles) push(outR, n, indices, this.inFront(tile));
      }
    }
    this.lines.setPolylines(polylines);
  }

  setTheme(theme: ThemeColors): void {
    this.lines.setColor(theme.pos);
  }

  dispose(): void {
    this.lines.dispose();
  }
}
