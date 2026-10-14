export const AVOID_STEP = 4;
export const AVOID_MAX_STEPS = 12;

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Placement {
  box: Box;
  movable: boolean;
  hideOnOverlap: boolean;
  priority: number;
}

export interface Adjustment {
  dx: number;
  dy: number;
  hidden: boolean;
}

export function intersects(a: Box, b: Box): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

const DIRECTIONS: [number, number][] = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

function shifted(box: Box, dx: number, dy: number): Box {
  return { x: box.x + dx, y: box.y + dy, w: box.w, h: box.h };
}

function inside(box: Box, bounds: Box | undefined): boolean {
  return bounds === undefined || (box.x >= bounds.x && box.y >= bounds.y && box.x + box.w <= bounds.x + bounds.w && box.y + box.h <= bounds.y + bounds.h);
}

export function avoidOverlaps(items: Placement[], bounds?: Box): Adjustment[] {
  const order = items.map((_, i) => i).sort((a, b) => items[b].priority - items[a].priority || a - b);
  const placed: { box: Box }[] = [];
  const result: Adjustment[] = items.map(() => ({ dx: 0, dy: 0, hidden: false }));
  const free = (box: Box) => inside(box, bounds) && placed.every((other) => !intersects(box, other.box));
  for (const i of order) {
    const item = items[i];
    if (item.movable) {
      let found: [number, number] | undefined = free(item.box) ? [0, 0] : undefined;
      for (let step = 1; found === undefined && step <= AVOID_MAX_STEPS; step++) {
        for (const [ux, uy] of DIRECTIONS) {
          const dx = ux * step * AVOID_STEP;
          const dy = uy * step * AVOID_STEP;
          if (free(shifted(item.box, dx, dy))) {
            found = [dx, dy];
            break;
          }
        }
      }
      const [dx, dy] = found ?? [0, 0];
      result[i] = { dx, dy, hidden: false };
      placed.push({ box: shifted(item.box, dx, dy) });
    } else if (item.hideOnOverlap) {
      if (free(item.box)) placed.push({ box: item.box });
      else result[i].hidden = true;
    } else {
      placed.push({ box: item.box });
    }
  }
  return result;
}

export function spreadLane(boxes: Box[], top: number, bottom: number, gap: number): void {
  const order = [...boxes].sort((a, b) => a.y + a.h / 2 - (b.y + b.h / 2));
  let end = top - gap;
  for (const box of order) {
    box.y = Math.max(box.y, end + gap);
    end = box.y + box.h;
  }
  let start = bottom + gap;
  for (let k = order.length - 1; k >= 0; k--) {
    const box = order[k];
    box.y = Math.min(box.y, start - gap - box.h);
    start = box.y;
  }
  end = top - gap;
  for (const box of order) {
    box.y = Math.max(box.y, end + gap);
    end = box.y + box.h;
  }
}
