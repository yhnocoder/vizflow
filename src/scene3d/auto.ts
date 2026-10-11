import type { Node } from '../autograd';
import {
  AXIS_OFFSET,
  BOARD_A_OFFSET,
  BOARD_COLUMN_SPACING,
  BATCH_ROW_SPACING,
  BOARD_FRONT,
  DISC_RADIUS,
  type LayoutItem,
  PLATE_SIDE_MARGIN,
  SCALAR_ROW_GAP,
  SIDE_NODE_OFFSET,
  board,
  rowsCols,
  stack,
} from './layout';
import type { SceneModule } from './modules';

export type LayoutEntry = { kind: 'node'; node: Node } | LayoutTree | { kind: 'parallel'; branches: LayoutTree[] };
export interface LayoutTree { kind: 'module'; title?: string; subtitle?: string; children: LayoutEntry[] }

export const AUTO_LEVEL = 4.2;
export const AUTO_LABEL_ROOM = 1.5;
export const AUTO_BOARD_GAP = 1.0;
export const AUTO_LANE_GAP = 1.0;
export const AUTO_SIDE_GAP = 1.5;
export const AUTO_TITLE_JOIN = ' · ';

type Step = { kind: 'node'; node: Node } | { kind: 'parallel'; branches: Step[][] };

interface Span {
  level: number;
  left: number;
  right: number;
}

interface Cursor {
  level: number;
  anchor?: Node;
}

class Sink {
  readonly items: LayoutItem[] = [];
  readonly spans: Span[] = [];
  readonly levels = new Map<Node, number>();
  readonly centers = new Map<Node, number>();
  readonly boardB = new Map<Node, Node>();

  constructor(readonly placed: Set<Node>) {}

  fork(): Sink {
    return new Sink(new Set(this.placed));
  }

  place(n: Node, level: number, center: number): void {
    this.placed.add(n);
    this.levels.set(n, level);
    this.centers.set(n, center);
  }
}

const colsOf = (n: Node) => rowsCols(n.shape)[1];
const halfSpan = (cols: number) => ((cols - 1) / 2) * BOARD_COLUMN_SPACING;
const leftReach = (cols: number) => halfSpan(cols) + AXIS_OFFSET + AUTO_LABEL_ROOM;
const rightReach = (cols: number) => halfSpan(cols) + PLATE_SIDE_MARGIN;
const flow = (at: [number, number, number]) => ({ at, level: AUTO_LEVEL, back: 0, scalars: 'row' as const });

function treeNodes(entries: readonly LayoutEntry[]): Node[] {
  return entries.flatMap((entry) => {
    if (entry.kind === 'node') return [entry.node];
    if (entry.kind === 'parallel') return entry.branches.flatMap((branch) => treeNodes(branch.children));
    return treeNodes(entry.children);
  });
}

function boardOperands(n: Node): [Node, Node] | undefined {
  if (n.kind !== 'op' || n.op !== 'matmul' || n.inputs.length !== 2) return undefined;
  const [a, b] = n.inputs;
  return b.shape.length === 2 && n.shape.length <= 2 ? [a, b] : undefined;
}

function boardReach(b: Node, aPlaced: boolean): [number, number] {
  const cols = b.shape[1];
  const left = halfSpan(cols) + PLATE_SIDE_MARGIN + AUTO_LABEL_ROOM;
  const aLeft = halfSpan(cols) + BOARD_A_OFFSET + PLATE_SIDE_MARGIN;
  return [aPlaced ? left : Math.max(left, aLeft), rightReach(cols)];
}

export function autoLayout(tree: LayoutTree): { layout: LayoutItem[]; modules: SceneModule[] } {
  const all = treeNodes(tree.children);
  const boardB = new Set(all.flatMap((n) => boardOperands(n)?.[1] ?? []));
  const direct = new Set(all.filter((n) => !boardB.has(n)).flatMap((n) => n.inputs));
  const isSide = (n: Node) => n.kind === 'buffer' || (n.kind === 'param' && n.shape.length < 2);
  const skipped = (n: Node) => n.kind === 'label' || boardB.has(n) || (n.kind === 'param' && !isSide(n) && !direct.has(n));
  const side: Node[] = [];

  const stepsOf = (entries: readonly LayoutEntry[]): Step[] =>
    entries.flatMap((entry): Step[] => {
      if (entry.kind === 'parallel') return [{ kind: 'parallel', branches: entry.branches.map((branch) => stepsOf(branch.children)) }];
      if (entry.kind === 'module') return stepsOf(entry.children);
      if (isSide(entry.node)) {
        if (!side.includes(entry.node)) side.push(entry.node);
        return [];
      }
      return skipped(entry.node) ? [] : [{ kind: 'node', node: entry.node }];
    });

  const asBoard = (step: Step, sink: Sink) => {
    if (step.kind !== 'node') return undefined;
    const operands = boardOperands(step.node);
    return operands !== undefined && !sink.placed.has(operands[1]) ? operands : undefined;
  };

  function flushRun(run: Node[], x: number, cursor: Cursor, sink: Sink): void {
    if (run.length === 0) return;
    const anchored = colsOf(run[0]) <= 1 && cursor.anchor !== undefined;
    const center = anchored ? (sink.centers.get(cursor.anchor!) ?? x) : x;
    const nodes = anchored ? [cursor.anchor!, ...run] : run;
    sink.items.push(stack(...nodes, flow([x, (cursor.level + 1) * AUTO_LEVEL, 0])));
    let rowIndex = 0;
    for (const n of run) {
      const cols = colsOf(n);
      if (cols > 1) {
        cursor.level++;
        cursor.anchor = n;
        rowIndex = 0;
        sink.spans.push({ level: cursor.level, left: center - leftReach(cols), right: center + rightReach(cols) });
      } else {
        const host = cursor.anchor === undefined ? 1 : colsOf(cursor.anchor);
        const right = center + halfSpan(host) + SIDE_NODE_OFFSET + rowIndex * SCALAR_ROW_GAP + DISC_RADIUS + AUTO_LABEL_ROOM;
        sink.spans.push({ level: cursor.level, left: center - leftReach(host), right });
        rowIndex++;
      }
      sink.place(n, cursor.level, center);
    }
    run.length = 0;
  }

  function placeBoards(group: [Node, Node, Node][], x: number, cursor: Cursor, sink: Sink): void {
    const level = cursor.level + 1;
    const reaches = group.map(([a, b]) => boardReach(b, sink.placed.has(a)));
    const width = reaches.reduce((sum, [l, r]) => sum + l + r, 0) + AUTO_BOARD_GAP * (group.length - 1);
    let edge = x - width / 2;
    group.forEach(([a, b, out], i) => {
      const [l, r] = reaches[i];
      const ax = group.length === 1 ? x : edge + l;
      edge += l + r + AUTO_BOARD_GAP;
      const rows = rowsCols(out.shape)[0];
      sink.items.push(board(a, b, out, { at: [ax, level * AUTO_LEVEL, -(BOARD_FRONT + (rows - 1) * BATCH_ROW_SPACING)] }));
      sink.spans.push({ level, left: ax - l, right: ax + r });
      if (!sink.placed.has(a)) sink.place(a, level, ax - BOARD_A_OFFSET);
      sink.place(b, level, ax);
      sink.place(out, level, ax);
      sink.boardB.set(out, b);
      cursor.anchor = out;
    });
    cursor.level = level;
  }

  function placeSteps(steps: Step[], x: number, start: Cursor, sink: Sink): Cursor {
    const cursor: Cursor = { ...start };
    const run: Node[] = [];
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      if (step.kind === 'parallel') {
        flushRun(run, x, cursor, sink);
        const extents = step.branches.map((branch) => {
          const probe = sink.fork();
          placeSteps(branch, 0, { level: cursor.level }, probe);
          return probe.spans.length === 0 ? [0, 0] : [Math.min(...probe.spans.map((s) => s.left)), Math.max(...probe.spans.map((s) => s.right))];
        });
        const width = extents.reduce((sum, [l, r]) => sum + r - l, 0) + AUTO_LANE_GAP * (extents.length - 1);
        let edge = x - width / 2;
        let top = cursor.level;
        step.branches.forEach((branch, b) => {
          const [l, r] = extents[b];
          top = Math.max(top, placeSteps(branch, edge - l, { level: cursor.level }, sink).level);
          edge += r - l + AUTO_LANE_GAP;
        });
        cursor.level = top;
        cursor.anchor = undefined;
        continue;
      }
      if (sink.placed.has(step.node)) continue;
      const operands = asBoard(step, sink);
      if (operands === undefined) {
        run.push(step.node);
        continue;
      }
      flushRun(run, x, cursor, sink);
      const group: [Node, Node, Node][] = [[...operands, step.node]];
      while (i + 1 < steps.length) {
        const next = steps[i + 1];
        const more = asBoard(next, sink);
        if (more === undefined || more[0] !== operands[0] || next.kind !== 'node') break;
        group.push([...more, next.node]);
        i++;
      }
      placeBoards(group, x, cursor, sink);
    }
    flushRun(run, x, cursor, sink);
    return cursor;
  }

  const sink = new Sink(new Set());
  placeSteps(stepsOf(tree.children), 0, { level: -1 }, sink);

  const order = new Map(all.map((n, i) => [n, i]));
  for (const n of side) {
    const users = all.filter((user) => user.inputs.includes(n) && sink.levels.has(user)).sort((p, q) => order.get(p)! - order.get(q)!);
    const level = users.length === 0 ? 0 : sink.levels.get(users[0])!;
    const cols = colsOf(n);
    const right = Math.max(0, ...sink.spans.filter((s) => s.level === level).map((s) => s.right));
    const x = right + AUTO_SIDE_GAP + leftReach(cols);
    sink.items.push(stack(n, flow([x, level * AUTO_LEVEL, 0])));
    sink.spans.push({ level, left: x - leftReach(cols), right: x + rightReach(cols) });
    sink.place(n, level, x);
  }

  const moduleOf = (sub: LayoutTree, depth: number): SceneModule[] => {
    const titled = sub.title !== undefined && sub.title !== '';
    const inner = modulesOf(sub.children, titled ? depth + 1 : depth);
    if (!titled) return inner;
    const own = treeNodes(sub.children);
    const nodes = [...own, ...own.flatMap((n) => sink.boardB.get(n) ?? [])];
    return [{ title: sub.title!, subtitle: sub.subtitle, nodes, side: depth === 0 ? 'left' : 'right' }, ...inner];
  };
  const modulesOf = (entries: readonly LayoutEntry[], depth: number): SceneModule[] =>
    entries.flatMap((entry) => {
      if (entry.kind === 'node') return [];
      if (entry.kind === 'module') return moduleOf(entry, depth);
      const merged: { module: SceneModule; titles: string[]; subtitles: string[] }[] = [];
      for (const branch of entry.branches) {
        moduleOf(branch, depth).forEach((module, position) => {
          const slot = merged[position];
          if (slot === undefined) {
            merged[position] = { module, titles: [module.title], subtitles: module.subtitle === undefined ? [] : [module.subtitle] };
            return;
          }
          slot.module.nodes.push(...module.nodes);
          if (!slot.titles.includes(module.title)) slot.titles.push(module.title);
          if (module.subtitle !== undefined && !slot.subtitles.includes(module.subtitle)) slot.subtitles.push(module.subtitle);
        });
      }
      return merged.map(({ module, titles, subtitles }) => ({
        ...module,
        title: titles.join(AUTO_TITLE_JOIN),
        subtitle: subtitles.length === 0 ? undefined : subtitles.join(AUTO_TITLE_JOIN),
      }));
    });
  const modules = modulesOf(tree.children, 0).filter((module) => module.nodes.some((n) => sink.placed.has(n)));
  return { layout: sink.items, modules };
}
