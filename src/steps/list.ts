import { graph } from '../autograd';
import type { StepInfo, StepState, Steps } from './steps';

export interface StepListItem {
  title: string;
  label?: string;
  text?: string;
  segment?: string;
  sub?: string;
  id?: string;
}

class StepListImpl implements Steps {
  readonly graph = graph({});
  readonly list: readonly StepInfo[];
  readonly visible = Object.freeze([]);
  readonly modules = Object.freeze([]);
  current = 0;
  focus = 0;
  private readonly subscribers: (() => void)[] = [];

  constructor(items: readonly StepListItem[]) {
    if (items.length === 0) throw new Error('createStepList: items must contain at least one step');
    const ids = new Set<string>();
    this.list = Object.freeze(
      items.map((item, index): StepInfo => {
        const label = item.label ?? item.title;
        const id = item.id ?? `s:${index}`;
        if (ids.has(id)) throw new Error(`createStepList: step id '${id}' is used more than once`);
        ids.add(id);
        const info: StepInfo = { dir: 'forward', index, id, kind: 'op', label, title: item.title, text: item.text ?? '', nodes: [], segment: item.segment ?? label };
        if (item.sub !== undefined) info.sub = item.sub;
        return info;
      }),
    );
  }

  go(i: number): void {
    const next = Math.max(0, Math.min(this.list.length - 1, i));
    if (next === this.current) return;
    this.current = next;
    this.notify();
  }

  next(): void {
    this.go(this.current + 1);
  }

  prev(): void {
    this.go(this.current - 1);
  }

  setFocus(b: number): void {
    if (b !== 0) throw new Error(`setFocus: ${b} is out of range; a step list has 1 row, so b must be 0`);
  }

  state(): StepState {
    return { dir: 'forward', index: this.current, kind: 'op', focus: this.focus, revealed: () => false, active: () => false };
  }

  subscribe(fn: () => void): () => void {
    this.subscribers.push(fn);
    return () => {
      const at = this.subscribers.indexOf(fn);
      if (at >= 0) this.subscribers.splice(at, 1);
    };
  }

  text(i: number = this.current): { title: string; body: string } {
    const info = this.list[i];
    if (info === undefined) throw new Error(`step ${i} does not exist (there are ${this.list.length} steps)`);
    return { title: info.title, body: info.text };
  }

  folded(_id: string): boolean {
    return false;
  }

  setFolded(_id: string, _folded: boolean): void {}

  setAllFolded(_folded: boolean): void {}

  dispose(): void {
    this.subscribers.length = 0;
  }

  private notify(): void {
    [...this.subscribers].forEach((fn) => fn());
  }
}

export function createStepList(items: readonly StepListItem[]): Steps {
  return new StepListImpl(items);
}
