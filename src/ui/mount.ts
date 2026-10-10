import { createSignal, onCleanup, type Accessor } from 'solid-js';
import type { Graph } from '../autograd';
import type { Steps } from '../steps';

export function trackGraph(g: Graph): Accessor<number> {
  const [version, setVersion] = createSignal(g.version);
  onCleanup(g.subscribe(() => setVersion(g.version)));
  return version;
}

export function trackSteps(steps: Steps | undefined): Accessor<number> {
  const [version, setVersion] = createSignal(0);
  if (steps !== undefined) onCleanup(steps.subscribe(() => setVersion((v) => v + 1)));
  return version;
}

export function resolveElement(el: string | HTMLElement, caller: string): HTMLElement {
  if (typeof el !== 'string') return el;
  const found = document.querySelector<HTMLElement>(el);
  if (found === null) throw new Error(`${caller}: no element matches '${el}'`);
  return found;
}
