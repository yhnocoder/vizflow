import { Node } from '../../autograd';
import { type NoteSide, type StepNote, isLeafLike } from '../../steps';
import { child, fail } from '../source';
import { describe, isView } from './expressions';
import type { Texts } from './texts';
import { type Env, Instance } from './types';
import { checkKeys, isRecord } from './util';

const NOTE_KEYS: ReadonlySet<string> = new Set(['at', 'text', 'side']);
const NOTE_SIDES: ReadonlySet<string> = new Set(['left', 'right', 'above', 'below']);

function target(c: Texts, text: unknown, env: Env): { nodes: Node[]; index?: number[] } {
  const found = c.reference(text, env);
  if (found instanceof Node) return { nodes: [found] };
  if (isView(found)) {
    if (found.grad) fail(env.where, 'a note points at a tensor, not at its gradient');
    if (!found.index.every((item) => typeof item === 'number') || found.index.length !== found.node.shape.length) fail(env.where, `'${String(text)}' must name one element, such as x[0, 1]`);
    return { nodes: [found.node], index: found.index as number[] };
  }
  if (found instanceof Instance) {
    const nodes = c.onlyCall(found, env, 0).created.filter((n) => !isLeafLike(n));
    if (nodes.length === 0) fail(env.where, `module instance '${String(text)}' has no computed tensors to point at`);
    return { nodes };
  }
  return fail(env.where, `'${String(text)}' is ${describe(found)}; a note points at a tensor, a tensor element or a module instance`);
}

export function notesOf(c: Texts, value: unknown, env: Env): StepNote[] {
  if (!Array.isArray(value)) fail(env.where, 'notes must be a list');
  return value.map((item, i) => {
    const at = child(env.where, value, i);
    if (!isRecord(item)) fail(at, 'a note must be a mapping with at and text');
    checkKeys(item, NOTE_KEYS, at);
    if (item.at === undefined) fail(at, 'a note needs at');
    if (typeof item.text !== 'string' || item.text.trim() === '') fail(at, 'a note needs a non-empty text');
    const note: StepNote = { ...target(c, item.at, { ...env, where: child(at, item, 'at') }), text: c.text(item.text, { ...env, where: child(at, item, 'text') }) };
    if (item.side !== undefined) {
      if (typeof item.side !== 'string' || !NOTE_SIDES.has(item.side)) fail(child(at, item, 'side'), `side must be one of ${[...NOTE_SIDES].join(', ')}`);
      note.side = item.side as NoteSide;
    }
    return note;
  });
}
