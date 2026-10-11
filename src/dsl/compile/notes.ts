import { Node } from '../../autograd';
import { type NoteSide, type StepNote, isLeafLike } from '../../steps';
import { DslError, child, fail } from '../source';
import { NOTE_KEYS, NOTE_SIDES } from './keys';
import type { Compiler } from './text';
import { type Env, type Resolved, Scope } from './types';
import { checkKeys, isRecord } from './util';

function scopeNodes(scope: Scope): Node[] {
  const inputs = isRecord(scope.def.decl.inputs) ? new Set(Object.keys(scope.def.decl.inputs)) : new Set<string>();
  const own = [...scope.entries].filter(([name]) => !inputs.has(name)).map(([, node]) => node);
  return [...own, ...[...scope.children.values()].flatMap(scopeNodes)];
}

function resolvedNodes(text: string, found: Resolved, env: Env): Node[] | undefined {
  if (found instanceof Node) return [found];
  if (Array.isArray(found)) return found;
  if (found instanceof Scope) {
    const nodes = scopeNodes(found).filter((n) => !isLeafLike(n));
    if (nodes.length === 0) fail(env.where, `module instance '${text}' has no computed tensors to point at`);
    return nodes;
  }
  return undefined;
}

function noteTarget(c: Compiler, text: string, env: Env): { nodes: Node[]; index?: number[] } {
  const element = /^(.+)\[\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*\]$/.exec(text.trim());
  if (element !== null) {
    let whole: Resolved | undefined;
    try {
      whole = c.reference(text, env);
    } catch (error) {
      if (!(error instanceof DslError)) throw error;
    }
    const wholeNodes = whole === undefined ? undefined : resolvedNodes(text, whole, env);
    if (wholeNodes !== undefined) return { nodes: wholeNodes };
    const base = element[1].trim();
    const node = c.node(base, env);
    const index = element[2].split(',').map((part) => Number(part.trim()));
    if (index.length !== node.shape.length) fail(env.where, `'${text}' gives ${index.length} indices, but '${base}' has shape (${node.shape.join(', ')})`);
    index.forEach((k, axis) => {
      if (k < 0 || k >= node.shape[axis]) fail(env.where, `'${text}': index ${k} is outside 0..${node.shape[axis] - 1} on axis ${axis}`);
    });
    return { nodes: [node], index };
  }
  const nodes = resolvedNodes(text, c.reference(text, env), env);
  if (nodes !== undefined) return { nodes };
  return fail(env.where, `'${text}' is not a tensor or a module instance`);
}

export function notesOf(c: Compiler, value: unknown, env: Env): StepNote[] {
  if (!Array.isArray(value)) fail(env.where, 'notes must be a list');
  return value.map((item, i) => {
    const at = child(env.where, value, i);
    if (!isRecord(item)) fail(at, 'a note must be a mapping with at and text');
    checkKeys(item, NOTE_KEYS, at);
    if (typeof item.at !== 'string') fail(at, 'a note needs at: a tensor name, a tensor element such as q[0,1,2], or a module instance');
    if (typeof item.text !== 'string' || item.text.trim() === '') fail(at, 'a note needs a non-empty text');
    const note: StepNote = { ...noteTarget(c, item.at, { ...env, where: child(at, item, 'at') }), text: c.ofText(item.text, { ...env, where: child(at, item, 'text') }) };
    if (item.side !== undefined) {
      if (typeof item.side !== 'string' || !NOTE_SIDES.has(item.side)) fail(child(at, item, 'side'), `side must be one of ${[...NOTE_SIDES].join(', ')}`);
      note.side = item.side as NoteSide;
    }
    return note;
  });
}
