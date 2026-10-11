import type { Node } from '../../autograd';
import type { BackwardOverride, ForwardStep } from '../../steps';
import { child, fail } from '../source';
import { BRACE_KEYS, PATCH_KEYS, STEP_KEYS, TEXT_KEYS } from './keys';
import { notesOf } from './notes';
import type { Compiler } from './text';
import type { Env, Scope, Slot, StepRec } from './types';
import { checkKeys, isRecord, join, list, record } from './util';

interface ResolvedStep {
  rec: StepRec;
  show: Node[];
}

function combineLabels(a: string | undefined, b: string | undefined): string | undefined {
  if (a === undefined || b === undefined || a === b) return a ?? b;
  const math = /^\$([^$]*)\$$/;
  const ma = math.exec(a);
  const mb = math.exec(b);
  return ma !== null && mb !== null ? `$${ma[1]},\\ ${mb[1]}$` : `${a}, ${b}`;
}

export function braceOf(c: Compiler, value: unknown, env: Env): { title: string; subtitle?: string; formula?: string } {
  const spec = record(value, env.where, 'brace');
  checkKeys(spec, BRACE_KEYS, env.where);
  if (typeof spec.title !== 'string') fail(env.where, 'brace needs a title');
  const brace: { title: string; subtitle?: string } = { title: c.ofText(spec.title, { ...env, where: child(env.where, spec, 'title') }) };
  if (spec.subtitle !== undefined) {
    if (typeof spec.subtitle !== 'string') fail(child(env.where, spec, 'subtitle'), 'subtitle must be a string');
    brace.subtitle = c.ofText(spec.subtitle, { ...env, where: child(env.where, spec, 'subtitle') });
  }
  return brace;
}

function resolveSteps(c: Compiler, scope: Scope, decl: Record<string, unknown>, env: Env, field: string): ResolvedStep[] {
  if (decl.steps === undefined) return [];
  const stepsWhere = child(env.where, decl, field);
  if (!Array.isArray(decl.steps)) fail(stepsWhere, 'steps must be a list');
  const seen = new Set<string>();
  return decl.steps.flatMap((step, i): ResolvedStep[] => {
    const at = child(stepsWhere, decl.steps as unknown[], i);
    if (!isRecord(step)) fail(at, 'a step must be a mapping');
    checkKeys(step, STEP_KEYS, at);
    if (typeof step.id !== 'string') fail(at, 'a step needs an id');
    if (!c.enabled(step, { ...env, where: at })) {
      c.switchedOffSteps.add(`${scope.path}#${step.id}`);
      return [];
    }
    if (seen.has(step.id)) fail(child(at, step, 'id'), `step id '${step.id}' is used twice`);
    seen.add(step.id);
    const fieldEnv = (key: string): Env => ({ ...env, where: child(at, step, key) });
    if (step.show === undefined) fail(at, 'a step needs show');
    const shown = list(step.show as unknown).flatMap((s, k) => c.nodeList(s, { ...env, where: Array.isArray(step.show) ? child(child(at, step, 'show'), step.show, k) : child(at, step, 'show') }));
    const withAnon = (nodes: Node[]) => [...new Set(nodes.flatMap((n) => [...(scope.anon.get(n) ?? []), n]))];
    const show = withAnon(shown);
    const forward: ForwardStep = {
      show,
      title: step.title === undefined ? step.id : c.text(step.title, fieldEnv('title')),
    };
    if (step.active !== undefined) forward.active = withAnon(list(step.active as unknown).flatMap((s) => c.nodeList(s, fieldEnv('active'))));
    if (step.label !== undefined) forward.label = c.text(step.label, fieldEnv('label'));
    if (step.text !== undefined) forward.text = c.text(step.text, fieldEnv('text'));
    if (step.segment !== undefined) {
      if (typeof step.segment !== 'string') fail(child(at, step, 'segment'), 'segment must be a string');
      forward.segment = step.segment;
    }
    if (step.brace !== undefined) forward.brace = braceOf(c, step.brace, fieldEnv('brace'));
    if (step.backward !== undefined) forward.backward = c.texts(step.backward, fieldEnv('backward'), TEXT_KEYS);
    if (step.notes !== undefined) forward.notes = notesOf(c, step.notes, fieldEnv('notes'));
    return [{ rec: { refs: [{ owner: scope.path, id: step.id }], step: forward }, show }];
  });
}

export function moduleSteps(c: Compiler, scope: Scope, decl: Record<string, unknown>, env: Env, slots: Slot[]): StepRec[] {
  const containers: { key: string; steps: ResolvedStep[] }[] = [{ key: '', steps: resolveSteps(c, scope, decl, env, 'steps') }];
  const events: { slot: Extract<Slot, { kind: 'node' | 'child' }>; key: string }[] = [];
  const keys = new Map<Slot, string>();
  const flatten = (level: Slot[], prefix: string) => {
    level.forEach((slot, i) => {
      const key = slot.kind === 'child' && slot.merge !== undefined ? (keys.get(slot.merge) ?? `${prefix}/${i}`) : `${prefix}/${i}`;
      keys.set(slot, key);
      if (slot.kind === 'group') {
        containers.push({ key, steps: resolveSteps(c, scope, slot.decl, { ...env, where: slot.where, loops: slot.loops }, 'steps') });
        flatten(slot.slots, key);
      } else if (slot.kind === 'loop') {
        slot.iterations.forEach((iteration, n) => flatten(iteration, slot.sequential ? `${key}*${n}` : `${key}*`));
      } else {
        events.push({ slot, key });
      }
    });
  };
  flatten(slots, '');
  const covered = new Set<Node>(containers.flatMap((container) => container.steps.flatMap((s) => s.show)));
  const anchored = new Map<number, { key: string; rec: StepRec }[]>();
  for (const container of containers) {
    let floor = -1;
    for (const step of container.steps) {
      const first = events.findIndex((event) => step.show.some((n) => event.slot.created.includes(n)));
      floor = Math.max(floor, first);
      anchored.set(floor, [...(anchored.get(floor) ?? []), { key: `own${container.key}:${step.rec.refs[0].id}`, rec: step.rec }]);
    }
  }
  const out: { key: string; rec: StepRec; combine?: boolean }[] = [...(anchored.get(-1) ?? [])];
  events.forEach(({ slot, key }, i) => {
    if (slot.kind === 'child') slot.steps.forEach((rec, j) => out.push({ key: `${key}#${j}`, rec, combine: slot.merge !== undefined }));
    else if (!covered.has(slot.node)) out.push({ key, rec: autoStep(scope, slot.name, slot.node, covered) });
    out.push(...(anchored.get(i) ?? []));
  });
  const firsts = new Map<string, StepRec>();
  const merged: StepRec[] = [];
  for (const { key, rec, combine } of out) {
    const first = firsts.get(key);
    if (first === undefined) {
      firsts.set(key, rec);
      merged.push(rec);
      continue;
    }
    first.refs.push(...rec.refs);
    if (combine) {
      first.step.label = combineLabels(first.step.label, rec.step.label);
      const backward = combineLabels(first.step.backward?.label, rec.step.backward?.label);
      if (backward !== undefined) first.step.backward = { ...first.step.backward, label: backward };
    }
    const union = (a: Node | Node[], b: Node | Node[]) => [...new Set([...list(a), ...list(b)])];
    if (first.step.active !== undefined || rec.step.active !== undefined) {
      first.step.active = union(first.step.active ?? first.step.show, rec.step.active ?? rec.step.show);
    }
    first.step.show = union(first.step.show, rec.step.show);
  }
  return merged;
}

function autoStep(scope: Scope, name: string, node: Node, covered: Set<Node>): StepRec {
  const anon = scope.anon.get(node) ?? [];
  const leaves = [...anon, node]
    .flatMap((n) => n.inputs)
    .flatMap((input) => (input.op === 'transpose' && input.inputs[0].kind === 'param' ? [input.inputs[0], input] : [input]))
    .filter((input) => (input.kind === 'param' || input.kind === 'buffer' || (input.op === 'transpose' && input.inputs[0].kind === 'param')) && !covered.has(input));
  const show = [...new Set([...leaves, ...anon, node])];
  for (const n of leaves) covered.add(n);
  return { refs: [{ owner: scope.path, id: name }], step: { show, title: name, label: name } };
}

export function moduleBackward(c: Compiler, decl: Record<string, unknown>, env: Env): void {
  const where = child(env.where, decl, 'backward');
  const backward = record(decl.backward, where, 'backward');
  for (const [name, text] of Object.entries(backward)) {
    const at = child(where, backward, name);
    const node = c.node(name, { ...env, where: at });
    const spec = record(text, at, 'a backward text');
    checkKeys(spec, new Set(['title', 'text', 'label']), at);
    const override: BackwardOverride = {};
    if (spec.title !== undefined) override.title = c.text(spec.title, { ...env, where: child(at, spec, 'title') });
    if (spec.text !== undefined) override.text = c.text(spec.text, { ...env, where: child(at, spec, 'text') });
    if (spec.label !== undefined) override.label = c.text(spec.label, { ...env, where: child(at, spec, 'label') });
    c.backward.push({ node, text: override, where: at });
  }
}

export function patchSteps(c: Compiler, scope: Scope, steps: StepRec[], use: Record<string, unknown>, env: Env): void {
  const where = child(env.where, use, 'steps');
  const patches = record(use.steps, where, 'steps');
  for (const [path, patchSpec] of Object.entries(patches)) {
    const at = child(where, patches, path);
    const spec = record(patchSpec, at, 'a step patch');
    checkKeys(spec, PATCH_KEYS, at);
    if (!c.enabled(spec, { ...env, where: at })) continue;
    const dot = path.lastIndexOf('.');
    const owner = dot < 0 ? scope.path : join(scope.path, path.slice(0, dot));
    const id = dot < 0 ? path : path.slice(dot + 1);
    const target = steps.find((s) => s.refs.some((ref) => ref.owner === owner && ref.id === id));
    if (target === undefined) fail(at, `step '${path}' does not exist in '${scope.path}'`);
    const fieldEnv = (key: string): Env => ({ ...env, where: child(at, spec, key) });
    if (spec.title !== undefined) target.step.title = c.text(spec.title, fieldEnv('title'));
    if (spec.label !== undefined) target.step.label = c.text(spec.label, fieldEnv('label'));
    if (spec.text !== undefined) target.step.text = c.text(spec.text, fieldEnv('text'));
    if (spec.append !== undefined) {
      const extra = c.text(spec.append, fieldEnv('append'));
      target.step.text = target.step.text ? `${target.step.text}\n${extra}` : extra;
    }
    if (spec.backward !== undefined) target.step.backward = { ...target.step.backward, ...c.texts(spec.backward, fieldEnv('backward'), TEXT_KEYS) };
    if (spec.notes !== undefined) target.step.notes = [...(target.step.notes ?? []), ...notesOf(c, spec.notes, fieldEnv('notes'))];
  }
}
