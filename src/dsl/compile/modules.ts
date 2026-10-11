import type { Node } from '../../autograd';
import { type ForwardStep, type StepModule, isLeafLike } from '../../steps';
import { child, fail } from '../source';
import { FOLD_KEYS, TEXT_KEYS } from './keys';
import type { Compiler } from './text';
import type { Env, ModuleRec } from './types';
import { list, record } from './util';

export function openModule(c: Compiler, id: string, title: string, subtitle: string | undefined, decl: Record<string, unknown>, env: Env): ModuleRec {
  const flag = (key: string) => {
    const value = decl[key];
    if (value !== undefined && typeof value !== 'boolean') fail(child(env.where, decl, key), `${key} must be true or false`);
    return value === true;
  };
  const parent = c.moduleStack[c.moduleStack.length - 1];
  const fixed = flag('fixed');
  const rec: ModuleRec = {
    id,
    parent: parent?.id ?? null,
    title,
    subtitle: subtitle ?? '',
    fixed,
    folded: fixed || flag('folded'),
    split: flag('split'),
    nodes: [],
    fold: { forward: {}, backward: {} },
    decl,
    env,
  };
  c.modules.push(rec);
  c.moduleStack.push(rec);
  return rec;
}

export function closeModule(c: Compiler, rec: ModuleRec, created: Node[]): void {
  c.moduleStack.pop();
  rec.nodes = created;
  for (const { decl, env } of [rec.defaults, rec].filter((x) => x !== undefined)) {
    if (decl.fold === undefined) continue;
    const where = child(env.where, decl, 'fold');
    const fold = record(decl.fold, where, 'fold');
    rec.fold.forward = { ...rec.fold.forward, ...c.texts(fold, { ...env, where }, FOLD_KEYS) };
    if (fold.backward !== undefined) rec.fold.backward = { ...rec.fold.backward, ...c.texts(fold.backward, { ...env, where: child(where, fold, 'backward') }, TEXT_KEYS) };
  }
}

export function titleOf(c: Compiler, decl: Record<string, unknown>, fallback: string, env: Env): string {
  if (decl.title === undefined) return fallback;
  if (typeof decl.title !== 'string') fail(child(env.where, decl, 'title'), 'title must be a string');
  return c.ofText(decl.title, { ...env, where: child(env.where, decl, 'title') });
}

function segmentOf(chain: ModuleRec[]): { segment: string | undefined; sub: string | undefined } {
  const head = chain.find((m) => !m.split);
  const last = chain[chain.length - 1];
  const sub = head !== undefined && last !== head && !last.split ? last.title : undefined;
  return { segment: head?.title, sub };
}

export function pageModules(c: Compiler): StepModule[] {
  const recs = c.modules.filter((m) => m.nodes.some((n) => !isLeafLike(n)));
  const byId = new Map(recs.map((m) => [m.id, m]));
  const chain = (m: ModuleRec): ModuleRec[] => (m.parent === null || !byId.has(m.parent) ? [m] : [...chain(byId.get(m.parent)!), m]);
  const consumers = (n: Node) => c.g.nodes.filter((u) => u.inputs.includes(n));
  return recs.map((m): StepModule => {
    const nodes = m.nodes.filter((n) => !isLeafLike(n));
    const inside = new Set(nodes);
    const used = nodes.flatMap((n) => n.inputs);
    const inputs = [...new Set(used.filter((n) => !inside.has(n) && !isLeafLike(n)))];
    const params = [...new Set(used.flatMap((n) => (isLeafLike(n) && n.kind === 'op' ? [n, n.inputs[0]] : [n])).filter((n) => isLeafLike(n) && n.kind !== 'label'))].sort((a, b) => a.id - b.id);
    const outputs = nodes.filter((n) => consumers(n).some((u) => !inside.has(u)));
    const { segment, sub } = segmentOf(chain(m));
    const module: StepModule = {
      id: m.id,
      parent: m.parent !== null && byId.has(m.parent) ? m.parent : null,
      title: m.title,
      subtitle: m.subtitle,
      inputs,
      outputs: outputs.length > 0 ? outputs : nodes.slice(-1),
      nodes,
      params,
      fixed: m.fixed,
      folded: m.folded,
      segment: segment ?? m.title,
      forward: m.fold.forward,
      backward: m.fold.backward,
    };
    if (sub !== undefined) module.sub = sub;
    if (m.formula !== undefined) module.formula = m.formula;
    return module;
  });
}

export function placeSteps(c: Compiler, forward: ForwardStep[], modules: StepModule[]): void {
  const recs = new Map(c.modules.map((m) => [m.id, m]));
  const depth = new Map<string, number>();
  for (const m of modules) depth.set(m.id, m.parent === null ? 0 : (depth.get(m.parent) ?? 0) + 1);
  const innermost = (node: Node) => modules.filter((m) => m.nodes.includes(node)).sort((a, b) => depth.get(b.id)! - depth.get(a.id)!)[0];
  for (const step of forward) {
    const first = list(step.show).find((n) => !isLeafLike(n));
    const module = first === undefined ? undefined : innermost(first);
    const chain: ModuleRec[] = [];
    for (let m = module === undefined ? undefined : recs.get(module.id); m !== undefined; m = m.parent === null ? undefined : recs.get(m.parent)) chain.unshift(m);
    const { segment, sub } = segmentOf(chain);
    if (module !== undefined) step.module = module.id;
    if (step.segment === undefined) step.segment = segment ?? step.label ?? step.title;
    if (sub !== undefined) step.sub = sub;
  }
}
