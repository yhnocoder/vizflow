import { DslError, child, fail } from '../source';
import { markSteps } from '../tex';
import type { Compiler } from './text';
import type { Env, StepRec } from './types';
import { join } from './util';

export function registerFormula(c: Compiler, owner: string, sources: { decl: Record<string, unknown>; env: Env }[], target: { formula?: string }): void {
  const source = sources.filter((s) => s.decl.formula !== undefined).pop();
  if (source === undefined) return;
  const where = child(source.env.where, source.decl, 'formula');
  if (typeof source.decl.formula !== 'string') fail(where, 'formula must be a string');
  c.formulas.push({ owner, tex: source.decl.formula, env: { ...source.env, where }, target });
}

export function markFormulas(c: Compiler, steps: StepRec[]): void {
  const marks = new Map<StepRec, string>();
  const markOf = (rec: StepRec): string => {
    let name = marks.get(rec);
    if (name === undefined) {
      name = `s${marks.size}`;
      marks.set(rec, name);
      rec.step.marks = [...(rec.step.marks ?? []), name];
    }
    return name;
  };
  for (const { owner, tex, env, target } of c.formulas) {
    const where = env.where;
    const recsOf = (path: string): StepRec[] => {
      if (path === '*' || path.endsWith('.*')) {
        const base = path === '*' ? owner : join(owner, path.slice(0, -2));
        const inside = (ref: { owner: string }) => ref.owner === base || ref.owner.startsWith(`${base}.`);
        const found = steps.filter((s) => s.refs.some(inside));
        if (found.length === 0) fail(where, `\\step{${path}}: '${base || 'page'}' has no steps`);
        return found;
      }
      const dot = path.lastIndexOf('.');
      const stepOwner = dot < 0 ? owner : join(owner, path.slice(0, dot));
      const id = dot < 0 ? path : path.slice(dot + 1);
      const rec = steps.find((s) => s.refs.some((ref) => ref.owner === stepOwner && ref.id === id));
      if (rec === undefined) {
        const known = steps.flatMap((s) => s.refs).filter((ref) => ref.owner === owner).map((ref) => ref.id);
        fail(where, `\\step{${path}}: '${owner || 'page'}' has no step '${path}'; its steps are ${known.join(', ') || '(none)'}`);
      }
      return [rec];
    };
    const mark = (spec: string): string => [...new Set(spec.split(',').flatMap((path) => recsOf(path.trim())).map(markOf))].join('.');
    try {
      target.formula = markSteps(c.ofText(tex, env), mark);
    } catch (error) {
      if (error instanceof DslError) throw error;
      fail(where, (error as Error).message);
    }
  }
}
