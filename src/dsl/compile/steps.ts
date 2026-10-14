import { Node } from '../../autograd';
import { isLayoutOp } from '../../scene3d/layout';
import { type BackwardOverride, type ForwardStep, type StepBrace, type StepGroup, type StepModule, isLeafLike } from '../../steps';
import type { Ast } from '../expr';
import { type Where, child, fail, merged } from '../source';
import { markSteps } from '../tex';
import { describe } from './expressions';
import { notesOf } from './notes';
import type { PageCompiler } from './page';
import { type Call, type Env, Instance, InstanceList, Result, type Value } from './types';
import { checkKeys, isNum, num, record } from './util';

const EXPLANATION_KEYS = ['label', 'title', 'text', 'append'];
const OPTION_KEYS: ReadonlySet<string> = new Set([...EXPLANATION_KEYS, 'show', 'subtitle', 'expand', 'flat', 'brace', 'formula', 'notes', 'fold', 'backward', 'edit']);
const STEP_KEYS: ReadonlySet<string> = new Set([...OPTION_KEYS, 'id', 'steps']);
const LOOP_DEFAULT_KEYS: ReadonlySet<string> = new Set([...OPTION_KEYS, 'steps']);
const LOOP_KEYS: ReadonlySet<string> = new Set(['id', 'loop', 'defaults', 'rules']);
const RULE_KEYS: ReadonlySet<string> = new Set([...OPTION_KEYS, 'at']);
const BACKWARD_KEYS: ReadonlySet<string> = new Set(['label', 'title', 'text', 'append', 'notes']);
const FOLD_KEYS: ReadonlySet<string> = new Set(['label', 'title', 'text', 'append', 'backward']);
const INTRO_KEYS = ['label', 'notes', 'backward'];
const LOOP = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*$/;
const RANGE_LOOP = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+range\((.+)\)\s*$/;

type Options = Record<string, unknown>;

interface Rec {
  step: ForwardStep & { show: Node[] };
}

interface Fold {
  forward: BackwardOverride;
  backward: BackwardOverride;
}

interface Formula {
  tex: string;
  env: Env;
}

interface Leaf {
  kind: 'leaf';
  id: string;
  rec: Rec;
  brace: boolean;
  call?: Call;
  subtitle?: string;
  braceTitle?: string;
}

interface Group {
  kind: 'group';
  id: string;
  title?: string;
  subtitle?: string;
  flat: boolean;
  call?: Call;
  children: Tree[];
  intro?: Rec;
  formula?: Formula;
  fold: Fold;
  expand?: boolean;
}

export type Tree = Leaf | Group;

type Shown = { kind: 'nodes'; nodes: Node[]; all: boolean } | { kind: 'call'; call: Call } | { kind: 'zip'; calls: [Call, Call]; where: Where };

export interface CallSteps {
  title: string;
  subtitle: string;
  fold: Fold;
  expand: boolean;
  flat: boolean;
}

function unique<T>(items: T[]): T[] {
  return [...new Set(items)];
}

function combineLabels(a: string | undefined, b: string | undefined): string | undefined {
  if (a === undefined || b === undefined || a === b) return a ?? b;
  const math = /^\$([^$]*)\$$/;
  const ma = math.exec(a);
  const mb = math.exec(b);
  return ma !== null && mb !== null ? `$${ma[1]},\\ ${mb[1]}$` : `${a}, ${b}`;
}

function leaves(trees: Tree[]): Rec[] {
  return trees.flatMap((tree) => (tree.kind === 'leaf' ? [tree.rec] : [...(tree.intro === undefined ? [] : [tree.intro]), ...leaves(tree.children)]));
}

function terms(ast: Ast): Ast[] {
  return ast.k === 'bin' && ast.op === '+' ? [...terms(ast.a), ...terms(ast.b)] : [ast];
}

export class StepCompiler {
  readonly shownCalls = new Set<Call>();
  readonly callSteps = new Map<Call, CallSteps>();
  readonly allSteps = new Set<ForwardStep>();
  private readonly carried = new Map<Call, Node[]>();
  private readonly marks = new Map<Rec, string>();
  private readonly absent = new WeakMap<Tree[], Set<string>>();

  constructor(private readonly c: PageCompiler) {}

  compileCall(call: Call): Tree[] {
    this.shownCalls.add(call);
    const def = call.instance.def;
    const where = child(def.where, def.decl, 'steps');
    const steps = def.decl.steps;
    if (steps === undefined) return [];
    return this.list(steps, this.c.callEnv(call, where));
  }

  private list(steps: unknown, env: Env): Tree[] {
    const where = env.where;
    if (!Array.isArray(steps)) fail(where, 'steps must be a list of steps');
    const seen = new Set<string>();
    const out: Tree[] = [];
    const absent = new Set<string>();
    this.absent.set(out, absent);
    steps.forEach((spec, i) => {
      const at = child(where, steps, i);
      const step = record(spec, at, 'a step');
      if (typeof step.id !== 'string') fail(at, 'a step needs an id');
      if (seen.has(step.id)) fail(child(at, step, 'id'), `step id '${step.id}' is used twice`);
      seen.add(step.id);
      const stepEnv = { ...env, where: at };
      if (step.loop === undefined && step.steps === undefined && this.isNone(step.show ?? step.id, this.field(stepEnv, step, step.show === undefined ? 'id' : 'show'))) {
        absent.add(step.id);
        return;
      }
      if (step.loop !== undefined) out.push(...this.loop(step, stepEnv));
      else {
        checkKeys(step, STEP_KEYS, at);
        out.push(this.step(step.id, step, stepEnv));
      }
    });
    return out;
  }

  private loop(step: Options, env: Env): Tree[] {
    const at = env.where;
    checkKeys(step, LOOP_KEYS, at);
    const loopWhere = child(at, step, 'loop');
    const { variable, index, items } = this.loopItems(step.loop, { ...env, where: loopWhere });
    const defaults = record(step.defaults, child(at, step, 'defaults'), 'defaults');
    checkKeys(defaults, LOOP_DEFAULT_KEYS, child(at, step, 'defaults'));
    const rulesWhere = child(at, step, 'rules');
    const rules = step.rules ?? [];
    if (!Array.isArray(rules)) fail(rulesWhere, 'rules must be a list');
    return items.map((item, i) => {
      const locals = new Map<string, Value>([...env.locals, ...(index === undefined ? [] : [[index, num(i)] as [string, Value]]), [variable, item]]);
      const matching = rules.filter((rule, k) => {
        const ruleWhere = child(rulesWhere, rules, k);
        const spec = record(rule, ruleWhere, 'a rule');
        checkKeys(spec, RULE_KEYS, ruleWhere);
        if (typeof spec.at !== 'number' || !Number.isInteger(spec.at)) fail(child(ruleWhere, spec, 'at'), 'at must be an integer index');
        const target = spec.at < 0 ? items.length + spec.at : spec.at;
        if (target < 0 || target >= items.length) fail(child(ruleWhere, spec, 'at'), `at ${spec.at} is out of range for ${items.length} items`);
        return target === i;
      });
      const options = merged([defaults, ...matching]);
      delete options.at;
      if (options.show === undefined && index !== undefined) options.show = variable;
      return this.step(`${String(step.id)}[${index === undefined ? (item as { toy: number }).toy : i}]`, options, { ...env, locals, where: child(at, step, 'defaults') });
    });
  }

  private loopItems(loop: unknown, env: Env): { variable: string; index: string | undefined; items: Value[] } {
    const text = typeof loop === 'string' ? loop : '';
    const range = RANGE_LOOP.exec(text);
    if (range !== null) {
      const bounds = this.c.reference(`[${range[2]}]`, env);
      const ints = Array.isArray(bounds) ? bounds.map((b) => (isNum(b) && Number.isInteger(b.toy) ? b.toy : undefined)) : [];
      if (ints.length < 1 || ints.length > 2 || ints.some((b) => b === undefined)) this.c.fail(env, `range(${range[2]}) needs one or two integers`);
      const [start, stop] = ints.length === 1 ? [0, ints[0] as number] : (ints as number[]);
      return { variable: range[1], index: undefined, items: Array.from({ length: Math.max(0, stop - start) }, (_, k) => num(start + k)) };
    }
    const match = LOOP.exec(text);
    if (match === null) fail(env.where, "loop must look like 'layer = layers[i]' or 'i in range(n)'");
    const [, variable, listText, index] = match;
    const listValue = this.c.reference(listText, env);
    const items: Value[] = listValue instanceof InstanceList ? listValue.items : Array.isArray(listValue) ? listValue : this.c.fail(env, `'${listText}' is ${describe(listValue)}, not a list`);
    return { variable, index, items };
  }

  private isNone(show: unknown, env: Env): boolean {
    if (typeof show !== 'string') return false;
    const ast = this.c.parse(show, env);
    return (ast.k === 'name' || ast.k === 'member') && this.c.evaluate(ast, env, { k: 'ref' }, false) === null;
  }

  private field(env: Env, options: Options, key: string): Env {
    return { ...env, where: child(env.where, options, key) };
  }

  private text(options: Options, key: string, env: Env): string {
    return this.c.text(options[key], this.field(env, options, key));
  }

  private step(id: string, options: Options, env: Env): Tree {
    if (options.steps !== undefined) {
      if (options.show !== undefined) fail(child(env.where, options, 'show'), 'a step with steps shows what its steps show; remove show');
      const group = this.group(id, this.list(options.steps, this.field(env, options, 'steps')));
      this.groupOptions(group, options, env);
      return group;
    }
    const shown = this.shown(options.show ?? id, this.field(env, options, 'show'));
    if (shown.kind === 'nodes') {
      const leaf = this.leaf(id, shown.nodes, env);
      if (shown.all) this.allSteps.add(leaf.rec.step);
      this.leafOptions(leaf, options, env);
      return leaf;
    }
    if (shown.kind === 'zip') {
      const [a, b] = shown.calls;
      const left = this.compileCall(a);
      const right = this.compileCall(b);
      if (left.length === 0 || right.length === 0) fail(shown.where, `zip needs two module calls whose modules have steps`);
      const group = this.group(id, this.zip(left, right, shown.where));
      this.groupOptions(group, options, env);
      return group;
    }
    const call = shown.call;
    if (call.instance.def.decl.steps === undefined) {
      this.shownCalls.add(call);
      const leaf = this.leaf(id, this.stepless(call), env);
      leaf.call = call;
      leaf.braceTitle = call.instance.name;
      this.leafOptions(leaf, options, env);
      return leaf;
    }
    const group = this.group(id, this.compileCall(call));
    group.call = call;
    group.title = call.instance.name;
    const def = call.instance.def;
    const childEnv = this.c.callEnv(call);
    if (def.decl.fold !== undefined) group.fold = this.fold(def.decl.fold, { ...childEnv, where: child(def.where, def.decl, 'fold') }, group.fold);
    if (typeof def.decl.formula === 'string') group.formula = { tex: def.decl.formula, env: { ...childEnv, where: child(def.where, def.decl, 'formula') } };
    else if (def.decl.formula !== undefined) fail(child(def.where, def.decl, 'formula'), 'formula must be LaTeX text');
    this.groupOptions(group, options, env);
    return group;
  }

  private group(id: string, children: Tree[]): Group {
    return { kind: 'group', id, flat: false, children, fold: { forward: {}, backward: {} } };
  }

  private leaf(id: string, nodes: Node[], env: Env): Leaf {
    const step: Rec['step'] = { show: this.withAnon(nodes), title: id };
    const call = env.call;
    if (call !== undefined && call.parent !== undefined) step.module = call.prefix;
    return { kind: 'leaf', id, rec: { step }, brace: false };
  }

  private shown(show: unknown, env: Env): Shown {
    const items = Array.isArray(show) ? show : [show];
    const nodes: Node[] = [];
    let all = false;
    let single: Shown | undefined;
    items.forEach((item, k) => {
      const at = Array.isArray(show) ? child(env.where, show, k) : env.where;
      const itemEnv = { ...env, where: at };
      for (const term of terms(this.c.parse(item, itemEnv))) {
        const special = term.k === 'call' && term.fn.k === 'name' && !env.locals.has(term.fn.name) ? term.fn.name : undefined;
        if (special === 'all' || special === 'inputs' || special === 'parameters') {
          if (term.k === 'call' && (term.args.length > 0 || term.kwargs.length > 0)) fail(at, `${special}() takes no arguments`);
          if (special === 'all') all = true;
          else nodes.push(...this.collection(special));
          continue;
        }
        if (special === 'zip' && term.k === 'call') {
          if (items.length !== 1 || terms(this.c.parse(item, itemEnv)).length !== 1) fail(at, 'zip(a, b) is the whole show of its step');
          if (term.args.length !== 2 || term.kwargs.length > 0) fail(at, 'zip takes two module calls: zip(a, b)');
          const calls = term.args.map((arg) => this.callOf(this.c.evaluate(arg, itemEnv, { k: 'ref' }, false), itemEnv, String(item))) as [Call, Call];
          if (calls[0].instance.def !== calls[1].instance.def) fail(at, `zip needs two calls of the same module; ${calls[0].instance.name} is ${calls[0].instance.def.name} and ${calls[1].instance.name} is ${calls[1].instance.def.name}`);
          single = { kind: 'zip', calls, where: at };
          continue;
        }
        const value = this.c.evaluate(term, itemEnv, { k: 'ref' }, false);
        if (value instanceof Node) {
          nodes.push(value);
          continue;
        }
        if (items.length !== 1 || terms(this.c.parse(item, itemEnv)).length !== 1) fail(at, `show '${String(item)}' mixes a module call with other things; a step shows tensors, one module call, or zip(a, b)`);
        single = { kind: 'call', call: this.callOf(value, itemEnv, String(item)) };
      }
    });
    if (single !== undefined) return single;
    return { kind: 'nodes', nodes: unique(nodes), all };
  }

  showCallInputs(trees: Tree[]): void {
    const shows = (t: Tree): Node[] => {
      const own = t.kind === 'leaf' ? [t.rec] : t.intro === undefined ? [] : [t.intro];
      const nodes = own.flatMap((r) => r.step.show);
      return t.kind === 'leaf' ? nodes : [...nodes, ...t.children.flatMap(shows)];
    };
    const shown = new Set(trees.flatMap(shows));
    const first = (t: Tree): Rec | undefined => (t.kind === 'leaf' ? t.rec : (t.intro ?? (t.children.length > 0 ? first(t.children[0]) : undefined)));
    const visit = (t: Tree) => {
      const call = t.call;
      const rec = first(t);
      if (call !== undefined && rec !== undefined) {
        const named = new Set([...(call.parent?.values.values() ?? [])]);
        const added = [...call.inputs.values()].filter((n) => named.has(n) && !shown.has(n) && n.kind === 'op' && !isLayoutOp(n));
        if (added.length > 0) {
          rec.step.show = unique([...added, ...rec.step.show]);
          this.carried.set(call, added);
          added.forEach((n) => shown.add(n));
        }
      }
      if (t.kind === 'group') t.children.forEach(visit);
    };
    trees.forEach(visit);
  }

  private callOf(value: Value, env: Env, text: string): Call {
    if (value instanceof Instance) return this.c.onlyCall(value, env, 0);
    if (value instanceof Result) return value.call;
    return fail(env.where, `show '${text}' is ${describe(value)}; show takes tensors, a module call, zip(a, b), all(), inputs() and parameters()`);
  }

  private collection(name: 'inputs' | 'parameters'): Node[] {
    if (name === 'inputs') return [...this.c.rootInputs].filter((n) => !this.c.tokenLists.has(n));
    return this.c.g.nodes.filter((n) => n.kind === 'param' || (n.kind === 'op' && n.op === 'transpose' && n.inputs[0].kind === 'param'));
  }

  private withAnon(nodes: Node[]): Node[] {
    return unique(nodes.flatMap((n) => [...(this.c.anon.get(n) ?? []).filter((a) => !isLayoutOp(a)), n]));
  }

  private stepless(call: Call): Node[] {
    const created = call.created;
    const used = created.flatMap((n) => n.inputs).filter((n) => n.kind === 'param' || n.kind === 'buffer');
    return unique([...used, ...created]);
  }

  private zip(left: Tree[], right: Tree[], where: Where): Tree[] {
    const ids = (list: Tree[]) => list.map((t) => t.id).join(', ');
    if (ids(left) !== ids(right)) fail(where, `zip needs the same steps on both sides; one has ${ids(left)}, the other ${ids(right)}`);
    return left.map((a, i) => {
      const b = right[i];
      if (a.kind === 'leaf' && b.kind === 'leaf') {
        const step = a.rec.step;
        const other = b.rec.step;
        step.show = unique([...step.show, ...other.show]);
        step.label = combineLabels(step.label, other.label);
        const backward = combineLabels(step.backward?.label, other.backward?.label);
        if (backward !== undefined) step.backward = { ...step.backward, label: backward };
        return { ...a, call: undefined };
      }
      if (a.kind === 'group' && b.kind === 'group') return { ...a, call: undefined, children: this.zip(a.children, b.children, where) };
      return fail(where, `zip: step '${a.id}' has steps on one side only`);
    });
  }

  private leafOptions(leaf: Leaf, options: Options, env: Env): void {
    for (const key of ['flat', 'expand', 'fold', 'formula', 'edit']) {
      if (options[key] !== undefined) fail(child(env.where, options, key), `'${key}' applies to a step with steps`);
    }
    const step = leaf.rec.step;
    if (options.title !== undefined) {
      step.title = this.text(options, 'title', env);
      leaf.braceTitle = step.title;
    }
    if (options.label !== undefined) step.label = this.text(options, 'label', env);
    if (options.text !== undefined) step.text = this.text(options, 'text', env);
    if (options.append !== undefined) step.text = step.text ? `${step.text}\n${this.text(options, 'append', env)}` : this.text(options, 'append', env);
    if (options.subtitle !== undefined) leaf.subtitle = this.text(options, 'subtitle', env);
    if (options.brace !== undefined) {
      if (typeof options.brace !== 'boolean') fail(child(env.where, options, 'brace'), 'brace must be true or false');
      leaf.brace = options.brace;
    }
    if (options.notes !== undefined) step.notes = [...(step.notes ?? []), ...notesOf(this.c, options.notes, this.field(env, options, 'notes'))];
    if (options.backward !== undefined) step.backward = this.backward(options.backward, this.field(env, options, 'backward'), step.backward ?? {});
  }

  private groupOptions(group: Group, options: Options, env: Env): void {
    if (options.brace !== undefined) fail(child(env.where, options, 'brace'), 'a step with steps always has a brace; brace applies to a step that shows tensors');
    if (options.title !== undefined) group.title = this.text(options, 'title', env);
    if (options.subtitle !== undefined) group.subtitle = this.text(options, 'subtitle', env);
    if (options.flat !== undefined) {
      if (typeof options.flat !== 'boolean') fail(child(env.where, options, 'flat'), 'flat must be true or false');
      group.flat = options.flat;
    }
    if (options.expand !== undefined) {
      const value = typeof options.expand === 'boolean' ? options.expand : this.c.expression(options.expand, this.field(env, options, 'expand'));
      if (typeof value !== 'boolean') fail(child(env.where, options, 'expand'), 'expand must be true or false');
      group.expand = value;
    }
    if (options.fold !== undefined) group.fold = this.fold(options.fold, this.field(env, options, 'fold'), group.fold);
    if (options.formula !== undefined) {
      if (typeof options.formula !== 'string') fail(child(env.where, options, 'formula'), 'formula must be LaTeX text');
      group.formula = { tex: options.formula, env: this.field(env, options, 'formula') };
    }
    if (options.edit !== undefined) {
      const where = child(env.where, options, 'edit');
      const patches = record(options.edit, where, 'edit');
      for (const [path, patch] of Object.entries(patches)) this.patch(group.children, path, patch, { ...env, where: child(where, patches, path) });
    }
    if (options.text !== undefined) {
      const intro: Rec = group.intro ?? { step: { show: [], title: group.title ?? group.id } };
      const owner = group.call ?? (env.call?.parent === undefined ? undefined : env.call);
      if (owner !== undefined) intro.step.module = owner.prefix;
      intro.step.text = this.text(options, 'text', env);
      group.intro = intro;
    }
    if (options.append !== undefined) {
      if (group.intro === undefined) fail(child(env.where, options, 'append'), 'append adds to the text of a step; this step with steps has no text');
      group.intro.step.text = `${group.intro.step.text ?? ''}\n${this.text(options, 'append', env)}`;
    }
    for (const key of INTRO_KEYS) {
      if (options[key] === undefined) continue;
      if (group.intro === undefined) fail(child(env.where, options, key), `'${key}' applies to a step of its own; give this step text, or set ${key} on one of its steps`);
      const step = group.intro.step;
      if (key === 'label') step.label = this.text(options, 'label', env);
      if (key === 'notes') step.notes = [...(step.notes ?? []), ...notesOf(this.c, options.notes, this.field(env, options, 'notes'))];
      if (key === 'backward') step.backward = this.backward(options.backward, this.field(env, options, 'backward'), step.backward ?? {});
    }
    if (group.call !== undefined) this.callSteps.set(group.call, { title: group.title ?? group.id, subtitle: group.subtitle ?? '', fold: group.fold, expand: group.expand ?? true, flat: group.flat });
  }

  private patch(trees: Tree[], path: string, value: unknown, env: Env): void {
    const target = this.find(trees, path.split('.'), env.where, path);
    if (target === undefined) return;
    const options = record(value, env.where, 'an edit');
    checkKeys(options, OPTION_KEYS, env.where);
    if (options.show !== undefined) fail(child(env.where, options, 'show'), 'edit cannot change what a step shows');
    if (target.kind === 'leaf') this.leafOptions(target, options, env);
    else this.groupOptions(target, options, env);
  }

  private find(trees: Tree[], parts: string[], where: Where, path: string): Tree | undefined {
    const [head, ...rest] = parts;
    const found = trees.find((t) => t.id === head);
    if (found === undefined && this.absent.get(trees)?.has(head)) return undefined;
    if (found === undefined) fail(where, `there is no step '${path}'; the steps here are ${trees.map((t) => t.id).join(', ') || '(none)'}`);
    if (rest.length === 0) return found;
    if (found.kind !== 'group') fail(where, `step '${head}' shows tensors, so it has no step '${rest.join('.')}'`);
    return this.find(found.children, rest, where, path);
  }

  private backward(value: unknown, env: Env, current: BackwardOverride): BackwardOverride {
    const spec = record(value, env.where, 'backward');
    checkKeys(spec, BACKWARD_KEYS, env.where);
    if (spec.notes !== undefined) fail(child(env.where, spec, 'notes'), 'backward steps do not show notes yet');
    const out: BackwardOverride = { ...current };
    for (const key of ['label', 'title', 'text'] as const) {
      if (spec[key] !== undefined) out[key] = this.c.text(spec[key], { ...env, where: child(env.where, spec, key) });
    }
    if (spec.append !== undefined) {
      const extra = this.c.text(spec.append, { ...env, where: child(env.where, spec, 'append') });
      out.text = out.text ? `${out.text}\n${extra}` : extra;
    }
    return out;
  }

  private fold(value: unknown, env: Env, current: Fold): Fold {
    const spec = record(value, env.where, 'fold');
    checkKeys(spec, FOLD_KEYS, env.where);
    const forward: BackwardOverride = { ...current.forward };
    for (const key of ['label', 'title', 'text'] as const) {
      if (spec[key] !== undefined) forward[key] = this.c.text(spec[key], { ...env, where: child(env.where, spec, key) });
    }
    if (spec.append !== undefined) {
      const extra = this.c.text(spec.append, { ...env, where: child(env.where, spec, 'append') });
      forward.text = forward.text ? `${forward.text}\n${extra}` : extra;
    }
    const backward = spec.backward === undefined ? { ...current.backward } : this.backward(spec.backward, { ...env, where: child(env.where, spec, 'backward') }, current.backward);
    return { forward, backward };
  }

  flatten(trees: Tree[]): { forward: ForwardStep[]; groups: StepGroup[]; braces: StepBrace[] } {
    const forward: ForwardStep[] = [];
    const groups: StepGroup[] = [];
    const braces: StepBrace[] = [];
    const shows = (list: Tree[]) => unique(leaves(list).flatMap((rec) => rec.step.show));
    const framed = (nodes: Node[], call: Call | undefined) => nodes.filter((n) => !(call === undefined ? [] : (this.carried.get(call) ?? [])).includes(n));
    const everShown = new Set(shows(trees));
    const lightEntering = (tree: Tree, intro: Rec) => {
      if (intro.step.show.length > 0 || intro.step.active !== undefined) return;
      const inside = new Set(shows([tree]));
      const entering = unique([...inside].flatMap((n) => n.inputs).filter((n) => !inside.has(n) && !isLeafLike(n) && everShown.has(n)));
      if (entering.length > 0) intro.step.active = entering;
    };
    const walk = (list: Tree[], path: string[], key: string, depth: number) => {
      for (const tree of list) {
        const id = key === '' ? tree.id : `${key}.${tree.id}`;
        if (tree.kind === 'leaf') {
          tree.rec.step.groups = path;
          forward.push(tree.rec.step);
          if (tree.brace || tree.call !== undefined) {
            const brace: StepBrace = { id, title: tree.braceTitle ?? tree.rec.step.title, nodes: framed(tree.rec.step.show, tree.call), depth };
            if (tree.subtitle !== undefined) brace.subtitle = tree.subtitle;
            braces.push(brace);
          }
          continue;
        }
        if (tree.flat) {
          if (tree.intro !== undefined) {
            lightEntering(tree, tree.intro);
            tree.intro.step.groups = path;
            forward.push(tree.intro.step);
          }
          walk(tree.children, path, id, depth);
          continue;
        }
        const title = tree.title ?? tree.id;
        groups.push({ id, title });
        const brace: StepBrace = { id, title, nodes: framed(shows([tree]), tree.call), depth };
        if (tree.subtitle !== undefined) brace.subtitle = tree.subtitle;
        if (tree.call !== undefined && tree.call.parent !== undefined) brace.module = tree.call.prefix;
        braces.push(brace);
        if (tree.intro !== undefined) {
          lightEntering(tree, tree.intro);
          tree.intro.step.groups = [...path, id];
          forward.push(tree.intro.step);
        }
        walk(tree.children, [...path, id], id, depth + 1);
      }
    };
    walk(trees, [], '', 0);
    return { forward, groups, braces };
  }

  modules(visible: ReadonlySet<Node>): StepModule[] {
    const modules: StepModule[] = [];
    const consumers = new Map<Node, Node[]>();
    for (const u of this.c.g.nodes) for (const n of u.inputs) consumers.set(n, [...(consumers.get(n) ?? []), u]);
    const moduleCall = (call: Call | undefined): Call | undefined => {
      for (let at = call; at !== undefined; at = at.parent) if (at.parent !== undefined && at.instance.def.decl.steps !== undefined) return at;
      return undefined;
    };
    for (const call of this.c.calls) {
      if (call.parent === undefined || call.instance.def.decl.steps === undefined) continue;
      const shown = this.callSteps.get(call);
      const nodes = call.created.filter((n) => !isLeafLike(n) && visible.has(n));
      if (nodes.length === 0) continue;
      const inside = new Set(nodes);
      const used = nodes.flatMap((n) => n.inputs);
      const outputs = nodes.filter((n) => (consumers.get(n) ?? []).some((u) => !inside.has(u)));
      modules.push({
        id: call.prefix,
        parent: moduleCall(call.parent)?.prefix ?? null,
        title: shown?.title ?? call.instance.name,
        subtitle: shown?.subtitle ?? '',
        inputs: unique(used.filter((n) => !inside.has(n) && !isLeafLike(n))),
        outputs: outputs.length > 0 ? outputs : nodes.slice(-1),
        nodes,
        params: unique(used.flatMap((n) => (isLeafLike(n) && n.kind === 'op' ? [n, n.inputs[0]] : [n])).filter((n) => isLeafLike(n) && n.kind !== 'label')).sort((a, b) => a.id - b.id),
        folded: false,
        foldable: false,
        forward: shown?.fold.forward ?? {},
        backward: shown?.fold.backward ?? {},
      });
    }
    return modules;
  }

  markFormulas(trees: Tree[]): void {
    const visit = (list: Tree[]) => {
      for (const tree of list) {
        if (tree.kind !== 'group') continue;
        const own = tree.title ?? tree.id;
        if (tree.formula !== undefined) {
          const formula = { title: own, tex: this.formula(tree.formula, tree.children) };
          for (const rec of leaves([tree])) rec.step.formula = formula;
        }
        visit(tree.children);
      }
    };
    visit(trees);
  }

  private formula(formula: Formula, scope: Tree[]): string {
    const env = formula.env;
    const resolve = (path: string): Rec[] => {
      if (path === '*') return leaves(scope);
      const all = path.endsWith('.*');
      const parts = (all ? path.slice(0, -2) : path).split('.');
      const target = this.find(scope, parts, env.where, path);
      if (target === undefined) return [];
      if (target.kind === 'leaf') return [target.rec];
      if (!all) fail(env.where, `\\step{${path}}: '${path}' has steps; write \\step{${path}.*} for all of them`);
      return leaves(target.children);
    };
    const mark = (spec: string): string =>
      unique(spec.split(',').flatMap((path) => resolve(path.trim())))
        .map((rec) => {
          let name = this.marks.get(rec);
          if (name === undefined) {
            name = `s${this.marks.size}`;
            this.marks.set(rec, name);
            rec.step.marks = [...(rec.step.marks ?? []), name];
          }
          return name;
        })
        .join('.');
    try {
      return markSteps(this.c.expandTex(formula.tex, env), mark);
    } catch (error) {
      if (error instanceof Error && error.name === 'DslError') throw error;
      return fail(env.where, (error as Error).message);
    }
  }
}
