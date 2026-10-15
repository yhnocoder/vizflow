import { Node } from '../../autograd';
import { isLayoutOp } from '../../scene3d/layout';
import { type BackwardOverride, type CopyText, type ForwardStep, type FunctionBackward, type StepBrace, type StepGroup, type StepModule, isLeafLike } from '../../steps';
import type { Ast } from '../expr';
import { type Where, child, fail, merged } from '../source';
import { markSteps } from '../tex';
import { describe } from './expressions';
import { notesOf } from './notes';
import type { PageCompiler } from './page';
import { type Call, type Env, type GridCopy, Instance, InstanceList, Result, type Value } from './types';
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
const NAME = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*$/;

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

interface CopyLabel {
  index: number[];
  title?: string;
  subtitle?: string;
}

interface Leaf {
  kind: 'leaf';
  id: string;
  rec: Rec;
  brace: boolean;
  call?: Call;
  subtitle?: string;
  braceTitle?: string;
  copies?: CopyLabel[];
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
  grid?: string;
  copies?: CopyLabel[];
}

export type Tree = Leaf | Group;

interface CopyEnv {
  index?: number[];
  env: Env;
}

interface Texts {
  title?: string;
  label?: string;
  text?: string;
  append?: string;
}

function applyTexts(target: { title: string; label?: string; text?: string }, texts: Texts): void {
  if (texts.title !== undefined) target.title = texts.title;
  if (texts.label !== undefined) target.label = texts.label;
  if (texts.text !== undefined) target.text = texts.text;
  if (texts.append !== undefined) target.text = target.text ? `${target.text}\n${texts.append}` : texts.append;
}

function sameIndex(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, d) => v === b[d]);
}

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

type BraceCopy = NonNullable<StepBrace['copies']>[number];

function braceCopy(index: number[], title: string, subtitle: string | undefined): BraceCopy {
  return subtitle === undefined ? { index, title } : { index, title, subtitle };
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
  readonly functionTrees = new Map<Call, Tree[]>();
  private readonly callGroups = new Map<Call, string[]>();

  constructor(private readonly c: PageCompiler) {}

  compileCall(call: Call): Tree[] {
    this.shownCalls.add(call);
    const def = call.instance.def;
    const backward = this.c.backwardCalls.get(call);
    if (backward !== undefined) {
      const at = child(def.where, def.decl, 'backward_steps');
      if (def.decl.backward_steps === undefined) fail(child(def.where, def.decl, 'backward'), `${def.name} has a backward, so it needs backward_steps that show what the backward computes`);
      this.functionTrees.set(call, this.list(def.decl.backward_steps, this.c.callEnv(backward, at)));
    }
    const where = child(def.where, def.decl, 'steps');
    const steps = def.decl.steps;
    if (steps === undefined) return [];
    const env = this.c.callEnv(call, where);
    return this.list(steps, call.grid === undefined ? env : { ...env, copies: call.grid.copies });
  }

  private copyEnvs(env: Env): CopyEnv[] {
    const { copies, ...plain } = env;
    if (copies === undefined) return [{ env }];
    return copies.map((copy) => ({ index: copy.index, env: { ...plain, locals: new Map([...env.locals, ...copy.locals]) } }));
  }

  private frontIndex(env: Env): number[] | undefined {
    const copies = env.copies;
    const all = env.call?.grid?.copies;
    if (copies === undefined || copies.length === 0 || all === undefined) return undefined;
    const last = all[all.length - 1].index;
    return (copies.find((copy) => sameIndex(copy.index, last)) ?? copies[0]).index;
  }

  private frontEnv(env: Env): Env {
    const envs = this.copyEnvs(env);
    const front = this.frontIndex(env);
    return (front === undefined ? envs[0] : (envs.find((e) => e.index !== undefined && sameIndex(e.index, front)) ?? envs[0])).env;
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
      if (step.loop === undefined && step.steps === undefined && this.copyEnvs(stepEnv).every(({ env: e }) => this.isNone(step.show ?? step.id, this.field(e, step, step.show === undefined ? 'id' : 'show')))) {
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
    const text = typeof step.loop === 'string' ? step.loop : '';
    const range = RANGE_LOOP.exec(text);
    if (range !== null) fail(loopWhere, `loop: '${text}' is no longer written with range; write loop: ${range[1]}, the variable of a for loop in the forward`);
    const defaultsWhere = child(at, step, 'defaults');
    const defaults = record(step.defaults, defaultsWhere, 'defaults');
    checkKeys(defaults, LOOP_DEFAULT_KEYS, defaultsWhere);
    const rulesWhere = child(at, step, 'rules');
    const rules = step.rules ?? [];
    if (!Array.isArray(rules)) fail(rulesWhere, 'rules must be a list');
    const optionsAt = (i: number, count: number): Options => {
      const matching = rules.filter((rule, k) => {
        const ruleWhere = child(rulesWhere, rules, k);
        const spec = record(rule, ruleWhere, 'a rule');
        checkKeys(spec, RULE_KEYS, ruleWhere);
        if (typeof spec.at !== 'number' || !Number.isInteger(spec.at)) fail(child(ruleWhere, spec, 'at'), 'at must be an integer index');
        const target = spec.at < 0 ? count + spec.at : spec.at;
        if (target < 0 || target >= count) fail(child(ruleWhere, spec, 'at'), `at ${spec.at} is out of range for ${count} items`);
        return target === i;
      });
      const options = merged([defaults, ...matching]);
      delete options.at;
      return options;
    };
    const named = NAME.exec(text);
    if (named !== null) return this.rounds(String(step.id), named[1], optionsAt, { ...env, where: loopWhere }, defaultsWhere);
    const { variable, index, items } = this.listItems(text, { ...env, where: loopWhere });
    return items.map((item, i) => {
      const locals = new Map<string, Value>([...env.locals, [index, num(i)], [variable, item]]);
      const options = optionsAt(i, items.length);
      if (options.show === undefined) options.show = variable;
      return this.step(`${String(step.id)}[${i}]`, options, { ...env, locals, where: defaultsWhere });
    });
  }

  private rounds(id: string, variable: string, optionsAt: (k: number, count: number) => Options, env: Env, defaultsWhere: Where): Tree[] {
    const call = env.call;
    const def = env.instance.def;
    const rounds = call?.rounds.get(variable);
    if (rounds === undefined) return fail(env.where, `loop '${variable}' is not a loop of the forward of ${def.name}`);
    if (rounds.loops.size > 1) fail(env.where, `loop '${variable}' names ${rounds.loops.size} loops of the forward of ${def.name}; give each loop its own variable`);
    const envs = this.copyEnvs(env);
    const values = envs.map(({ env: e }) => {
      const key = rounds.outer.map((outer) => {
        const value = e.locals.get(outer);
        if (!isNum(value)) return fail(env.where, `loop '${variable}' runs inside the loop over '${outer}' in the forward of ${def.name}; put this step inside a loop step over '${outer}'`);
        return value.toy;
      });
      return rounds.values.get(key.join(',')) ?? [];
    });
    const count = Math.max(0, ...values.map((list) => list.length));
    return Array.from({ length: count }, (_, k) => {
      const options = optionsAt(k, count);
      if (options.show === undefined && options.steps === undefined) fail(defaultsWhere, `the loop step '${id}' needs show, or steps`);
      const round = (c: number) => new Map<string, Value>([[variable, num(values[c][k])], ['round', num(k)]]);
      const copies = env.copies;
      const stepEnv: Env =
        copies === undefined
          ? { ...env, locals: new Map([...env.locals, ...round(0)]), where: defaultsWhere }
          : { ...env, copies: copies.flatMap((copy, c): GridCopy[] => (values[c].length > k ? [{ index: copy.index, locals: new Map([...copy.locals, ...round(c)]) }] : [])), where: defaultsWhere };
      return this.step(`${id}[${k}]`, options, stepEnv);
    });
  }

  private listItems(text: string, env: Env): { variable: string; index: string; items: Value[] } {
    const match = LOOP.exec(text);
    if (match === null) fail(env.where, "loop must look like 'layer = layers[i]', or name the variable of a for loop in the forward, such as loop: i");
    const [, variable, listText, index] = match;
    const listValue = this.c.reference(listText, env);
    const items: Value[] = listValue instanceof InstanceList ? listValue.items : Array.isArray(listValue) ? listValue : this.c.fail(env, `'${listText}' is ${describe(listValue)}, not a list`);
    return { variable, index, items };
  }

  private isNone(show: unknown, env: Env): boolean {
    if (Array.isArray(show)) return show.length > 0 && show.every((item, k) => this.isNone(item, { ...env, where: child(env.where, show, k) }));
    if (typeof show !== 'string') return false;
    const ast = this.c.parse(show, env);
    if (ast.k !== 'name' && ast.k !== 'member') return false;
    const value = this.c.evaluate(ast, env, { k: 'ref' }, false);
    return value === null || (value instanceof Instance && value.calls.length === 0);
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
      const group = this.group(id, this.list(options.steps, this.field(env, options, 'steps')), env);
      this.groupOptions(group, options, env);
      return group;
    }
    const shown = this.shownInCopies(options.show ?? id, env, options);
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
    if (group.copies !== undefined) for (const copy of group.copies) copy.title = call.instance.name;
    const def = call.instance.def;
    const childEnv = this.c.callEnv(call);
    if (def.decl.fold !== undefined) group.fold = this.fold(def.decl.fold, { ...childEnv, where: child(def.where, def.decl, 'fold') }, group.fold);
    if (typeof def.decl.formula === 'string') group.formula = { tex: def.decl.formula, env: { ...childEnv, where: child(def.where, def.decl, 'formula') } };
    else if (def.decl.formula !== undefined) fail(child(def.where, def.decl, 'formula'), 'formula must be LaTeX text');
    this.groupOptions(group, options, env);
    return group;
  }

  private group(id: string, children: Tree[], env?: Env): Group {
    const group: Group = { kind: 'group', id, flat: false, children, fold: { forward: {}, backward: {} } };
    if (env?.copies !== undefined && env.call !== undefined) {
      group.grid = env.call.prefix;
      group.copies = env.copies.map((copy) => ({ index: copy.index }));
    }
    return group;
  }

  private leaf(id: string, nodes: Node[], env: Env): Leaf {
    const step: Rec['step'] = { show: this.withAnon(nodes), title: id };
    const call = env.call;
    if (call !== undefined && call.parent !== undefined) step.module = call.prefix;
    const leaf: Leaf = { kind: 'leaf', id, rec: { step }, brace: false };
    if (env.copies !== undefined && call !== undefined) {
      step.grid = call.prefix;
      step.copies = env.copies.map((copy) => ({ index: copy.index, title: id }));
      leaf.copies = env.copies.map((copy) => ({ index: copy.index }));
    }
    return leaf;
  }

  private shownInCopies(show: unknown, env: Env, options: Options): Shown {
    const envs = this.copyEnvs(env);
    const results = envs.map(({ env: e }) => this.shown(show, this.field(e, options, 'show')));
    if (results.length === 1) return results[0];
    const nodes: Node[] = [];
    let all = false;
    for (const result of results) {
      if (result.kind !== 'nodes') fail(child(env.where, options, 'show'), `show '${String(show)}' is a module call inside a grid module; a step of a grid module shows tensors`);
      nodes.push(...result.nodes);
      all = all || result.all;
    }
    return { kind: 'nodes', nodes: unique(nodes), all };
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
        if (value === null && Array.isArray(show)) continue;
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

  private texts(options: Options, env: Env, keys: readonly string[]): Texts & { subtitle?: string } {
    const out: Record<string, string> = {};
    for (const key of keys) if (options[key] !== undefined) out[key] = this.text(options, key, env);
    return out;
  }

  private frontOf(indices: number[][], env: Env): number {
    const all = env.call?.grid?.copies;
    const last = all === undefined ? undefined : all[all.length - 1].index;
    const at = last === undefined ? -1 : indices.findIndex((index) => sameIndex(index, last));
    return at < 0 ? 0 : at;
  }

  private perCopy(copies: { index: number[] }[], env: Env, visit: (k: number, env: Env) => void): void {
    for (const { index, env: e } of this.copyEnvs(env)) {
      const k = index === undefined ? -1 : copies.findIndex((copy) => sameIndex(copy.index, index));
      if (k >= 0) visit(k, e);
    }
  }

  private copyNotes(options: Options, env: Env): ForwardStep['notes'] {
    return this.copyEnvs(env).flatMap(({ env: e }) => notesOf(this.c, options.notes, this.field(e, options, 'notes')));
  }

  private leafOptions(leaf: Leaf, options: Options, env: Env): void {
    for (const key of ['flat', 'expand', 'fold', 'formula', 'edit']) {
      if (options[key] !== undefined) fail(child(env.where, options, key), `'${key}' applies to a step with steps`);
    }
    const step = leaf.rec.step;
    const keys = ['title', 'label', 'text', 'append', 'subtitle'];
    const copies = step.copies;
    const labels = leaf.copies;
    if (copies === undefined || labels === undefined) {
      const texts = this.texts(options, env, keys);
      applyTexts(step, texts);
      if (texts.title !== undefined) leaf.braceTitle = step.title;
      if (texts.subtitle !== undefined) leaf.subtitle = texts.subtitle;
    } else {
      this.perCopy(copies, env, (k, e) => {
        const texts = this.texts(options, e, keys);
        applyTexts(copies[k], texts);
        if (texts.title !== undefined) labels[k].title = texts.title;
        if (texts.subtitle !== undefined) labels[k].subtitle = texts.subtitle;
      });
      const front = this.frontOf(
        copies.map((copy) => copy.index),
        env,
      );
      const chosen: CopyText = copies[front];
      step.title = chosen.title;
      if (chosen.label !== undefined) step.label = chosen.label;
      if (chosen.text !== undefined) step.text = chosen.text;
      if (options.title !== undefined) leaf.braceTitle = step.title;
      if (options.subtitle !== undefined) leaf.subtitle = labels[front].subtitle;
    }
    if (options.brace !== undefined) {
      if (typeof options.brace !== 'boolean') fail(child(env.where, options, 'brace'), 'brace must be true or false');
      leaf.brace = options.brace;
    }
    if (options.notes !== undefined) step.notes = [...(step.notes ?? []), ...(this.copyNotes(options, env) ?? [])];
    if (options.backward !== undefined) step.backward = this.backward(options.backward, this.field(this.frontEnv(env), options, 'backward'), step.backward ?? {});
  }

  private groupOptions(group: Group, options: Options, env: Env): void {
    if (options.brace !== undefined) fail(child(env.where, options, 'brace'), 'a step with steps always has a brace; brace applies to a step that shows tensors');
    const front = this.frontEnv(env);
    const labels = group.copies;
    const frontLabel = labels === undefined ? 0 : this.frontOf(
      labels.map((copy) => copy.index),
      env,
    );
    if (labels === undefined) {
      if (options.title !== undefined) group.title = this.text(options, 'title', env);
      if (options.subtitle !== undefined) group.subtitle = this.text(options, 'subtitle', env);
    } else {
      this.perCopy(labels, env, (k, e) => {
        if (options.title !== undefined) labels[k].title = this.text(options, 'title', e);
        if (options.subtitle !== undefined) labels[k].subtitle = this.text(options, 'subtitle', e);
      });
      if (options.title !== undefined) group.title = labels[frontLabel].title;
      if (options.subtitle !== undefined) group.subtitle = labels[frontLabel].subtitle;
    }
    if (options.flat !== undefined) {
      if (typeof options.flat !== 'boolean') fail(child(env.where, options, 'flat'), 'flat must be true or false');
      group.flat = options.flat;
    }
    if (options.expand !== undefined) {
      const value = typeof options.expand === 'boolean' ? options.expand : this.c.expression(options.expand, this.field(front, options, 'expand'));
      if (typeof value !== 'boolean') fail(child(env.where, options, 'expand'), 'expand must be true or false');
      group.expand = value;
    }
    if (options.fold !== undefined) group.fold = this.fold(options.fold, this.field(front, options, 'fold'), group.fold);
    if (options.formula !== undefined) {
      if (typeof options.formula !== 'string') fail(child(env.where, options, 'formula'), 'formula must be LaTeX text');
      group.formula = { tex: options.formula, env: this.field(front, options, 'formula') };
    }
    if (options.edit !== undefined) {
      const where = child(env.where, options, 'edit');
      const patches = record(options.edit, where, 'edit');
      for (const [path, patch] of Object.entries(patches)) this.patch(group.children, path, patch, { ...env, where: child(where, patches, path) });
    }
    const introKeys = ['text', 'append', 'label'];
    if (options.text !== undefined && group.intro === undefined) {
      const intro: Rec = { step: { show: [], title: group.title ?? group.id } };
      const owner = group.call ?? (env.call?.parent === undefined ? undefined : env.call);
      if (owner !== undefined) intro.step.module = owner.prefix;
      if (group.grid !== undefined && labels !== undefined) {
        intro.step.grid = group.grid;
        intro.step.copies = labels.map((copy) => ({ index: copy.index, title: copy.title ?? group.title ?? group.id }));
      }
      group.intro = intro;
    }
    if (options.append !== undefined && group.intro === undefined) fail(child(env.where, options, 'append'), 'append adds to the text of a step; this step with steps has no text');
    for (const key of INTRO_KEYS) {
      if (options[key] !== undefined && group.intro === undefined) fail(child(env.where, options, key), `'${key}' applies to a step of its own; give this step text, or set ${key} on one of its steps`);
    }
    const intro = group.intro;
    if (intro !== undefined) {
      const step = intro.step;
      const copies = step.copies;
      if (copies === undefined) applyTexts(step, this.texts(options, env, introKeys));
      else {
        this.perCopy(copies, env, (k, e) => applyTexts(copies[k], this.texts(options, e, introKeys)));
        const chosen = copies[this.frontOf(
          copies.map((copy) => copy.index),
          env,
        )];
        if (chosen.label !== undefined) step.label = chosen.label;
        if (chosen.text !== undefined) step.text = chosen.text;
      }
      if (options.notes !== undefined) step.notes = [...(step.notes ?? []), ...(this.copyNotes(options, env) ?? [])];
      if (options.backward !== undefined) step.backward = this.backward(options.backward, this.field(front, options, 'backward'), step.backward ?? {});
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
            const step = tree.rec.step;
            if (step.grid !== undefined && tree.copies !== undefined) {
              brace.grid = step.grid;
              brace.copies = tree.copies.map((copy, k) => braceCopy(copy.index, copy.title ?? step.copies?.[k]?.title ?? brace.title, copy.subtitle));
            }
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
        const group: StepGroup = { id, title };
        if (tree.call !== undefined) this.callGroups.set(tree.call, [...path, id]);
        const brace: StepBrace = { id, title, nodes: framed(shows([tree]), tree.call), depth };
        if (tree.subtitle !== undefined) brace.subtitle = tree.subtitle;
        if (tree.grid !== undefined && tree.copies !== undefined) {
          group.grid = tree.grid;
          group.copies = tree.copies.map((copy) => ({ index: copy.index, title: copy.title ?? title }));
          brace.grid = tree.grid;
          brace.copies = tree.copies.map((copy) => braceCopy(copy.index, copy.title ?? title, copy.subtitle ?? tree.subtitle));
        }
        groups.push(group);
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
      if (call.parent === undefined || call.instance.def.decl.steps === undefined || this.c.inBackward(call)) continue;
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

  functionBackward(forward: ForwardStep[], groups: StepGroup[]): FunctionBackward[] {
    return [...this.functionTrees].map(([call, trees]) => {
      const inside = (module: string | undefined) => module !== undefined && (module === call.prefix || module.startsWith(`${call.prefix}.`) || module.startsWith(`${call.prefix}#`));
      const outputs = [...call.outputs.values()];
      const after = [...forward].reverse().find((step) => inside(step.module) || (Array.isArray(step.show) ? step.show : [step.show]).some((n) => outputs.includes(n)));
      const def = call.instance.def;
      if (after === undefined) fail(child(def.where, def.decl, 'backward_steps'), `no forward step shows the call '${call.prefix}', so its backward steps have no place`);
      const spec = this.c.g.functionOf(outputs[0]);
      const base = this.callGroups.get(call) ?? after.groups ?? [];
      const placed: { rec: Rec; path: string[] }[] = [];
      const walk = (list: Tree[], path: string[], key: string) => {
        for (const tree of list) {
          const id = `${key}.${tree.id}`;
          if (tree.kind === 'leaf') {
            placed.push({ rec: tree.rec, path });
            continue;
          }
          const inner = tree.flat ? path : [...path, id];
          if (!tree.flat) {
            const group: StepGroup = { id, title: tree.title ?? tree.id };
            if (tree.grid !== undefined && tree.copies !== undefined) {
              group.grid = tree.grid;
              group.copies = tree.copies.map((copy) => ({ index: copy.index, title: copy.title ?? group.title }));
            }
            groups.push(group);
          }
          if (tree.intro !== undefined) placed.push({ rec: tree.intro, path: inner });
          walk(tree.children, inner, id);
        }
      };
      walk(trees, base, `${base.at(-1) ?? call.prefix}.backward`);
      const steps = placed.map(({ rec, path }, k) => {
        const step = rec.step;
        const out: FunctionBackward['steps'][number] = {
          id: String(k),
          title: step.title,
          show: step.show,
          receive: (spec?.gradInputs ?? []).filter((g) => step.show.includes(g.grad)).map((g) => g.input),
          groups: path,
        };
        if (step.label !== undefined) out.label = step.label;
        if (step.text !== undefined) out.text = step.text;
        if (step.notes !== undefined) out.notes = step.notes;
        if (step.grid !== undefined && step.copies !== undefined) {
          out.grid = step.grid;
          out.copies = step.copies;
        }
        return out;
      });
      return { module: call.prefix, after, steps };
    });
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
