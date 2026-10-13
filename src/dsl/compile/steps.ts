import { Node } from '../../autograd';
import { type BackwardOverride, type ForwardStep, type StepModule, type StepsSpec, isLeafLike } from '../../steps';
import { type Where, child, fail, merged } from '../source';
import { markSteps } from '../tex';
import { describe } from './expressions';
import { notesOf } from './notes';
import type { PageCompiler } from './page';
import { type Call, type Env, Instance, InstanceList, Result, type Value } from './types';
import { checkKeys, isRecord, join, num, record } from './util';

const EXPLANATION_KEYS = ['label', 'title', 'subtitle', 'text', 'append'];
const OPTION_KEYS: ReadonlySet<string> = new Set([...EXPLANATION_KEYS, 'show', 'segment', 'brace', 'notes', 'formula', 'folded', 'beside', 'inline', 'fold', 'backward', 'steps']);
const STEP_KEYS: ReadonlySet<string> = new Set([...OPTION_KEYS, 'id']);
const LOOP_KEYS: ReadonlySet<string> = new Set(['id', 'loop', 'defaults', 'rules']);
const RULE_KEYS: ReadonlySet<string> = new Set([...OPTION_KEYS, 'at']);
const BACKWARD_KEYS: ReadonlySet<string> = new Set(['label', 'title', 'text', 'append', 'notes']);
const FOLD_KEYS: ReadonlySet<string> = new Set(['label', 'title', 'subtitle', 'text', 'append', 'backward']);
const BRACE_KEYS: ReadonlySet<string> = new Set(['title', 'subtitle']);
const LOOP = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*$/;

type Options = Record<string, unknown>;

export interface Rec {
  step: ForwardStep & { show: Node[] };
  merged?: Rec;
}

interface Fold {
  forward: BackwardOverride;
  backward: BackwardOverride;
}

interface Formula {
  tex: string;
  env: Env;
}

interface Box {
  title: string;
  subtitle: string;
  fold: Fold;
  formula?: Formula;
  folded: boolean;
  inline: boolean;
  brace?: { title: string; subtitle?: string; formula?: string };
  segment?: string;
  nodes: Node[];
  calls: Call[];
  beside?: { target: string; where: Where };
  where: Where;
}

export type Entry =
  | { kind: 'step'; id: string; rec: Rec; beside?: { target: string; where: Where } }
  | { kind: 'box'; id: string; key: string; box: Box; children: Entry[]; group: boolean };

export interface Special {
  overview?: NonNullable<StepsSpec['overview']>;
  inputs?: ForwardStep;
}

function isEntryBox(entry: Entry): entry is Extract<Entry, { kind: 'box' }> {
  return entry.kind === 'box';
}

function leaves(entries: Entry[]): Rec[] {
  return entries.flatMap((entry) => (entry.kind === 'step' ? [entry.rec] : leaves(entry.children)));
}

function live(rec: Rec): Rec {
  return rec.merged === undefined ? rec : live(rec.merged);
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

export class StepCompiler {
  readonly shownCalls = new Set<Call>();
  readonly special: Special = {};
  private readonly marks = new Map<Rec, string>();

  constructor(private readonly c: PageCompiler) {}

  compileCall(call: Call): Entry[] {
    this.shownCalls.add(call);
    const def = call.instance.def;
    const where = child(def.where, def.decl, 'steps');
    const steps = def.decl.steps;
    if (steps === undefined) return [];
    if (!Array.isArray(steps)) fail(where, 'steps must be a list');
    const seen = new Set<string>();
    const out: Entry[] = [];
    steps.forEach((spec, i) => {
      const at = child(where, steps, i);
      const step = record(spec, at, 'a step');
      if (typeof step.id !== 'string') fail(at, 'a step needs an id');
      if (seen.has(step.id)) fail(child(at, step, 'id'), `step id '${step.id}' is used twice`);
      seen.add(step.id);
      const env = this.c.callEnv(call, at);
      if (step.loop !== undefined) out.push(...this.loop(step, env));
      else {
        checkKeys(step, STEP_KEYS, at);
        out.push(...this.step(step.id, step, env));
      }
    });
    this.besides(out);
    return out;
  }

  private loop(step: Options, env: Env): Entry[] {
    const at = env.where;
    checkKeys(step, LOOP_KEYS, at);
    const loopWhere = child(at, step, 'loop');
    const match = typeof step.loop === 'string' ? LOOP.exec(step.loop) : null;
    if (match === null) fail(loopWhere, "loop must look like 'layer = layers[i]'");
    const [, variable, listText, index] = match;
    const listValue = this.c.reference(listText, { ...env, where: loopWhere });
    const items: Value[] = listValue instanceof InstanceList ? listValue.items : Array.isArray(listValue) ? listValue : this.c.fail({ ...env, where: loopWhere }, `'${listText}' is ${describe(listValue)}, not a list`);
    const defaults = record(step.defaults, child(at, step, 'defaults'), 'defaults');
    checkKeys(defaults, OPTION_KEYS, child(at, step, 'defaults'));
    const rulesWhere = child(at, step, 'rules');
    const rules = step.rules ?? [];
    if (!Array.isArray(rules)) fail(rulesWhere, 'rules must be a list');
    return items.flatMap((item, i) => {
      const locals = new Map<string, Value>([...env.locals, [index, num(i)], [variable, item]]);
      const loopEnv: Env = { ...env, locals };
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
      if (options.show === undefined) options.show = variable;
      return this.step(`${String(step.id)}[${i}]`, options, loopEnv, child(at, step, 'defaults'));
    });
  }

  private field(env: Env, options: Options, key: string, base: Where): Env {
    return { ...env, where: child(base, options, key) };
  }

  step(id: string, options: Options, env: Env, base: Where = env.where): Entry[] {
    const show = options.show ?? id;
    const showEnv = this.field(env, options, 'show', base);
    if (isRecord(show) && show.all !== undefined) {
      if (show.all !== true || Object.keys(show).length !== 1) fail(showEnv.where, 'write show: { all: true }');
      this.overview(options, env, base);
      return [];
    }
    if (isRecord(show) && (show.inputs !== undefined || show.parameters !== undefined)) {
      checkKeys(show, new Set(['inputs', 'parameters']), showEnv.where);
      this.inputsStep(show, options, env, base);
      return [];
    }
    if (isRecord(show) && show.producer_of !== undefined) {
      checkKeys(show, new Set(['producer_of']), showEnv.where);
      const node = this.c.node(show.producer_of, { ...showEnv, where: child(showEnv.where, show, 'producer_of') });
      const producer = this.c.producers.get(node);
      if (producer === undefined) fail(showEnv.where, `'${String(show.producer_of)}' is not the output of a module call`);
      return [this.callEntry(id, producer, options, env, base)];
    }
    const refs = Array.isArray(show) ? show : [show];
    const values = refs.map((ref, k) => {
      const at = Array.isArray(show) ? child(showEnv.where, show, k) : showEnv.where;
      return { value: this.c.reference(ref, { ...env, where: at }), where: at, ref };
    });
    if (values.every(({ value }) => value instanceof Node)) return [this.tensorEntry(id, values.map(({ value }) => value as Node), options, env, base)];
    const calls = values.map(({ value, where, ref }) => {
      const call = value instanceof Instance ? this.c.onlyCall(value, { ...env, where }, 0) : value instanceof Result ? value.call : undefined;
      if (call === undefined) fail(where, `show '${String(ref)}' is ${describe(value)}; show takes tensors or module instances`);
      return { call, ref: String(ref) };
    });
    if (calls.length === 1) return [this.callEntry(id, calls[0].call, options, env, base)];
    return [this.groupEntry(id, calls, options, env, base)];
  }

  private overview(options: Options, env: Env, base: Where): void {
    for (const key of Object.keys(options)) {
      if (!['id', 'show', 'label', 'title', 'text', 'notes', 'backward'].includes(key)) fail(child(base, options, key), `the overview step does not take '${key}'`);
    }
    const title = options.title === undefined ? '' : this.c.text(options.title, this.field(env, options, 'title', base));
    const overview: NonNullable<Special['overview']> = { title, text: this.c.text(options.text, this.field(env, options, 'text', base)) };
    if (options.label !== undefined) overview.label = this.c.text(options.label, this.field(env, options, 'label', base));
    if (options.notes !== undefined) overview.notes = notesOf(this.c, options.notes, this.field(env, options, 'notes', base));
    if (options.backward !== undefined) overview.backward = this.backward(options.backward, this.field(env, options, 'backward', base), {});
    this.special.overview = overview;
  }

  private inputsStep(show: Options, options: Options, env: Env, base: Where): void {
    for (const key of Object.keys(options)) {
      if (!['id', 'show', 'label', 'title', 'text', 'notes', 'segment'].includes(key)) fail(child(base, options, key), `the inputs step does not take '${key}'`);
    }
    const nodes: Node[] = [];
    const g = this.c.g;
    if (show.inputs === true) nodes.push(...[...this.c.rootInputs].filter((n) => !this.c.tokenLists.has(n)));
    if (show.parameters === true) nodes.push(...g.nodes.filter((n) => n.kind === 'param' || (n.kind === 'op' && n.op === 'transpose' && n.inputs[0].kind === 'param')));
    const step: ForwardStep = { kind: 'inputs', show: nodes, title: this.c.text(options.title ?? '', this.field(env, options, 'title', base)) };
    if (options.label !== undefined) step.label = this.c.text(options.label, this.field(env, options, 'label', base));
    if (options.text !== undefined) step.text = this.c.text(options.text, this.field(env, options, 'text', base));
    if (options.segment !== undefined) step.segment = this.c.text(options.segment, this.field(env, options, 'segment', base));
    if (options.notes !== undefined) step.notes = notesOf(this.c, options.notes, this.field(env, options, 'notes', base));
    this.special.inputs = step;
  }

  private withAnon(nodes: Node[]): Node[] {
    return unique(nodes.flatMap((n) => [...(this.c.anon.get(n) ?? []), n]));
  }

  private stepless(call: Call): Node[] {
    const created = call.created;
    const used = created.flatMap((n) => n.inputs).filter((n) => n.kind === 'param' || n.kind === 'buffer');
    return unique([...used, ...created]);
  }

  private tensorEntry(id: string, show: Node[], options: Options, env: Env, base: Where): Entry {
    for (const key of ['subtitle', 'fold', 'folded', 'inline', 'steps', 'formula']) {
      if (options[key] !== undefined) fail(child(base, options, key), `'${key}' applies to a step that shows a module call`);
    }
    const rec: Rec = { step: { show: this.withAnon(show), title: id } };
    this.apply(rec, options, env, base);
    const entry: Entry = { kind: 'step', id, rec };
    if (options.beside !== undefined) entry.beside = this.besideOf(options, base);
    return entry;
  }

  private besideOf(options: Options, base: Where): { target: string; where: Where } {
    const where = child(base, options, 'beside');
    if (typeof options.beside !== 'string') fail(where, 'beside must name an earlier step');
    return { target: options.beside, where };
  }

  private apply(rec: Rec, options: Options, env: Env, base: Where): void {
    const step = rec.step;
    const text = (key: string) => this.c.text(options[key], this.field(env, options, key, base));
    if (options.title !== undefined) step.title = text('title');
    if (options.label !== undefined) step.label = text('label');
    if (options.text !== undefined) step.text = text('text');
    if (options.append !== undefined) step.text = step.text ? `${step.text}\n${text('append')}` : text('append');
    if (options.segment !== undefined) step.segment = text('segment');
    if (options.brace !== undefined) step.brace = this.brace(options.brace, this.field(env, options, 'brace', base));
    if (options.notes !== undefined) step.notes = [...(step.notes ?? []), ...notesOf(this.c, options.notes, this.field(env, options, 'notes', base))];
    if (options.backward !== undefined) step.backward = this.backward(options.backward, this.field(env, options, 'backward', base), step.backward ?? {});
  }

  private brace(value: unknown, env: Env): { title: string; subtitle?: string } {
    const spec = record(value, env.where, 'brace');
    checkKeys(spec, BRACE_KEYS, env.where);
    if (spec.title === undefined) fail(env.where, 'brace needs a title');
    const out: { title: string; subtitle?: string } = { title: this.c.text(spec.title, { ...env, where: child(env.where, spec, 'title') }) };
    if (spec.subtitle !== undefined) out.subtitle = this.c.text(spec.subtitle, { ...env, where: child(env.where, spec, 'subtitle') });
    return out;
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
    if (spec.subtitle !== undefined) fail(child(env.where, spec, 'subtitle'), 'a folded box keeps the subtitle of the box');
    const backward = spec.backward === undefined ? { ...current.backward } : this.backward(spec.backward, { ...env, where: child(env.where, spec, 'backward') }, current.backward);
    return { forward, backward };
  }

  private callEntry(id: string, call: Call, options: Options, env: Env, base: Where): Entry {
    const def = call.instance.def;
    if (def.decl.steps === undefined) {
      this.shownCalls.add(call);
      return this.tensorEntry(id, this.stepless(call), options, env, base);
    }
    const childEnv = this.c.callEnv(call);
    let fold: Fold = { forward: {}, backward: {} };
    if (def.decl.fold !== undefined) fold = this.fold(def.decl.fold, { ...childEnv, where: child(def.where, def.decl, 'fold') }, fold);
    const box: Box = { title: call.instance.name, subtitle: '', fold, folded: false, inline: false, nodes: call.created, calls: [call], where: base };
    if (typeof def.decl.formula === 'string') box.formula = { tex: def.decl.formula, env: { ...childEnv, where: child(def.where, def.decl, 'formula') } };
    else if (def.decl.formula !== undefined) fail(child(def.where, def.decl, 'formula'), 'formula must be LaTeX text');
    const entry: Extract<Entry, { kind: 'box' }> = { kind: 'box', id, key: call.prefix, box, children: this.compileCall(call), group: false };
    this.boxOptions(entry, options, env, base);
    return entry;
  }

  private groupEntry(id: string, calls: { call: Call; ref: string }[], options: Options, env: Env, base: Where): Entry {
    const children = calls.map(({ call, ref }) => this.callEntry(ref, call, {}, env, base));
    const parentKey = env.call?.prefix ?? '';
    const box: Box = { title: id, subtitle: '', fold: { forward: {}, backward: {} }, folded: false, inline: false, nodes: unique(calls.flatMap(({ call }) => call.created)), calls: calls.map(({ call }) => call), where: base };
    const entry: Extract<Entry, { kind: 'box' }> = { kind: 'box', id, key: join(parentKey, id), box, children, group: true };
    this.boxOptions(entry, options, env, base);
    return entry;
  }

  private boxOptions(entry: Extract<Entry, { kind: 'box' }>, options: Options, env: Env, base: Where): void {
    const box = entry.box;
    for (const key of ['label', 'backward']) {
      if (options[key] !== undefined) fail(child(base, options, key), `'${key}' applies to a step that shows tensors; set it on a step of the module with steps:`);
    }
    const text = (key: string) => this.c.text(options[key], this.field(env, options, key, base));
    if (options.title !== undefined) box.title = text('title');
    if (options.subtitle !== undefined) box.subtitle = text('subtitle');
    if (options.segment !== undefined) box.segment = text('segment');
    if (options.fold !== undefined) box.fold = this.fold(options.fold, this.field(env, options, 'fold', base), box.fold);
    if (options.formula !== undefined) {
      if (typeof options.formula !== 'string') fail(child(base, options, 'formula'), 'formula must be LaTeX text');
      box.formula = { tex: options.formula, env: this.field(env, options, 'formula', base) };
    }
    for (const key of ['folded', 'inline'] as const) {
      if (options[key] === undefined) continue;
      if (typeof options[key] !== 'boolean') fail(child(base, options, key), `${key} must be true or false`);
      box[key] = options[key] as boolean;
    }
    if (options.brace !== undefined) box.brace = this.brace(options.brace, this.field(env, options, 'brace', base));
    if (options.beside !== undefined) box.beside = this.besideOf(options, base);
    if (options.steps !== undefined) {
      const where = child(base, options, 'steps');
      const patches = record(options.steps, where, 'steps');
      for (const [path, patch] of Object.entries(patches)) this.patch(entry.children, path, patch, env, child(where, patches, path));
      this.besides(entry.children);
    }
    const first = leaves(entry.children)[0];
    if (options.text !== undefined || options.append !== undefined || options.notes !== undefined) {
      if (first === undefined) fail(base, 'this module call has no steps to put the text on');
      if (options.text !== undefined) first.step.text = first.step.text ? `${text('text')}\n${first.step.text}` : text('text');
      if (options.append !== undefined) first.step.text = first.step.text ? `${first.step.text}\n${text('append')}` : text('append');
      if (options.notes !== undefined) first.step.notes = [...(first.step.notes ?? []), ...notesOf(this.c, options.notes, this.field(env, options, 'notes', base))];
    }
  }

  private patch(entries: Entry[], path: string, value: unknown, env: Env, where: Where): void {
    const target = this.find(entries, path.split('.'), where, path);
    const options = record(value, where, 'a step patch');
    checkKeys(options, OPTION_KEYS, where);
    if (options.show !== undefined) fail(child(where, options, 'show'), 'a patch cannot change what a step shows');
    if (target.kind === 'step') {
      for (const key of ['subtitle', 'fold', 'folded', 'inline', 'steps', 'formula']) {
        if (options[key] !== undefined) fail(child(where, options, key), `'${key}' applies to a step that shows a module call`);
      }
      this.apply(target.rec, options, env, where);
      if (options.beside !== undefined) target.beside = this.besideOf(options, where);
      return;
    }
    this.boxOptions(target, options, env, where);
  }

  private find(entries: Entry[], parts: string[], where: Where, path: string): Entry {
    const [head, ...rest] = parts;
    const direct = entries.find((e) => e.id === head);
    const nested = direct ?? entries.filter(isEntryBox).filter((e) => e.group).map((e) => e.children.find((c) => c.id === head)).find((e) => e !== undefined);
    if (nested === undefined) fail(where, `there is no step '${path}'; the steps here are ${entries.map((e) => e.id).join(', ') || '(none)'}`);
    if (rest.length === 0) return nested;
    if (nested.kind !== 'box') fail(where, `step '${head}' shows tensors, so it has no step '${rest.join('.')}'`);
    return this.find(nested.children, rest, where, path);
  }

  private besides(entries: Entry[]): void {
    for (const entry of [...entries]) {
      const beside = entry.kind === 'step' ? entry.beside : entry.box.beside;
      if (beside === undefined) continue;
      const at = entries.indexOf(entry);
      const target = entries.slice(0, at).find((e) => e.id === beside.target);
      if (target === undefined) fail(beside.where, `beside '${beside.target}': no earlier step with that id`);
      if (entry.kind === 'step') {
        if (target.kind !== 'step') fail(beside.where, `beside '${beside.target}': a tensor step can only stand beside a tensor step`);
        this.merge(entry.rec, target.rec);
      } else {
        if (target.kind !== 'box') fail(beside.where, `beside '${beside.target}': a module call can only stand beside a module call`);
        const mine = leaves(entry.children);
        const theirs = leaves(target.children);
        if (mine.length !== theirs.length) fail(beside.where, `beside '${beside.target}': ${entry.id} has ${mine.length} steps, ${target.id} has ${theirs.length}`);
        mine.forEach((rec, i) => this.merge(rec, theirs[i]));
      }
      if (entry.kind === 'step') entry.beside = undefined;
      else entry.box.beside = undefined;
    }
  }

  private merge(rec: Rec, into: Rec): void {
    const target = live(into);
    target.step.show = unique([...target.step.show, ...rec.step.show]);
    target.step.label = combineLabels(target.step.label, rec.step.label);
    const backward = combineLabels(target.step.backward?.label, rec.step.backward?.label);
    if (backward !== undefined) target.step.backward = { ...target.step.backward, label: backward };
    rec.merged = target;
  }

  flatten(entries: Entry[]): { forward: ForwardStep[]; modules: StepModule[] } {
    const forward: ForwardStep[] = [];
    const modules: StepModule[] = [];
    const visible = new Set(leaves(entries).filter((r) => r.merged === undefined).flatMap((r) => r.step.show));
    const consumers = new Map<Node, Node[]>();
    for (const u of this.c.g.nodes) {
      for (const n of u.inputs) {
        const list = consumers.get(n);
        if (list === undefined) consumers.set(n, [u]);
        else list.push(u);
      }
    }
    const walk = (list: Entry[], chain: { module: StepModule; box: Box }[], braces: Box['brace'][]) => {
      for (const entry of list) {
        if (entry.kind === 'step') {
          if (entry.rec.merged !== undefined) continue;
          const step = entry.rec.step;
          const owner = chain[chain.length - 1];
          if (owner !== undefined) step.module = owner.module.id;
          const brace = braces[braces.length - 1];
          if (step.brace === undefined && brace !== undefined) step.brace = brace;
          if (step.segment === undefined) step.segment = chain.length > 0 ? chain[0].module.segment : (step.label ?? step.title);
          if (chain.length > 1) step.sub = owner.module.title;
          forward.push(step);
          continue;
        }
        const box = entry.box;
        if (box.inline) {
          walk(entry.children, chain, box.brace === undefined ? braces : [...braces, box.brace]);
          continue;
        }
        if (box.brace !== undefined) fail(box.where, `brace applies to an inline module call; '${entry.id}' has its own box`);
        const parent = chain[chain.length - 1];
        const nodes = box.nodes.filter((n) => !isLeafLike(n) && visible.has(n));
        const module: StepModule = {
          id: entry.key,
          parent: parent?.module.id ?? null,
          title: box.title,
          subtitle: box.subtitle,
          inputs: [],
          outputs: [],
          nodes,
          params: [],
          folded: box.folded,
          segment: box.segment ?? (chain.length > 0 ? chain[0].module.segment : box.title),
          forward: box.fold.forward,
          backward: box.fold.backward,
        };
        if (chain.length > 0) module.sub = box.title;
        const inside = new Set(nodes);
        const used = nodes.flatMap((n) => n.inputs);
        module.inputs = unique(used.filter((n) => !inside.has(n) && !isLeafLike(n)));
        module.params = unique(used.flatMap((n) => (isLeafLike(n) && n.kind === 'op' ? [n, n.inputs[0]] : [n])).filter((n) => isLeafLike(n) && n.kind !== 'label')).sort((a, b) => a.id - b.id);
        const outputs = nodes.filter((n) => (consumers.get(n) ?? []).some((u) => !inside.has(u)));
        module.outputs = outputs.length > 0 ? outputs : nodes.slice(-1);
        modules.push(module);
        walk(entry.children, [...chain, { module, box }], braces);
      }
    };
    walk(entries, [], []);
    return { forward, modules };
  }

  markFormulas(entries: Entry[], modules: StepModule[]): void {
    const visit = (list: Entry[]) => {
      for (const entry of list) {
        if (entry.kind !== 'box') continue;
        const box = entry.box;
        if (box.formula !== undefined) {
          const formula = this.formula(box.formula, entry.children);
          if (box.inline) {
            if (box.brace !== undefined) box.brace.formula = formula;
          } else {
            const module = modules.find((m) => m.id === entry.key);
            if (module !== undefined) module.formula = formula;
          }
        }
        visit(entry.children);
      }
    };
    visit(entries);
  }

  private formula(formula: Formula, scope: Entry[]): string {
    const env = formula.env;
    const resolve = (path: string): Rec[] => {
      if (path === '*') return leaves(scope);
      const all = path.endsWith('.*');
      const parts = (all ? path.slice(0, -2) : path).split('.');
      const target = this.find(scope, parts, env.where, path);
      if (target.kind === 'step') return [target.rec];
      if (!all) fail(env.where, `\\step{${path}}: '${path}' is a module call; write \\step{${path}.*} for all of its steps`);
      return leaves(target.children);
    };
    const mark = (spec: string): string => {
      const recs = unique(spec.split(',').flatMap((path) => resolve(path.trim())).map(live));
      return recs
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
    };
    try {
      return markSteps(this.c.expandTex(formula.tex, env), mark);
    } catch (error) {
      if (error instanceof Error && error.name === 'DslError') throw error;
      return fail(env.where, (error as Error).message);
    }
  }
}
