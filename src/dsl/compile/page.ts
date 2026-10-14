import { DTYPES, type Dtype, type Graph, Node, ProgramBuilder } from '../../autograd';
import type { CameraOptions, LayoutTree } from '../../scene3d';
import type { LevelSpec } from '../../scene3d/levels';
import { layoutOf } from './layout';
import { isLayoutOp } from '../../scene3d/layout';
import type { ForwardStep, PageModule, StepsSpec } from '../../steps';
import { type Nested, Tensor } from '../../tensor';
import type { PanelItem, SectionItem } from '../../ui';
import type { ModuleDef } from '../load';
import { DslError, type Where, child, fail } from '../source';
import { elemOf } from '../tex';
import { compilePanels } from './panels';
import { type SectionSources, compileSections } from './sections';
import { StepCompiler } from './steps';
import { Texts } from './texts';
import { describe } from './expressions';
import { Call, Eager, type Env, Instance, type Num, PageFunction, Result, type Value } from './types';
import { checkKeys, fieldSpec, isNum, isRecord, num, record } from './util';

export interface CompiledPage {
  graph: Graph;
  steps: StepsSpec;
  tree: LayoutTree;
  levels: LevelSpec;
  modules: PageModule[];
  panels: PanelItem[];
  sections: SectionItem[];
  nodes: Map<string, Node>;
  sequence?: PageSequence;
  page: { title: string; camera?: CameraOptions };
}

export interface PageSequence {
  tokens: string[];
  length: number;
  get(): number[];
  set(ids: number[]): void;
}

export const PAGE_KEYS: ReadonlySet<string> = new Set(['title', 'camera', 'import', 'preamble', 'model', 'config', 'data', 'modules', 'panels', 'sections']);
const PAGE_ONLY = new Set(['seed', 'tokenizer', 'real', 'dtype']);
const DEFAULT_PAGE_DTYPE: Dtype = 'fp32';
const DEFAULT_SEED = 0;

type DataValue = number | string | boolean | DataValue[];

interface DataEntry {
  name: string;
  expr?: string;
  value: Value;
  where: Where;
}

export class PageCompiler extends Texts {
  readonly rootInputs = new Set<Node>();
  readonly data = new Map<string, DataEntry>();
  readonly config = new Map<string, DataValue>();
  vocab: string[] | undefined;
  rootCall!: Call;

  constructor(
    readonly page: Record<string, unknown>,
    readonly pageWhere: Where,
  ) {
    super();
  }

  rootEnv(where: Where = this.pageWhere): Env {
    return this.callEnv(this.rootCall, where);
  }

  readonly tokensToIds = new PageFunction('tokens_to_ids', (args, kwargs, failWith) => {
    const tokens = args[0];
    if (args.length !== 1 || kwargs.size > 0 || !Array.isArray(tokens) || !tokens.every((t) => typeof t === 'string')) return failWith('tokens_to_ids takes a list of tokens');
    const vocab = this.vocab;
    if (vocab === undefined) return failWith('tokens_to_ids needs config.tokenizer.vocab');
    const ids = (tokens as string[]).map((token) => {
      const id = vocab.indexOf(token);
      if (id < 0) failWith(`token '${token}' is not in the vocab`);
      return id;
    });
    return new Eager(Tensor.from(ids, [ids.length]).asInt64(), 'int64', undefined, vocab);
  });

  dataEnv(root: ModuleDef, where: Where): Env {
    const locals = new Map<string, Value>();
    for (const [key, value] of this.config) locals.set(key, valueOf(value));
    for (const entry of this.data.values()) locals.set(entry.name, entry.value);
    locals.set(this.tokensToIds.name, this.tokensToIds);
    return { instance: new Instance(root, '', '', undefined, []), call: undefined, locals, where };
  }

  dataValue(expr: string, root: ModuleDef, where: Where): Value {
    return this.computed(expr, this.dataEnv(root, where), new ProgramBuilder(), false);
  }
}

function exprText(text: unknown, where: Where): string {
  if (typeof text !== 'string') fail(where, 'expr must be an expression string');
  return text;
}

function valueOf(value: DataValue): Value {
  if (typeof value === 'number') return num(value);
  if (Array.isArray(value)) return value.map(valueOf);
  return value;
}

function plainOf(value: Value, where: Where): DataValue {
  if (isNum(value)) return value.toy;
  if (typeof value === 'string' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.map((v) => plainOf(v, where));
  return fail(where, `a config expression gives numbers, strings, booleans or lists, got ${describe(value)}`);
}

function shapeOfData(value: Value): number[] {
  if (value instanceof Eager) return [...value.value.shape];
  return Array.isArray(value) ? [value.length, ...(value.length > 0 ? shapeOfData(value[0]) : [])] : [];
}

function nestedOf(value: Value, where: Where): Nested {
  if (value instanceof Eager) return value.value.toNested();
  if (isNum(value)) return value.toy;
  if (Array.isArray(value)) return value.map((v) => nestedOf(v, where));
  return fail(where, `expected numbers, got ${describe(value)}`);
}

function isTokenIds(value: Value): boolean {
  return value instanceof Eager && value.tokens !== undefined;
}

function literal(value: unknown, where: Where): DataValue {
  if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.map((v, i) => literal(v, child(where, value, i)));
  return fail(where, 'expected a number, a string, a boolean, a list or { expr: ... }');
}

export function compilePage(c: PageCompiler, root: ModuleDef, sectionSources: SectionSources): CompiledPage {
  const page = c.page;
  const where = c.pageWhere;
  const configWhere = child(where, page, 'config');
  const config = record(page.config, configWhere, 'config');
  const tokenizer = config.tokenizer === undefined ? undefined : record(config.tokenizer, child(configWhere, config, 'tokenizer'), 'tokenizer');
  if (tokenizer !== undefined) {
    if (!Array.isArray(tokenizer.vocab) || !tokenizer.vocab.every((t) => typeof t === 'string')) fail(child(child(configWhere, config, 'tokenizer'), tokenizer, 'vocab'), 'vocab must be a list of tokens');
    c.vocab = tokenizer.vocab as string[];
  }
  const realWhere = child(configWhere, config, 'real');
  const reals = record(config.real, realWhere, 'real');
  const realSpecs = Object.entries(reals).map(([key, spec]) => {
    const at = child(realWhere, reals, key);
    const out = record(spec, at, 'a real size');
    if (typeof out.name !== 'string') fail(at, 'a real size needs a name');
    return out as { name: string } & Record<string, number | string>;
  });
  const real = realSpecs[0];
  c.realSizes = Object.fromEntries(Object.keys(reals).map((key, i) => [key, realSpecs[i]]));
  const seed = config.seed ?? DEFAULT_SEED;
  if (typeof seed !== 'number' || !Number.isInteger(seed)) fail(child(configWhere, config, 'seed'), 'seed must be an integer');
  const configValues = c.config;
  for (const [key, value] of Object.entries(config)) {
    if (PAGE_ONLY.has(key)) continue;
    const at = child(configWhere, config, key);
    if (isRecord(value)) {
      checkKeys(value, new Set(['expr']), at);
      const exprWhere = child(at, value, 'expr');
      configValues.set(key, plainOf(c.dataValue(exprText(value.expr, exprWhere), root, exprWhere), exprWhere));
    } else configValues.set(key, literal(value, at));
  }
  const realOf = (name: string, toy: number): number => {
    const value = real?.[name];
    return typeof value === 'number' ? value : toy;
  };
  const dataWhere = child(where, page, 'data');
  const data = record(page.data, dataWhere, 'data');
  for (const [name, value] of Object.entries(data)) {
    const at = child(dataWhere, data, name);
    if (isRecord(value)) {
      checkKeys(value, new Set(['expr']), at);
      const exprWhere = child(at, value, 'expr');
      const expr = exprText(value.expr, exprWhere);
      c.data.set(name, { name, expr, value: c.dataValue(expr, root, exprWhere), where: exprWhere });
    } else c.data.set(name, { name, value: valueOf(literal(value, at)), where: at });
  }

  const argsDecl = record(root.decl.args, child(root.where, root.decl, 'args'), 'args');
  const dims = new Map<string, Num>();
  const rootArgs = new Map<string, { value: Value; where: Where }>();
  for (const [name, spec] of Object.entries(argsDecl)) {
    const type = typeof spec === 'string' ? spec : isRecord(spec) ? spec.type : undefined;
    const raw = configValues.get(name);
    if (raw === undefined) continue;
    const at = child(configWhere, config, name);
    let value: Value;
    if (typeof raw === 'number') {
      value = type === 'dim' || type === 'int' ? { toy: raw, real: realOf(name, raw), dim: name } : { toy: raw, real: realOf(name, raw) };
      if (type === 'dim' || type === 'int') dims.set(name, value);
    } else if (typeof raw === 'string' && type === 'dtype') {
      if (!(DTYPES as readonly string[]).includes(raw)) fail(at, `unknown dtype '${raw}'`);
      value = { dtype: raw as (typeof DTYPES)[number] };
    } else if (typeof raw === 'string' && type === 'module') {
      const module = root.namespace.get(raw);
      if (module === undefined) fail(at, `module '${raw}' is not imported`);
      value = { module };
    } else if (typeof raw === 'boolean') value = raw;
    else return fail(at, `config '${name}' cannot be passed to arg '${name}' of ${root.name}`);
    rootArgs.set(name, { value, where: at });
  }
  const dtypeWhere = child(configWhere, config, 'dtype');
  const pageDtype = config.dtype ?? DEFAULT_PAGE_DTYPE;
  if (typeof pageDtype !== 'string' || !(DTYPES as readonly string[]).includes(pageDtype)) fail(dtypeWhere, `dtype must be one of ${DTYPES.join(', ')}`);
  if ('dtype' in argsDecl && config.dtype !== undefined) rootArgs.set('dtype', { value: { dtype: pageDtype as Dtype }, where: dtypeWhere });
  for (const name of configValues.keys()) {
    if (!(name in argsDecl)) fail(child(configWhere, config, name), `config '${name}' is not an arg of ${root.name}`);
  }

  const inputsWhere = child(root.where, root.decl, 'inputs');
  const inputs = record(root.decl.inputs, inputsWhere, 'inputs');
  const bound = new Map<string, Num>();
  for (const [name, spec] of Object.entries(inputs)) {
    const at = child(inputsWhere, inputs, name);
    const field = record(fieldSpec(spec), at, `input '${name}'`);
    const entry = c.data.get(name);
    if (entry === undefined) fail(dataWhere, `data needs '${name}', an input of ${root.name}`);
    if (field.type !== 'tensor' || !Array.isArray(field.shape)) continue;
    const shape = shapeOfData(entry.value);
    if (shape.length !== field.shape.length) fail(entry.where, `'${name}' has shape (${shape.join(', ')}), but ${root.name} expects (${field.shape.join(', ')})`);
    field.shape.forEach((dim, d) => {
      if (typeof dim !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(dim) || dim in argsDecl) return;
      const known = bound.get(dim);
      if (known !== undefined && known.toy !== shape[d]) fail(entry.where, `'${name}' has ${shape[d]} for ${dim}, but another input has ${known.toy}`);
      bound.set(dim, { toy: shape[d], real: realOf(dim, shape[d]), dim });
    });
  }
  for (const [name, value] of bound) dims.set(name, value);
  c.setupGraph({ dims, real, dtype: pageDtype, seed }, where);

  const rootInstance = c.instantiate(root, '', '', undefined, [], rootArgs, child(where, page, 'model'));
  const call = new Call(rootInstance, '', undefined, new Map());
  c.rootCall = call;
  for (const [name, value] of bound) call.bound.set(name, value);
  for (const [name, spec] of Object.entries(inputs)) {
    const at = child(inputsWhere, inputs, name);
    const field = record(fieldSpec(spec), at, `input '${name}'`);
    const entry = c.data.get(name) as DataEntry;
    const node = rootInput(c, name, field, entry, at);
    c.rootInputs.add(node);
    c.nodes.set(name, node);
    c.bindField(call, name, node, spec, c.callEnv(call, at), (message) => fail(entry.where, `'${name}': ${message}`), true);
    call.values.set(name, node);
    call.inputs.set(name, node);
  }
  rootInstance.calls.push(call);
  c.calls.push(call);
  c.runForward(call);
  call.created = [...c.g.nodes];
  c.checkOutputs(call);

  const sc = new StepCompiler(c);
  const trees = sc.compileCall(call);
  sc.showCallInputs(trees);
  const { forward, groups, braces } = sc.flatten(trees);
  placeLeaves(c, forward);
  const others = forward.filter((s) => !sc.allSteps.has(s));
  const drawn = [...new Set(others.flatMap((s) => (Array.isArray(s.show) ? s.show : [s.show])))];
  for (const step of sc.allSteps) step.show = [...new Set([...drawn, ...(Array.isArray(step.show) ? step.show : [step.show])])];
  sc.markFormulas(trees);
  checkShown(c, sc, forward);
  const visible = new Set(forward.flatMap((s) => (Array.isArray(s.show) ? s.show : [s.show])));
  const modules = sc.modules(visible);
  const steps: StepsSpec = { forward, groups, braces, backward: 'reverse', modules };
  const env = c.rootEnv();
  const compiled: CompiledPage = {
    graph: c.g,
    steps,
    tree: { kind: 'module', children: c.g.nodes.filter((n) => visible.has(n)).map((node) => ({ kind: 'node', node })) },
    levels: layoutOf(c, call),
    modules,
    panels: compilePanels(c, env),
    sections: compileSections(c, env, sectionSources),
    nodes: c.nodes,
    page: { title: String(page.title) },
  };
  if (isRecord(page.camera)) compiled.page.camera = page.camera as CameraOptions;
  const sequence = pageSequence(c, root, env);
  if (sequence !== undefined) compiled.sequence = sequence;
  return compiled;
}

function rootInput(c: PageCompiler, name: string, field: Record<string, unknown>, entry: DataEntry, at: Where): Node {
  try {
    if (field.type === 'list') {
      if (field.dtype !== 'str') fail(child(at, field, 'dtype'), 'a list input of the page model holds tokens, so its dtype is str');
      const vocab = c.vocab;
      if (vocab === undefined) fail(entry.where, `'${name}' holds tokens, so the page needs config.tokenizer.vocab`);
      if (!Array.isArray(entry.value) || !entry.value.every((t) => typeof t === 'string')) fail(entry.where, `'${name}' must be a list of tokens`);
      const ids = entry.value.map((token) => {
        const id = vocab.indexOf(token as string);
        if (id < 0) fail(entry.where, `token '${String(token)}' is not in the vocab`);
        return id;
      });
      const node = c.g.label(name, ids, { tokens: vocab, tex: `\\mathrm{${name}}`, elem: `\\mathrm{${name}}` });
      c.tokenLists.add(node);
      return node;
    }
    const shape = (field.shape as unknown[]).map((dim, d) => {
      if (typeof dim === 'string' && c.g.dims[dim] !== undefined) return dim;
      return c.dimSpec(c.number(dim, { ...c.rootEnv(), where: child(child(at, field, 'shape'), field.shape as unknown[], d) }));
    });
    const dtype = c.expression(field.dtype, { ...c.rootEnv(), where: child(at, field, 'dtype') });
    if (!isRecord(dtype) || typeof dtype.dtype !== 'string') fail(child(at, field, 'dtype'), 'dtype must be a dtype');
    const tex = c.memberTex(c.rootCall.instance, name);
    const values = nestedOf(entry.value, entry.where);
    if (isTokenIds(entry.value)) return c.g.label(name, values, { tokens: c.vocab, shape, tex, elem: elemOf(tex), dtype: dtype.dtype as string });
    return c.g.input(name, shape, values, { tex, elem: elemOf(tex), dtype: dtype.dtype as string });
  } catch (error) {
    if (error instanceof DslError) throw error;
    return fail(entry.where, (error as Error).message);
  }
}

function pageSequence(c: PageCompiler, root: ModuleDef, env: Env): PageSequence | undefined {
  const panels = Array.isArray(c.page.panels) ? c.page.panels : [];
  const panelIndex = panels.findIndex((p) => isRecord(p) && p.sequence !== undefined);
  if (panelIndex < 0) return undefined;
  const panel = panels[panelIndex] as Record<string, unknown>;
  const at = child(child(c.pageWhere, c.page, 'panels'), panels, panelIndex);
  const tokens = c.node(panel.tokens, { ...env, where: child(at, panel, 'tokens') });
  if (!c.tokenLists.has(tokens)) fail(child(at, panel, 'tokens'), `'${String(panel.tokens)}' is not a list of tokens in data`);
  for (const key of ['input', 'label']) {
    const node = c.node(panel[key], { ...env, where: child(at, panel, key) });
    if (!c.rootInputs.has(node)) fail(child(at, panel, key), `'${String(panel[key])}' is not an input of the page model`);
  }
  const vocab = c.vocab as string[];
  const tokenName = [...c.data.keys()].find((name) => c.nodes.get(name) === tokens) as string;
  const length = tokens.shape[0];
  const derived = [...c.data.values()].filter((entry) => entry.expr !== undefined && c.nodes.get(entry.name) !== undefined);
  const g = c.g;
  return {
    tokens: [...vocab],
    length,
    get: () => Array.from(tokens.value.data, Math.round),
    set: (next) => {
      if (next.length !== length || !next.every((id) => Number.isInteger(id) && id >= 0 && id < vocab.length)) {
        throw new Error(`sequence.set: expected ${length} token ids between 0 and ${vocab.length - 1}, got [${next.join(', ')}]`);
      }
      const entry = c.data.get(tokenName) as DataEntry;
      entry.value = next.map((id) => vocab[id]);
      g.batch(() => {
        tokens.set(next);
        for (const item of derived) {
          item.value = c.dataValue(item.expr as string, root, item.where);
          (c.nodes.get(item.name) as Node).set(nestedOf(item.value, item.where));
        }
      });
    },
  };
}

function placeLeaves(c: PageCompiler, forward: ForwardStep[]): void {
  const shown = new Set(forward.flatMap((s) => (Array.isArray(s.show) ? s.show : [s.show])));
  const leaves = c.g.nodes.filter((n) => (n.kind === 'param' || n.kind === 'buffer') && !shown.has(n));
  for (const leaf of leaves) {
    const step = forward.find((s) => (Array.isArray(s.show) ? s.show : [s.show]).some((n) => n.inputs.includes(leaf)));
    if (step === undefined) continue;
    step.show = [leaf, ...(Array.isArray(step.show) ? step.show : [step.show])];
  }
}

function drawnOn(node: Node, shown: (n: Node) => boolean): boolean {
  if (shown(node)) return true;
  return isLayoutOp(node) && drawnOn(node.inputs[0], shown);
}

function checkShown(c: PageCompiler, sc: StepCompiler, forward: ForwardStep[]): void {
  const shown = new Set(forward.flatMap((s) => (Array.isArray(s.show) ? s.show : [s.show])));
  const inputsOfShown = new Set([...sc.shownCalls].flatMap((call) => [...call.inputs.values()]));
  for (const call of sc.shownCalls) {
    const def = call.instance.def;
    if (def.decl.steps === undefined) continue;
    const unshownWhere = child(def.where, def.decl, 'unshown');
    const unshown = def.decl.unshown ?? [];
    if (!Array.isArray(unshown) || !unshown.every((n) => typeof n === 'string')) fail(unshownWhere, 'unshown must be a list of forward names');
    for (const [i, name] of unshown.entries()) {
      if (!call.results.includes(name as string)) fail(child(unshownWhere, unshown, i), `'${String(name)}' is not a forward result of ${def.name}`);
    }
    const forwardWhere = child(def.where, def.decl, 'forward');
    for (const name of call.results) {
      if (unshown.includes(name)) continue;
      const value = call.values.get(name);
      const producer = value instanceof Result ? value.call : value instanceof Node ? c.producers.get(value) : undefined;
      if (value instanceof Node && drawnOn(value, (n) => shown.has(n) || inputsOfShown.has(n))) continue;
      if (producer !== undefined && producer !== call && sc.shownCalls.has(producer)) continue;
      const index = (def.decl.forward as Record<string, unknown>[]).findIndex((item) => isRecord(item) && name in item);
      fail(child(forwardWhere, def.decl.forward as unknown[], index), `'${name}' of ${def.name} is not shown by any step; show it, show the module call that computes it, or list it in unshown`);
    }
  }
}
