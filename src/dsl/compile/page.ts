import { DTYPES, type Graph, Node } from '../../autograd';
import type { CameraOptions, LayoutTree } from '../../scene3d';
import { isLayoutOp } from '../../scene3d/layout';
import type { ForwardStep, PageModule, StepsSpec } from '../../steps';
import type { Nested } from '../../tensor';
import type { PanelItem, SectionItem } from '../../ui';
import { type Ast, ParseError, parseExpr } from '../expr';
import type { ModuleDef } from '../load';
import { DslError, type Where, child, fail } from '../source';
import { elemOf } from '../tex';
import { compilePanels } from './panels';
import { type SectionSources, compileSections } from './sections';
import { StepCompiler } from './steps';
import { Texts } from './texts';
import { Call, type Env, type Num, Result, type Value } from './types';
import { checkKeys, isRecord, record } from './util';

export interface CompiledPage {
  graph: Graph;
  steps: StepsSpec;
  tree: LayoutTree;
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
const PAGE_ONLY = new Set(['seed', 'tokenizer', 'real']);
const DEFAULT_SEED = 0;

type DataValue = number | string | boolean | DataValue[];

interface DataEntry {
  name: string;
  ast?: Ast;
  value: DataValue;
  ids: boolean;
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

  dataValue(ast: Ast, where: Where, config: ReadonlyMap<string, DataValue>): { value: DataValue; ids: boolean } {
    const failAt = (message: string, at: number): never => fail(where, message, at);
    const evaluate = (node: Ast): { value: DataValue; ids: boolean } => {
      switch (node.k) {
        case 'num':
        case 'str':
        case 'bool':
          return { value: node.v, ids: false };
        case 'list':
          return { value: node.items.map((item) => evaluate(item).value), ids: false };
        case 'neg': {
          const inner = evaluate(node.arg).value;
          if (typeof inner !== 'number') return failAt('only numbers can be negated', node.pos);
          return { value: -inner, ids: false };
        }
        case 'name': {
          const entry = this.data.get(node.name);
          if (entry !== undefined) return { value: entry.value, ids: entry.ids };
          const value = config.get(node.name);
          if (value === undefined) return failAt(`name '${node.name}' is not defined in data or config`, node.pos);
          return { value, ids: false };
        }
        case 'index': {
          const base = evaluate(node.obj);
          if (!Array.isArray(base.value)) return failAt('only lists can be indexed', node.pos);
          const list = base.value;
          if (node.items.length !== 1 || node.items[0].k === 'ellipsis') return failAt('a list takes one index or slice', node.pos);
          const item = node.items[0];
          const int = (ast: Ast | undefined, fallback: number): number => {
            if (ast === undefined) return fallback;
            const v = evaluate(ast).value;
            if (typeof v !== 'number' || !Number.isInteger(v)) return failAt('an index must be an integer', ast.pos);
            return Math.min(list.length, Math.max(0, v < 0 ? list.length + v : v));
          };
          if (item.k === 'slice') return { value: list.slice(int(item.start, 0), int(item.stop, list.length)), ids: base.ids };
          const at = int(item.value, 0);
          if (at >= list.length) return failAt('index out of range', node.pos);
          return { value: list[at], ids: false };
        }
        case 'call': {
          if (node.fn.k !== 'name' || node.fn.name !== 'tokens_to_ids' || node.args.length !== 1 || node.kwargs.length > 0) return failAt('data expressions can call tokens_to_ids(tokens)', node.pos);
          const tokens = evaluate(node.args[0]).value;
          if (!Array.isArray(tokens) || !tokens.every((t) => typeof t === 'string')) return failAt('tokens_to_ids takes a list of tokens', node.pos);
          const vocab = this.vocab;
          if (vocab === undefined) return failAt('tokens_to_ids needs config.tokenizer.vocab', node.pos);
          return {
            value: tokens.map((token) => {
              const id = vocab.indexOf(token as string);
              if (id < 0) failAt(`token '${String(token)}' is not in the vocab`, node.pos);
              return id;
            }),
            ids: true,
          };
        }
        default:
          return failAt('data expressions use names, indexes, slices and tokens_to_ids', node.pos);
      }
    };
    return evaluate(ast);
  }
}

function parseAt(text: unknown, where: Where): Ast {
  if (typeof text !== 'string') fail(where, 'expr must be an expression string');
  try {
    return parseExpr(text);
  } catch (error) {
    if (error instanceof ParseError) fail(where, `${error.message} in '${text}'`, error.offset);
    throw error;
  }
}

function shapeOfData(value: DataValue): number[] {
  return Array.isArray(value) ? [value.length, ...(value.length > 0 ? shapeOfData(value[0]) : [])] : [];
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
  const seed = config.seed ?? DEFAULT_SEED;
  if (typeof seed !== 'number' || !Number.isInteger(seed)) fail(child(configWhere, config, 'seed'), 'seed must be an integer');
  const configValues = c.config;
  for (const [key, value] of Object.entries(config)) {
    if (PAGE_ONLY.has(key)) continue;
    const at = child(configWhere, config, key);
    if (isRecord(value)) {
      checkKeys(value, new Set(['expr']), at);
      configValues.set(key, c.dataValue(parseAt(value.expr, child(at, value, 'expr')), child(at, value, 'expr'), configValues).value);
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
      const ast = parseAt(value.expr, exprWhere);
      c.data.set(name, { name, ast, ...c.dataValue(ast, exprWhere, configValues), where: exprWhere });
    } else c.data.set(name, { name, value: literal(value, at), ids: false, where: at });
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
      value = { toy: raw, real: realOf(name, raw) };
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
  for (const name of configValues.keys()) {
    if (!(name in argsDecl)) fail(child(configWhere, config, name), `config '${name}' is not an arg of ${root.name}`);
  }

  const inputsWhere = child(root.where, root.decl, 'inputs');
  const inputs = record(root.decl.inputs, inputsWhere, 'inputs');
  const bound = new Map<string, Num>();
  for (const [name, spec] of Object.entries(inputs)) {
    const at = child(inputsWhere, inputs, name);
    const field = record(spec, at, `input '${name}'`);
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
  const dtypeArg = rootArgs.get('dtype')?.value;
  c.setupGraph({ dims, real, dtype: dtypeArg !== undefined && isRecord(dtypeArg) && typeof dtypeArg.dtype === 'string' ? dtypeArg.dtype : 'fp32', seed }, where);

  const rootInstance = c.instantiate(root, '', '', undefined, [], rootArgs, child(where, page, 'model'));
  const call = new Call(rootInstance, '', undefined, new Map());
  c.rootCall = call;
  for (const [name, value] of bound) call.bound.set(name, value);
  for (const [name, spec] of Object.entries(inputs)) {
    const at = child(inputsWhere, inputs, name);
    const field = record(spec, at, `input '${name}'`);
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
  const entries = sc.compileCall(call);
  const { forward, modules } = sc.flatten(entries);
  placeLeaves(c, forward);
  sc.markFormulas(entries, modules);
  checkShown(c, sc, forward);
  const inputsStep = sc.special.inputs;
  const steps: StepsSpec = { forward: inputsStep === undefined ? forward : [inputsStep, ...forward], backward: 'reverse', modules };
  if (sc.special.overview !== undefined) steps.overview = sc.special.overview;
  const env = c.rootEnv();
  const compiled: CompiledPage = {
    graph: c.g,
    steps,
    tree: { kind: 'module', children: c.g.nodes.filter((n) => steps.forward.some((s) => (Array.isArray(s.show) ? s.show : [s.show]).includes(n))).map((node) => ({ kind: 'node', node })) },
    modules,
    panels: compilePanels(c, env),
    sections: compileSections(c, env, sectionSources),
    nodes: c.nodes,
    page: { title: String(page.title) },
  };
  if (isRecord(page.camera)) compiled.page.camera = page.camera as CameraOptions;
  const sequence = pageSequence(c, env);
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
    if (entry.ids) return c.g.label(name, entry.value as Nested, { tokens: c.vocab, shape, tex, elem: elemOf(tex), dtype: dtype.dtype as string });
    return c.g.input(name, shape, entry.value as Nested, { tex, elem: elemOf(tex), dtype: dtype.dtype as string });
  } catch (error) {
    if (error instanceof DslError) throw error;
    return fail(entry.where, (error as Error).message);
  }
}

function pageSequence(c: PageCompiler, env: Env): PageSequence | undefined {
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
  const derived = [...c.data.values()].filter((entry) => entry.ast !== undefined && c.nodes.get(entry.name) !== undefined);
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
          item.value = c.dataValue(item.ast as Ast, item.where, c.config).value;
          (c.nodes.get(item.name) as Node).set(item.value as Nested);
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
