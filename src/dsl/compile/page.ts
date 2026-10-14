import { DTYPES, type Dtype, type Graph, Node, ProgramBuilder } from '../../autograd';
import type { CameraOptions, LayoutTree } from '../../scene3d';
import type { LevelSpec } from '../../scene3d/levels';
import { layoutOf } from './layout';
import { isLayoutOp } from '../../scene3d/layout';
import { type ForwardStep, type InferSpec, type PageModule, type SampleDraw, type SamplerSpec, type StepsSpec, type UpdateSpec, checkCache, drawToken, uniforms } from '../../steps';
import { type Nested, Tensor } from '../../tensor';
import type { PanelItem, SectionItem } from '../../ui';
import type { ModuleDef } from '../load';
import { DslError, type Where, child, fail } from '../source';
import { elemOf } from '../tex';
import { PENDING_INFER, compilePanels, panelModes } from './panels';
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
  modes?: { train: StepsSpec; infer: StepsSpec };
  modePanels?: { train: PanelItem[]; infer: PanelItem[] };
  train?: CompiledPage;
  page: { title: string; camera?: CameraOptions };
}

export interface PageSequence {
  prompt?: number;
  tokens: string[];
  length: number;
  get(): number[];
  set(ids: number[]): void;
}

export const PAGE_KEYS: ReadonlySet<string> = new Set(['title', 'camera', 'import', 'preamble', 'model', 'config', 'data', 'modules', 'panels', 'sections', 'generate', 'optimizer']);
const GENERATE_KEYS: ReadonlySet<string> = new Set(['steps', 'temperature', 'top_k', 'top_p', 'seed', 'note', 'config']);
const OPTIMIZER_KEYS: ReadonlySet<string> = new Set(['type', 'lr', 'weight_decay']);
const DEFAULT_ADAMW_LR = 0.1;
const SAMPLER_ONLY = ['top_k', 'top_p', 'seed', 'note'];
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
    readonly trainTokens?: string[],
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
  const overrides = c.trainTokens === undefined ? inferConfig(c) : undefined;
  const entries: [string, unknown, Where][] = Object.entries(config).map(([key, value]) => [key, value, child(configWhere, config, key)]);
  if (overrides !== undefined) for (const [key, value] of Object.entries(overrides.config)) entries.push([key, value, child(overrides.where, overrides.config, key)]);
  for (const [key, value, at] of entries) {
    if (PAGE_ONLY.has(key)) continue;
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
  const steps = generateSteps(c);
  const prompt = steps === undefined ? undefined : promptTokens(c, data, dataWhere);
  for (const [name, value] of Object.entries(data)) {
    const at = child(dataWhere, data, name);
    if (prompt !== undefined && name === prompt.name) {
      const tokens = c.trainTokens ?? [...prompt.tokens, ...Array<string>(steps as number).fill((c.vocab as string[])[0])];
      c.data.set(name, { name, value: tokens, where: at });
    } else if (isRecord(value)) {
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
    else if (typeof raw === 'string' && type === 'str') value = raw;
    else if (Array.isArray(raw) && type === 'list') {
      if (!raw.every((item) => typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean')) fail(at, `config '${name}' must be a list of strings, numbers or booleans`);
      value = raw.map((item) => (typeof item === 'number' ? num(item) : item));
    } else return fail(at, `config '${name}' cannot be passed to arg '${name}' of ${root.name}`);
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
    if (entry === undefined && field.optional === true) continue;
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
    const entry = c.data.get(name);
    if (entry === undefined) {
      call.values.set(name, null);
      continue;
    }
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
  const spec: StepsSpec = { forward, groups, braces, backward: 'reverse', modules };
  if (page.optimizer !== undefined) spec.update = updateSpec(c);
  const env = c.rootEnv();
  const compiled: CompiledPage = {
    graph: c.g,
    steps: spec,
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
  if (c.trainTokens !== undefined) {
    const modes = panelModes(c);
    compiled.panels = compiled.panels.filter((_, i) => modes[i] !== 'infer');
  } else if (prompt !== undefined) {
    if (sequence === undefined) fail(child(where, page, 'generate'), 'generate needs a sequence panel that names the tokens, input and label');
    sequence.prompt = prompt.tokens.length;
    const sampler = samplerSpec(c);
    const base = inferSpec(c, sequence);
    const infer: InferSpec = { ...base, draws: generate(base, sequence, sampler), ...(sampler === undefined ? {} : { sampler }) };
    compiled.modes = { train: spec, infer: { ...spec, update: undefined, infer } };
    compiled.steps = compiled.modes.infer;
    const modes = panelModes(c);
    const panels = compiled.panels.map((item) => (typeof item === 'object' && 'generation' in item ? { ...item, generation: infer } : item));
    const inMode = (mode: 'train' | 'infer') => panels.filter((_, i) => modes[i] === undefined || modes[i] === mode);
    compiled.modePanels = { train: inMode('train'), infer: inMode('infer') };
    compiled.panels = compiled.modePanels.infer;
  } else if (compiled.panels.some((item) => typeof item === 'object' && 'generation' in item && item.generation === PENDING_INFER)) {
    fail(child(where, page, 'panels'), 'a generation panel needs generate on the page');
  }
  return compiled;
}

function generateSteps(c: PageCompiler): number | undefined {
  const page = c.page;
  if (page.generate === undefined) return undefined;
  const where = child(c.pageWhere, page, 'generate');
  const spec = record(page.generate, where, 'generate');
  checkKeys(spec, GENERATE_KEYS, where);
  if (typeof spec.steps !== 'number' || !Number.isInteger(spec.steps) || spec.steps < 2) fail(child(where, spec, 'steps'), 'generate.steps must be an integer of at least 2');
  return spec.steps;
}

function inferConfig(c: PageCompiler): { config: Record<string, unknown>; where: Where } | undefined {
  if (!isRecord(c.page.generate) || c.page.generate.config === undefined) return undefined;
  const where = child(child(c.pageWhere, c.page, 'generate'), c.page.generate, 'config');
  return { config: record(c.page.generate.config, where, 'generate.config'), where };
}

export function hasInferConfig(page: Record<string, unknown>): boolean {
  return isRecord(page.generate) && page.generate.config !== undefined;
}

export function withTrainPage(compiled: CompiledPage, train: CompiledPage): CompiledPage {
  const modes = compiled.modes as NonNullable<CompiledPage['modes']>;
  const modePanels = compiled.modePanels as NonNullable<CompiledPage['modePanels']>;
  return { ...compiled, train, modes: { ...modes, train: train.steps }, modePanels: { ...modePanels, train: train.panels } };
}

function sequencePanel(c: PageCompiler): Record<string, unknown> | undefined {
  const panels = Array.isArray(c.page.panels) ? c.page.panels : [];
  return panels.find((p): p is Record<string, unknown> => isRecord(p) && p.sequence !== undefined);
}

function promptTokens(c: PageCompiler, data: Record<string, unknown>, dataWhere: Where): { name: string; tokens: string[] } {
  const where = child(c.pageWhere, c.page, 'generate');
  const name = sequencePanel(c)?.tokens;
  if (typeof name !== 'string') return fail(where, 'generate needs a sequence panel whose tokens names the prompt in data');
  const tokens = data[name];
  if (!Array.isArray(tokens) || tokens.length < 1 || !tokens.every((t) => typeof t === 'string')) return fail(child(dataWhere, data, name), `'${name}' is the prompt of generate, so it must be a list of tokens`);
  if (c.vocab === undefined) return fail(where, 'generate needs config.tokenizer.vocab');
  return { name, tokens: tokens as string[] };
}

function samplerSpec(c: PageCompiler): SamplerSpec | undefined {
  const where = child(c.pageWhere, c.page, 'generate');
  const spec = c.page.generate as Record<string, unknown>;
  const at = (key: string) => child(where, spec, key);
  const { temperature, top_k: topK, top_p: topP, seed, note } = spec;
  if (temperature === undefined || temperature === 0) {
    const why = temperature === undefined ? 'needs a temperature above 0; without one generate is greedy' : 'has no effect when temperature is 0, which is greedy decoding';
    for (const key of SAMPLER_ONLY) if (spec[key] !== undefined) fail(at(key), `generate.${key} ${why}`);
    return undefined;
  }
  if (typeof temperature !== 'number' || !(temperature > 0)) fail(at('temperature'), 'generate.temperature must be a number of at least 0');
  if (topK !== undefined && (typeof topK !== 'number' || !Number.isInteger(topK) || topK < 1)) fail(at('top_k'), 'generate.top_k must be a positive integer');
  if (topP !== undefined && (typeof topP !== 'number' || !(topP > 0 && topP <= 1))) fail(at('top_p'), 'generate.top_p must be a number in (0, 1]');
  if (seed !== undefined && (typeof seed !== 'number' || !Number.isInteger(seed) || seed < 0)) fail(at('seed'), 'generate.seed must be a non-negative integer');
  const text = note === undefined ? undefined : c.text(note, c.rootEnv(at('note')));
  return { temperature, seed: (seed as number | undefined) ?? 0, ...(text === undefined ? {} : { note: text }), ...(topK === undefined ? {} : { topK: topK as number }), ...(topP === undefined ? {} : { topP: topP as number }) };
}

function logProbsOf(g: Graph, label: Node): Node | undefined {
  const fromLabel = new Set<Node>([label]);
  for (const n of g.nodes) if (n.kind === 'op' && (n.op === 'view' || n.op === 'reshape' || n.op === 'unsqueeze') && fromLabel.has(n.inputs[0])) fromLabel.add(n);
  const pick = g.nodes.find((n) => n.kind === 'op' && (n.op === 'gather' || n.op === 'nll_loss') && fromLabel.has(n.inputs[1]));
  return pick?.inputs[0];
}

function inferSpec(c: PageCompiler, sequence: PageSequence): Omit<InferSpec, 'draws'> {
  const panel = sequencePanel(c) as Record<string, unknown>;
  const panels = c.page.panels as unknown[];
  const at = child(child(c.pageWhere, c.page, 'panels'), panels, panels.indexOf(panel));
  const env = c.rootEnv();
  const input = c.node(panel.input, { ...env, where: child(at, panel, 'input') });
  const label = c.node(panel.label, { ...env, where: child(at, panel, 'label') });
  const logits = logProbsOf(c.g, label);
  if (logits === undefined) fail(child(at, panel, 'label'), `generate samples from the log probabilities that '${String(panel.label)}' picks from, but no gather reads '${String(panel.label)}'`);
  const scored = new Set<Node>();
  for (const n of c.g.nodes) if (n.inputs.some((i) => i === label || scored.has(i))) scored.add(n);
  const scope = new Set(c.g.nodes.filter((n) => !scored.has(n)));
  const prompt = sequence.prompt as number;
  const problem = checkCache(c.g, label.shape[0], prompt, c.caches, scope, [logits]);
  if (problem !== undefined) fail(child(c.pageWhere, c.page, 'generate'), problem);
  return { prompt, input, label, logits, caches: c.caches };
}

function generate(infer: Omit<InferSpec, 'draws'>, sequence: PageSequence, sampler: SamplerSpec | undefined): Map<number, SampleDraw> {
  const { logits } = infer;
  const ids = sequence.get();
  const V = logits.shape[1];
  const random = uniforms(sampler?.seed ?? 0);
  const draws = new Map<number, SampleDraw>();
  for (let t = infer.prompt - 1; t < ids.length - 1; t++) {
    const draw = drawToken(logits.value.data.subarray(t * V, (t + 1) * V), sampler, random());
    draws.set(t, draw);
    ids[t + 1] = draw.token;
    sequence.set(ids);
  }
  return draws;
}

function updateSpec(c: PageCompiler): UpdateSpec {
  const page = c.page;
  const where = child(c.pageWhere, page, 'optimizer');
  const spec = record(page.optimizer, where, 'optimizer');
  checkKeys(spec, OPTIMIZER_KEYS, where);
  if (spec.type !== 'adamw') fail(child(where, spec, 'type'), "optimizer.type must be 'adamw'");
  const number = (key: string, fallback: number) => {
    const value = spec[key] ?? fallback;
    if (typeof value !== 'number' || value < 0) fail(child(where, spec, key), `optimizer.${key} must be a non-negative number`);
    return value;
  };
  return { lr: number('lr', DEFAULT_ADAMW_LR), weightDecay: number('weight_decay', 0), title: '参数更新：AdamW 第一步', label: '更新' };
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
    for (const { name, value, where } of call.resultValues) {
      if (unshown.includes(name)) continue;
      const producer = value instanceof Result ? value.call : value instanceof Node ? c.producers.get(value) : undefined;
      if (value instanceof Node && drawnOn(value, (n) => shown.has(n) || inputsOfShown.has(n))) continue;
      if (producer !== undefined && producer !== call && sc.shownCalls.has(producer)) continue;
      fail(where, `'${name}' of ${def.name} is not shown by any step; show it, show the module call that computes it, or list it in unshown`);
    }
  }
}
