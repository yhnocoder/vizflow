import type { Graph, Node } from '../../autograd';
import type { CameraOptions, LayoutEntry, LayoutTree } from '../../scene3d';
import { type ForwardStep, type InferSpec, type PageModule, type StepsSpec, type UpdateSpec, isLeafLike } from '../../steps';
import { type Nested, inferShape } from '../../tensor';
import type { PanelItem, SectionItem } from '../../ui';
import { DslError, type Where, child, fail } from '../source';
import { defaultTex, elemOf } from '../tex';
import { compileBody } from './body';
import { declareBuffers, declareParams, declareWeights, define } from './declare';
import { markFormulas } from './formula';
import { GENERATE_KEYS, OPTIMIZER_KEYS, OVERVIEW_KEYS, SEQUENCE_KEYS, TEXT_KEYS } from './keys';
import { pageModules, placeSteps } from './modules';
import { compilePanels } from './panels';
import { type SectionSources, compileSections } from './sections';
import { notesOf } from './notes';
import { moduleBackward, moduleSteps } from './steps';
import type { Compiler } from './text';
import type { Env } from './types';
import { checkKeys, dtypeOption, isRecord, list, record } from './util';

export interface CompiledPage {
  graph: Graph;
  steps: StepsSpec;
  tree: LayoutTree;
  modules: PageModule[];
  panels: PanelItem[];
  sections: SectionItem[];
  nodes: Map<string, Node>;
  sequence?: PageSequence;
  modes?: { train: StepsSpec; infer: StepsSpec };
  layout?: unknown;
  page: { title: string; lede?: string; hint?: string; camera?: CameraOptions };
}

export interface PageSequence {
  prompt?: number;
  tokens: string[];
  length: number;
  get(): number[];
  set(ids: number[]): void;
}

export function assemblePage(c: Compiler, sectionSources: SectionSources): CompiledPage {
  const page = c.page;
  const env: Env = { scope: c.root, loops: new Map(), where: c.pageWhere };
  const tree: LayoutTree = { kind: 'module', children: [] };
  declareParams(c, c.root, page, env, undefined, {}, c.pageWhere);
  const inputsWhere = child(c.pageWhere, page, 'inputs');
  const inputs = record(page.inputs, inputsWhere, 'inputs');
  const vocab = page.vocab;
  const sequence = sequenceInputs(c, env, tree);
  for (const [name, spec] of Object.entries(inputs)) {
    const at = child(inputsWhere, inputs, name);
    if (!isRecord(spec)) fail(at, 'a page input needs tokens or value');
    const given = c.takeValue(name);
    let node: Node;
    const tex = defaultTex(name);
    try {
      if (Array.isArray(spec.tokens)) {
        node = c.g.label(name, given ?? tokenIds(c, spec.tokens as unknown[], child(at, spec, 'tokens')), { tokens: vocab as string[], tex, elem: elemOf(tex) });
      } else if (spec.value !== undefined) {
        const values = (given ?? spec.value) as Nested;
        node = c.g.input(name, inferShape(values), values, { tex, elem: elemOf(tex), ...dtypeOption(spec, at) });
      } else {
        fail(at, 'a page input needs tokens or value');
      }
    } catch (error) {
      if (error instanceof DslError) throw error;
      fail(at, (error as Error).message);
    }
    define(c, c.root, name, node, { ...env, where: at });
    tree.children.push({ kind: 'node', node });
    c.setOwner(node, tree.children);
  }
  declareWeights(c, c.root, page, env, undefined, tree.children);
  declareBuffers(c, c.root, page, env, tree.children);
  const { slots } = compileBody(c, c.root, page, env, tree.children);
  const steps = moduleSteps(c, c.root, page, env, slots);
  moduleBackward(c, page, env);
  c.checkValuesUsed();
  placeLeaves(c, tree);
  const forward = steps.map((s) => s.step);
  for (const entry of c.backward) {
    const target = forward.find((step) => list(step.show).filter((n) => !isLeafLike(n)).includes(entry.node));
    if (target === undefined) fail(entry.where, `node '${entry.node.name}' is not computed in any forward step`);
    target.backward = { ...target.backward, ...entry.text };
  }
  markFormulas(c, steps);
  const modules = pageModules(c);
  placeSteps(c, forward, modules);
  const leaves = leavesStep(c, env);
  const overviewWhere = child(c.pageWhere, page, 'overview');
  const overview = page.overview === undefined ? undefined : record(page.overview, overviewWhere, 'overview');
  const stepsSpec: StepsSpec = { forward: [...leaves, ...forward], backward: 'reverse', modules };
  if (overview !== undefined) {
    const texts = c.texts(overview, { ...env, where: overviewWhere }, OVERVIEW_KEYS);
    stepsSpec.overview = { title: texts.title ?? String(page.title), text: texts.text ?? '' };
    if (texts.label !== undefined) stepsSpec.overview.label = texts.label;
    if (overview.notes !== undefined) stepsSpec.overview.notes = notesOf(c, overview.notes, { ...env, where: child(overviewWhere, overview, 'notes') });
    if (overview.backward !== undefined) stepsSpec.overview.backward = c.texts(overview.backward, { ...env, where: child(overviewWhere, overview, 'backward') }, TEXT_KEYS);
  }
  warnDuplicateTex(c);
  const compiled: CompiledPage = {
    graph: c.g,
    steps: stepsSpec,
    tree,
    modules,
    panels: compilePanels(c, env),
    sections: compileSections(c, env, sectionSources),
    nodes: c.nodes,
    page: {
      title: String(page.title),
      lede: page.lede === undefined ? undefined : c.text(page.lede, { ...env, where: child(c.pageWhere, page, 'lede') }),
      hint: typeof page.hint === 'string' ? page.hint : undefined,
      camera: isRecord(page.camera) ? (page.camera as CameraOptions) : undefined,
    },
  };
  if (page.optimizer !== undefined) stepsSpec.update = updateSpec(c);
  if (sequence !== undefined) compiled.sequence = sequence;
  if (sequence?.prompt !== undefined) {
    generate(c.g, sequence);
    compiled.modes = { train: stepsSpec, infer: { ...stepsSpec, update: undefined, infer: inferSpec(c, sequence.prompt) } };
    compiled.steps = compiled.modes.infer;
  }
  if (page.layout !== undefined) compiled.layout = page.layout;
  return compiled;
}

function tokenIds(c: Compiler, tokens: unknown[], where: Where): number[] {
  const vocab = c.page.vocab;
  if (!Array.isArray(vocab)) fail(where, 'tokens need a vocab list on the page');
  return tokens.map((token, i) => {
    const id = (vocab as unknown[]).indexOf(token);
    if (id < 0) fail(child(where, tokens, i), `token '${String(token)}' is not in vocab`);
    return id;
  });
}

function sequenceInputs(c: Compiler, env: Env, tree: LayoutTree): PageSequence | undefined {
  const page = c.page;
  if (page.sequence === undefined) return undefined;
  const where = child(c.pageWhere, page, 'sequence');
  const spec = record(page.sequence, where, 'sequence');
  checkKeys(spec, SEQUENCE_KEYS, where);
  if (!Array.isArray(spec.tokens) || spec.tokens.length < 2) fail(child(where, spec, 'tokens'), 'sequence needs a list of at least two tokens');
  for (const key of ['input', 'label']) if (typeof spec[key] !== 'string') fail(child(where, spec, key), `sequence needs '${key}', the name of the node it creates`);
  const prompt = tokenIds(c, spec.tokens as unknown[], child(where, spec, 'tokens'));
  const steps = generateSteps(c);
  const ids = [...prompt, ...Array<number>(steps ?? 0).fill(0)];
  const T = ids.length - 1;
  const rows = c.root.dims.get('T');
  if (rows !== undefined && rows.toy !== T) {
    const reason = steps === undefined ? `sequence has ${ids.length} tokens` : `the prompt has ${prompt.length} tokens and generate adds ${steps}`;
    fail(child(where, spec, 'tokens'), `${reason}, so T must be ${T}, but dims gives T = ${rows.toy}`);
  }
  const tokens = [...(page.vocab as string[])];
  const make = (key: 'input' | 'label', values: number[]) => {
    const name = spec[key] as string;
    const tex = defaultTex(name);
    const node = c.g.label(name, values, { tokens, tex, elem: elemOf(tex) });
    define(c, c.root, name, node, { ...env, where: child(where, spec, key) });
    tree.children.push({ kind: 'node', node });
    c.setOwner(node, tree.children);
    return node;
  };
  const input = make('input', ids.slice(0, T));
  const label = make('label', ids.slice(1));
  const g = c.g;
  return {
    ...(steps === undefined ? {} : { prompt: prompt.length }),
    tokens,
    length: ids.length,
    get: () => [...Array.from(input.value.data, Math.round), Math.round(label.value.data[T - 1])],
    set: (next) => {
      if (next.length !== ids.length || !next.every((id) => Number.isInteger(id) && id >= 0 && id < tokens.length)) {
        throw new Error(`sequence.set: expected ${ids.length} token ids between 0 and ${tokens.length - 1}, got [${next.join(', ')}]`);
      }
      g.batch(() => {
        input.set(next.slice(0, T));
        label.set(next.slice(1));
      });
    },
  };
}

function leavesStep(c: Compiler, env: Env): ForwardStep[] {
  const page = c.page;
  if (page.leaves === undefined) return [];
  const where = child(c.pageWhere, page, 'leaves');
  const texts = c.texts(page.leaves, { ...env, where }, TEXT_KEYS);
  if (texts.title === undefined) fail(where, 'leaves needs a title');
  const show = c.g.nodes.filter((n) => isLeafLike(n));
  return [{ kind: 'inputs', show, title: texts.title, label: texts.label ?? texts.title, text: texts.text ?? '' }];
}

function warnDuplicateTex(c: Compiler): void {
  const byTex = new Map<string, string[]>();
  for (const node of c.g.nodes) byTex.set(node.tex, [...(byTex.get(node.tex) ?? []), node.name]);
  for (const [tex, names] of byTex) {
    if (names.length > 1) console.warn(`vizflow dsl: nodes ${names.map((n) => `'${n}'`).join(', ')} share the tex '${tex}'`);
  }
}

function placeLeaves(c: Compiler, tree: LayoutTree): void {
  const placed = new Set<Node>();
  const walk = (entries: LayoutEntry[]) => {
    for (const entry of entries) {
      if (entry.kind === 'node') placed.add(entry.node);
      else if (entry.kind === 'parallel') entry.branches.forEach((b) => walk(b.children));
      else walk(entry.children);
    }
  };
  walk(tree.children);
  const leaves = c.g.nodes.filter((n) => (n.kind === 'param' || n.kind === 'buffer') && !placed.has(n));
  const uses = (node: Node, leaf: Node) => node.inputs.some((input) => input === leaf || (input.op === 'transpose' && input.inputs[0] === leaf));
  const insert = (entries: LayoutEntry[], leaf: Node): boolean => {
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      if (entry.kind === 'node') {
        if (uses(entry.node, leaf)) {
          entries.splice(i, 0, { kind: 'node', node: leaf });
          return true;
        }
      } else if (entry.kind === 'parallel') {
        if (entry.branches.some((b) => insert(b.children, leaf))) return true;
      } else if (insert(entry.children, leaf)) {
        return true;
      }
    }
    return false;
  };
  for (const leaf of leaves) {
    if (!insert(tree.children, leaf)) (c.ownerOf(leaf) ?? tree.children).push({ kind: 'node', node: leaf });
  }
}

function generateSteps(c: Compiler): number | undefined {
  const page = c.page;
  if (page.generate === undefined) return undefined;
  const where = child(c.pageWhere, page, 'generate');
  const spec = record(page.generate, where, 'generate');
  checkKeys(spec, GENERATE_KEYS, where);
  if (typeof spec.steps !== 'number' || !Number.isInteger(spec.steps) || spec.steps < 2) fail(child(where, spec, 'steps'), 'generate.steps must be an integer of at least 2');
  return spec.steps;
}

function nllOf(g: Graph): Node {
  const nll = g.nodes.find((n) => n.op === 'nll_loss');
  if (nll === undefined || nll.inputs[1]?.kind !== 'label') throw new Error('generate: the page needs an nll_loss of the log probabilities and the sequence label');
  return nll;
}

function generate(g: Graph, sequence: PageSequence): void {
  const logits = nllOf(g).inputs[0];
  const ids = sequence.get();
  const V = logits.shape[1];
  for (let t = (sequence.prompt as number) - 1; t < ids.length - 1; t++) {
    const row = logits.value.data.subarray(t * V, (t + 1) * V);
    ids[t + 1] = row.indexOf(Math.max(...row));
    sequence.set(ids);
  }
}

function inferSpec(c: Compiler, prompt: number): InferSpec {
  const [logits, label] = nllOf(c.g).inputs;
  const input = c.g.nodes.find((n) => n.kind === 'label' && n !== label && n.shape[0] === label.shape[0]) as Node;
  return { prompt, input, label, logits, caches: c.caches };
}

function updateSpec(c: Compiler): UpdateSpec {
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
  return { lr: number('lr', 0.1), weightDecay: number('weight_decay', 0), title: '参数更新：AdamW 第一步', label: '更新' };
}
