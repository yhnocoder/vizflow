import type { Node } from '../../autograd';
import type { InferSpec } from '../../steps';
import type { PanelItem } from '../../ui';
import { type Where, child, fail } from '../source';
import type { PageCompiler } from './page';
import type { Env } from './types';
import { checkKeys, isRecord, record } from './util';

const TEXT_KEYS = ['title', 'note'];
const PANEL_MODES = ['train', 'infer'] as const;

export type PanelMode = (typeof PANEL_MODES)[number];

export const PENDING_INFER = Object.freeze({}) as InferSpec;

export function panelModes(c: PageCompiler): (PanelMode | undefined)[] {
  const panels = Array.isArray(c.page.panels) ? c.page.panels : [];
  return panels.map((panel) => (isRecord(panel) ? (panel.mode as PanelMode | undefined) : undefined));
}

export function compilePanels(c: PageCompiler, env: Env): PanelItem[] {
  const page = c.page;
  if (page.panels === undefined) return [];
  const where = child(c.pageWhere, page, 'panels');
  if (!Array.isArray(page.panels)) fail(where, 'panels must be a list');
  const panels = page.panels;
  return panels.map((panel, i): PanelItem => {
    const at = child(where, panels, i);
    if (panel === 'cost') return 'cost';
    if (!isRecord(panel)) fail(at, "a panel must be 'cost' or a mapping");
    if (panel.mode !== undefined && !PANEL_MODES.includes(panel.mode as PanelMode)) fail(child(at, panel, 'mode'), "mode must be 'train' or 'infer'");
    if (panel.mode !== undefined && page.generate === undefined) fail(child(at, panel, 'mode'), 'a panel mode needs generate on the page');
    const field = (key: string): Env => ({ ...env, where: child(at, panel, key) });
    const node = (key: string, obj: Record<string, unknown> = panel, base: Where = at): Node => c.node(obj[key], { ...env, where: child(base, obj, key) });
    const texts = (allowed: string[]): Record<string, string> => {
      const out: Record<string, string> = {};
      for (const key of allowed) if (panel[key] !== undefined) out[key] = c.text(panel[key], field(key));
      return out;
    };
    if (panel.cost !== undefined) {
      checkKeys(panel, new Set(['mode', 'cost']), at);
      if (panel.cost !== true) fail(field('cost').where, 'cost must be true');
      return 'cost';
    }
    if (panel.loss !== undefined) {
      checkKeys(panel, new Set(['mode', 'loss']), at);
      const loss = record(panel.loss, child(at, panel, 'loss'), 'loss');
      const lossWhere = child(at, panel, 'loss');
      checkKeys(loss, new Set(['parts', 'hint']), lossWhere);
      const out: { parts?: Node; hint?: string } = {};
      if (loss.parts !== undefined) out.parts = node('parts', loss, lossWhere);
      if (loss.hint !== undefined) out.hint = c.text(loss.hint, { ...env, where: child(lossWhere, loss, 'hint') });
      return { loss: out };
    }
    if (panel.sequence !== undefined) {
      checkKeys(panel, new Set(['mode', 'sequence', 'tokens', 'input', 'label', 'labels', 'sgd', 'random', 'reset', ...TEXT_KEYS]), at);
      if (panel.sequence !== true) fail(field('sequence').where, 'sequence must be true');
      const out: Extract<PanelItem, { sequence: true }> = { sequence: true, ...texts(TEXT_KEYS) };
      if (panel.labels !== undefined) {
        if (!Array.isArray(panel.labels)) fail(field('labels').where, 'labels must be a list');
        out.labels = panel.labels.map((label, k) => c.text(label, { ...env, where: child(field('labels').where, panel.labels as unknown[], k) }));
      }
      if (panel.sgd !== undefined) {
        const sgdWhere = field('sgd').where;
        const sgd = record(panel.sgd, sgdWhere, 'sgd');
        checkKeys(sgd, new Set(['params', 'lr', 'label']), sgdWhere);
        if (typeof sgd.lr !== 'number') fail(child(sgdWhere, sgd, 'lr'), 'lr must be a number');
        const paramsWhere = child(sgdWhere, sgd, 'params');
        const params =
          sgd.params === 'all'
            ? c.g.nodes.filter((n) => n.kind === 'param')
            : Array.isArray(sgd.params)
              ? sgd.params.map((p, k) => c.node(p, { ...env, where: child(paramsWhere, sgd.params as unknown[], k) }))
              : fail(paramsWhere, "params must be 'all' or a list of parameters");
        out.sgd = { params, lr: sgd.lr, ...(sgd.label === undefined ? {} : { label: c.text(sgd.label, { ...env, where: child(sgdWhere, sgd, 'label') }) }) };
      }
      for (const key of ['random', 'reset'] as const) {
        if (panel[key] === undefined) continue;
        if (typeof panel[key] !== 'boolean') fail(field(key).where, `${key} must be true or false`);
        out[key] = panel[key] as boolean;
      }
      return out;
    }
    if (panel.generation !== undefined) {
      checkKeys(panel, new Set(['mode', 'generation', ...TEXT_KEYS]), at);
      if (panel.generation !== true) fail(field('generation').where, 'generation must be true');
      if (page.generate === undefined) fail(field('generation').where, 'a generation panel needs generate on the page');
      return { generation: PENDING_INFER, ...texts(TEXT_KEYS) };
    }
    if (panel.kv_blocks !== undefined) {
      checkKeys(panel, new Set(['mode', 'kv_blocks', ...TEXT_KEYS]), at);
      if (page.generate === undefined) fail(field('kv_blocks').where, 'a kv_blocks panel needs generate on the page');
      const specWhere = field('kv_blocks').where;
      const spec = record(panel.kv_blocks, specWhere, 'kv_blocks');
      checkKeys(spec, new Set(['block_size', 'num_blocks', 'requests']), specWhere);
      const count = (key: string, min: number) => {
        const value = spec[key];
        if (typeof value !== 'number' || !Number.isInteger(value) || value < min) fail(child(specWhere, spec, key), `${key} must be an integer of at least ${min}`);
        return value as number;
      };
      const listed = spec.requests ?? [];
      const requestsWhere = child(specWhere, spec, 'requests');
      if (!Array.isArray(listed)) fail(requestsWhere, 'requests must be a list');
      const others = listed.map((entry, k) => {
        const entryWhere = child(requestsWhere, listed, k);
        const request = record(entry, entryWhere, 'a request');
        checkKeys(request, new Set(['id', 'prompt', 'arrive']), entryWhere);
        if (typeof request.id !== 'string' || request.id === 'A') fail(child(entryWhere, request, 'id'), "id must be text other than 'A', which names the page's own request");
        if (!Array.isArray(request.prompt) || request.prompt.length === 0 || !request.prompt.every((t) => typeof t === 'string')) fail(child(entryWhere, request, 'prompt'), 'prompt must be a list of tokens');
        if (typeof request.arrive !== 'number' || !Number.isInteger(request.arrive) || request.arrive < 1) fail(child(entryWhere, request, 'arrive'), 'arrive must be a decode step, counted from 1');
        return { id: request.id as string, prompt: request.prompt as string[], arrive: request.arrive as number };
      });
      return { kvBlocks: { infer: PENDING_INFER, blockSize: count('block_size', 1), numBlocks: count('num_blocks', 2), others }, ...texts(TEXT_KEYS) };
    }
    if (panel.code !== undefined) {
      checkKeys(panel, new Set(['mode', 'code', ...TEXT_KEYS]), at);
      if (typeof panel.code !== 'string') fail(field('code').where, 'code must be text');
      return { code: panel.code, ...texts(TEXT_KEYS) };
    }
    if (panel.attention !== undefined) {
      checkKeys(panel, new Set(['mode', 'attention', 'ids', ...TEXT_KEYS]), at);
      return { attention: [node('attention')], ids: node('ids'), ...texts(TEXT_KEYS) };
    }
    if (panel.matrices !== undefined) {
      checkKeys(panel, new Set(['mode', 'matrices', 'highlight', 'rows', ...TEXT_KEYS]), at);
      const matrices = panel.matrices;
      if (!Array.isArray(matrices)) fail(field('matrices').where, 'matrices must be a list');
      const out: Extract<PanelItem, { matrices: unknown }> = {
        matrices: matrices.map((entry, k) => {
          const entryAt = child(field('matrices').where, matrices, k);
          const spec = record(entry, entryAt, 'a matrix');
          checkKeys(spec, new Set(['matrix', 'grad', 'caption', 'limit']), entryAt);
          const item: { matrix: Node; grad?: boolean; caption?: string; limit?: number } = { matrix: node('matrix', spec, entryAt) };
          if (typeof spec.grad === 'boolean') item.grad = spec.grad;
          if (spec.caption !== undefined) item.caption = c.text(spec.caption, { ...env, where: child(entryAt, spec, 'caption') });
          if (typeof spec.limit === 'number') item.limit = spec.limit;
          return item;
        }),
        ...texts(TEXT_KEYS),
      };
      if (panel.highlight !== undefined) {
        const highlightWhere = field('highlight').where;
        const highlight = record(panel.highlight, highlightWhere, 'highlight');
        checkKeys(highlight, new Set(['node', 'all']), highlightWhere);
        out.highlight = { node: node('node', highlight, highlightWhere), ...(typeof highlight.all === 'boolean' ? { all: highlight.all } : {}) };
      }
      if (panel.rows !== undefined) out.rows = node('rows').name;
      return out;
    }
    return fail(at, 'a panel needs one of loss, sequence, generation, kv_blocks, code, attention or matrices, or is cost');
  });
}
