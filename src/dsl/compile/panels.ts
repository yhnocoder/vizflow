import type { Node } from '../../autograd';
import type { PanelItem } from '../../ui';
import { type Where, child, fail } from '../source';
import type { PageCompiler } from './page';
import type { Env } from './types';
import { argSpec } from './instance';
import { checkKeys, isRecord, record } from './util';

const TEXT_KEYS = ['title', 'note'];

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
    const field = (key: string): Env => ({ ...env, where: child(at, panel, key) });
    const node = (key: string, obj: Record<string, unknown> = panel, base: Where = at): Node => c.node(obj[key], { ...env, where: child(base, obj, key) });
    const texts = (allowed: string[]): Record<string, string> => {
      const out: Record<string, string> = {};
      for (const key of allowed) if (panel[key] !== undefined) out[key] = c.text(panel[key], field(key));
      return out;
    };
    if (panel.loss !== undefined) {
      checkKeys(panel, new Set(['loss']), at);
      const loss = record(panel.loss, child(at, panel, 'loss'), 'loss');
      const lossWhere = child(at, panel, 'loss');
      checkKeys(loss, new Set(['parts', 'hint']), lossWhere);
      const out: { parts?: Node; hint?: string } = {};
      if (loss.parts !== undefined && c.reference(loss.parts, { ...env, where: child(lossWhere, loss, 'parts') }) !== null) out.parts = node('parts', loss, lossWhere);
      if (loss.hint !== undefined) out.hint = c.text(loss.hint, { ...env, where: child(lossWhere, loss, 'hint') });
      return { loss: out };
    }
    if (panel.sequence !== undefined) {
      checkKeys(panel, new Set(['sequence', 'tokens', 'input', 'label', 'labels', 'sgd', 'random', 'reset', ...TEXT_KEYS]), at);
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
    if (panel.code !== undefined) {
      checkKeys(panel, new Set(['code', ...TEXT_KEYS]), at);
      if (typeof panel.code !== 'string') fail(field('code').where, 'code must be text');
      return { code: panel.code, ...texts(TEXT_KEYS) };
    }
    if (panel.attention !== undefined) {
      checkKeys(panel, new Set(['attention', 'ids', ...TEXT_KEYS]), at);
      return { attention: [node('attention')], ids: node('ids'), ...texts(TEXT_KEYS) };
    }
    if (panel.matrices !== undefined) {
      checkKeys(panel, new Set(['matrices', 'highlight', 'rows', ...TEXT_KEYS]), at);
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
    if (panel.implementations !== undefined) {
      checkKeys(panel, new Set(['implementations', ...TEXT_KEYS]), at);
      const listWhere = field('implementations').where;
      const list = panel.implementations;
      if (!Array.isArray(list) || list.length === 0) fail(listWhere, 'implementations lists the config args a reader can switch');
      const root = c.rootCall.instance.def;
      const declared = record(root.decl.args, child(root.where, root.decl, 'args'), 'args');
      return {
        implementations: list.map((name, k) => {
          const itemAt = child(listWhere, list, k);
          if (typeof name !== 'string' || !(name in declared)) fail(itemAt, `'${String(name)}' is not an arg of ${root.name}`);
          const { type, options } = argSpec(declared[name], child(child(root.where, root.decl, 'args'), declared, name));
          const value = c.config.get(name);
          if (value === undefined) fail(itemAt, `config does not set '${name}'`);
          if (options !== undefined) return { name, value: value as string, options };
          if (type === 'bool') return { name, value: value as boolean, options: [false, true] };
          return fail(itemAt, `'${name}' needs options or type bool to be switched`);
        }),
        ...texts(TEXT_KEYS),
      };
    }
    return fail(at, 'a panel needs one of loss, sequence, code, attention, matrices or implementations, or is cost');
  });
}
