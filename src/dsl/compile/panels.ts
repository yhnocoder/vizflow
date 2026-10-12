import type { PanelItem } from '../../ui';
import { type Where, child, fail } from '../source';
import type { Compiler } from './text';
import type { Env } from './types';
import { isRecord, list } from './util';

export function compilePanels(c: Compiler, env: Env): PanelItem[] {
  const page = c.page;
  if (page.panels === undefined) return [];
  const where = child(c.pageWhere, page, 'panels');
  if (!Array.isArray(page.panels)) fail(where, 'panels must be a list');
  const nodeAt = (value: unknown, at: Where) => c.node(value, { ...env, where: at });
  const textAt = (value: unknown, at: Where) => c.text(value, { ...env, where: at });
  const nodesAt = (value: unknown, at: Where) =>
    Array.isArray(value) ? value.flatMap((v, k) => c.nodeList(v, { ...env, where: child(at, value, k) })) : c.nodeList(value, { ...env, where: at });
  return page.panels.map((panel, i): PanelItem => {
    const at = child(where, page.panels as unknown[], i);
    if (panel === 'loss' || panel === 'cost') return panel;
    if (!isRecord(panel)) fail(at, "a panel must be 'loss', 'cost' or a mapping");
    const out: Record<string, unknown> = { ...panel };
    const field = (key: string) => child(at, panel, key);
    for (const key of ['title', 'note', 'caption']) if (typeof panel[key] === 'string') out[key] = textAt(panel[key], field(key));
    for (const key of ['tokens', 'matrix', 'sliders', 'ids']) {
      if (panel[key] === undefined) continue;
      out[key] = nodeAt(panel[key], field(key));
    }
    if (panel.attention !== undefined) out.attention = nodesAt(panel.attention, field('attention'));
    if (panel.extra_keys !== undefined) {
      const extra = panel.extra_keys;
      if (!Array.isArray(extra) || extra.some((label) => typeof label !== 'string')) fail(field('extra_keys'), 'extra_keys must be a list of strings');
      out.extra_keys = extra.map((label, k) => textAt(label, child(field('extra_keys'), extra, k)));
    }
    if (panel.highlight !== undefined) {
      const highlight = panel.highlight;
      out.highlight = isRecord(highlight) ? { ...highlight, node: nodeAt(highlight.node, child(field('highlight'), highlight, 'node')) } : nodeAt(highlight, field('highlight'));
    }
    if (panel.matrices !== undefined) {
      const matrices = panel.matrices;
      if (!Array.isArray(matrices)) fail(field('matrices'), 'matrices must be a list');
      out.matrices = matrices.map((entry, k) => {
        const entryAt = child(field('matrices'), matrices, k);
        if (!isRecord(entry)) fail(entryAt, 'a matrices entry must be a mapping');
        const converted: Record<string, unknown> = { ...entry, matrix: nodeAt(entry.matrix, child(entryAt, entry, 'matrix')) };
        if (typeof entry.caption === 'string') converted.caption = textAt(entry.caption, child(entryAt, entry, 'caption'));
        return converted;
      });
    }
    if (Array.isArray(panel.columns)) {
      const columns = panel.columns;
      out.columns = columns.map((column, k) => {
        const columnAt = child(field('columns'), columns, k);
        if (isRecord(column)) {
          const converted: Record<string, unknown> = { ...column, node: nodeAt(column.node, child(columnAt, column, 'node')) };
          if (typeof column.title === 'string') converted.title = textAt(column.title, child(columnAt, column, 'title'));
          return converted;
        }
        return nodeAt(column, columnAt);
      });
    }
    if (panel.sequence !== undefined) {
      if (panel.sequence !== true) fail(field('sequence'), 'sequence must be true');
      Object.assign(out, trainable(c, panel, ['sgd'], env, at));
    }
    if (isRecord(panel.loss)) out.loss = lossPanel(c, panel.loss, { ...env, where: field('loss') });
    if (isRecord(panel.cost)) out.cost = costPanel(c, panel.cost, { ...env, where: field('cost') });
    return out as PanelItem;
  });
}

function lossPanel(c: Compiler, loss: Record<string, unknown>, env: Env): Record<string, unknown> {
  const out: Record<string, unknown> = { ...loss, ...trainable(c, loss, ['sgd', 'random'], env, env.where) };
  const at = (key: string, obj: object = loss) => ({ ...env, where: child(env.where, obj, key) });
  if (typeof loss.text === 'string') out.text = c.text(loss.text, at('text'));
  if (loss.parts !== undefined) out.parts = c.node(loss.parts, at('parts'));
  return out;
}

function trainable(c: Compiler, spec: Record<string, unknown>, keys: string[], env: Env, where: Where): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    const value = spec[key];
    if (value === undefined) continue;
    const keyWhere = child(where, spec, key);
    if (!isRecord(value)) fail(keyWhere, `${key} must be a mapping`);
    const nodesKey = key === 'sgd' ? 'params' : 'nodes';
    const selected = value[nodesKey];
    const nodesWhere = child(keyWhere, value, nodesKey);
    const nodes = selected === 'all'
      ? c.g.nodes.filter((n) => n.kind === 'param')
      : list(selected as unknown).flatMap((s) => c.nodeList(s, { ...env, where: nodesWhere }));
    out[key] = { ...value, [nodesKey]: nodes };
  }
  return out;
}

function costPanel(c: Compiler, cost: Record<string, unknown>, env: Env): Record<string, unknown> {
  const out: Record<string, unknown> = { ...cost };
  if (typeof cost.title === 'string') out.title = c.text(cost.title, { ...env, where: child(env.where, cost, 'title') });
  if (cost.dims !== undefined && cost.dims !== 'toy' && cost.dims !== 'real') fail(child(env.where, cost, 'dims'), "dims must be 'toy' or 'real'");
  return out;
}
