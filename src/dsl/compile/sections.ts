import type { SectionBlock, SectionItem, SectionRow } from '../../ui';
import { type Where, child, fail } from '../source';
import type { PageCompiler } from './page';
import type { Env } from './types';
import { checkKeys, isRecord, record } from './util';

const ROPE_POSITIONS = 24;

export type SectionSources = Map<object, { file: string; rows: unknown[] }>;

export function compileSections(c: PageCompiler, env: Env, sectionSources: SectionSources): SectionItem[] {
  const page = c.page;
  if (page.sections === undefined) return [];
  const where = child(c.pageWhere, page, 'sections');
  if (!Array.isArray(page.sections)) fail(where, 'sections must be a list');
  const sections = page.sections;
  const textAt = (value: unknown, at: Where) => c.text(value, { ...env, where: at });
  const nodeAt = (value: unknown, at: Where) => c.node(value, { ...env, where: at });
  const intAt = (value: unknown, at: Where): number => {
    const found = c.number(value, { ...env, where: at });
    if (!Number.isInteger(found.toy)) fail(at, 'expected an integer');
    return found.toy;
  };
  const block = (value: unknown, at: Where): SectionBlock => {
    if (value === 'rope_dial' || value === 'rope_curve') return value;
    if (!isRecord(value) || Object.keys(value).length !== 1) fail(at, "a block is 'rope_dial', 'rope_curve' or a mapping with one key");
    const key = Object.keys(value)[0];
    const keyAt = child(at, value, key);
    if (key === 'text') return { text: textAt(value.text, keyAt) };
    if (key === 'lead') return { lead: textAt(value.lead, keyAt) };
    if (key === 'buttons') {
      const buttons = value.buttons;
      if (!Array.isArray(buttons)) fail(keyAt, 'buttons must be a list');
      return {
        buttons: buttons.map((b, k) => {
          const buttonAt = child(keyAt, buttons, k);
          const spec = record(b, buttonAt, 'a button');
          checkKeys(spec, new Set(['label', 'go']), buttonAt);
          return { label: textAt(spec.label, child(buttonAt, spec, 'label')), go: nodeAt(spec.go, child(buttonAt, spec, 'go')) };
        }),
      };
    }
    if (key === 'rope_compare') {
      const spec = record(value.rope_compare, keyAt, 'rope_compare');
      checkKeys(spec, new Set(['ids']), keyAt);
      return { rope_compare: { ids: nodeAt(spec.ids, child(keyAt, spec, 'ids')) } };
    }
    return fail(keyAt, `unknown block '${key}'`);
  };
  return sections.map((section, i): SectionItem => {
    const at = child(where, sections, i);
    const spec = record(section, at, 'a section');
    checkKeys(spec, new Set(['title', 'src', 'rope']), at);
    const field = (key: string) => child(at, spec, key);
    const out: SectionItem = { rows: [] };
    if (spec.title !== undefined) out.title = textAt(spec.title, field('title'));
    if (spec.rope !== undefined) {
      const rope = record(spec.rope, field('rope'), 'rope');
      const ropeAt = (key: string) => child(field('rope'), rope, key);
      checkKeys(rope, new Set(['q', 'k', 'rows', 'head', 'theta', 'd_head', 'positions']), field('rope'));
      const rows = rope.rows ?? [-1, 0];
      if (!Array.isArray(rows) || rows.length !== 2) fail(ropeAt('rows'), 'rows must be [q row, k row]');
      const row = (k: number) => {
        const value = intAt(rows[k], child(ropeAt('rows'), rows, k));
        return value >= 0 ? value + 1 : value;
      };
      const dh = intAt(rope.d_head, ropeAt('d_head'));
      if (dh < 2 || dh % 2 !== 0) fail(ropeAt('d_head'), 'd_head must be an even integer');
      const positions = rope.positions === undefined ? ROPE_POSITIONS : intAt(rope.positions, ropeAt('positions'));
      if (positions < 2) fail(ropeAt('positions'), 'positions must be at least 2');
      out.rope = {
        q: [nodeAt(rope.q, ropeAt('q'))],
        k: [nodeAt(rope.k, ropeAt('k'))],
        rows: [row(0), row(1)],
        head: (rope.head === undefined ? 0 : intAt(rope.head, ropeAt('head'))) + 1,
        theta: c.number(rope.theta, { ...env, where: ropeAt('theta') }).toy,
        dh,
        positions,
      };
    }
    const source = sectionSources.get(section as object);
    if (source === undefined) fail(field('src'), 'a section needs src, a Markdown file');
    const rows = source.rows;
    const rowsField: Where = { file: source.file, path: 'rows', obj: undefined };
    out.rows = rows.map((r, n): SectionRow => {
      const rowAt = child(rowsField, rows, n);
      if (Array.isArray(r)) return { cells: [r.map((b, k) => block(b, child(rowAt, r, k)))] };
      if (!isRecord(r) || !Array.isArray(r.cells)) fail(rowAt, 'a row must be a list of blocks or a mapping with cells');
      const cells = r.cells;
      const columns = r.columns;
      if (columns !== undefined && (!Array.isArray(columns) || columns.length !== cells.length)) fail(child(rowAt, r, 'columns'), 'columns must list one width per cell');
      return {
        columns: columns as (number | string)[] | undefined,
        cells: cells.map((cell, k) => {
          const cellAt = child(child(rowAt, r, 'cells'), cells, k);
          if (!Array.isArray(cell)) fail(cellAt, 'a cell must be a list of blocks');
          return cell.map((b, m) => block(b, child(cellAt, cell, m)));
        }),
      };
    });
    return out;
  });
}
