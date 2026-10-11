import type { SectionBlock, SectionItem, SectionRow } from '../../ui';
import { type Where, child, fail } from '../source';
import type { Compiler } from './text';
import type { Env } from './types';
import { isRecord } from './util';

const ROPE_POSITIONS = 24;

export type SectionSources = Map<object, { file: string; rows: unknown[] }>;

export function compileSections(c: Compiler, env: Env, sectionSources: SectionSources): SectionItem[] {
  const page = c.page;
  if (page.sections === undefined) return [];
  const where = child(c.pageWhere, page, 'sections');
  if (!Array.isArray(page.sections)) fail(where, 'sections must be a list');
  const sections = page.sections;
  const nodesAt = (value: unknown, at: Where) =>
    Array.isArray(value) ? value.flatMap((v, k) => c.nodeList(v, { ...env, where: child(at, value, k) })) : c.nodeList(value, { ...env, where: at });
  const textAt = (value: unknown, at: Where) => {
    if (typeof value !== 'string') fail(at, 'expected text');
    return c.text(value, { ...env, where: at });
  };
  const numberAt = (value: unknown, at: Where) => {
    if (typeof value === 'number') return value;
    const found = typeof value === 'string' ? c.expression(value, { ...env, where: at }, undefined) : undefined;
    if (typeof found !== 'object' || found === null || !('toy' in found)) return fail(at, 'expected a number or a dimension expression');
    return found.toy;
  };
  const block = (value: unknown, at: Where): SectionBlock => {
    if (value === 'rope_dial' || value === 'rope_curve') return value;
    if (!isRecord(value)) fail(at, "a block must be 'rope_dial', 'rope_curve' or a mapping");
    if (Object.keys(value).length !== 1) fail(at, 'a block must have exactly one key');
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
          if (!isRecord(b)) fail(buttonAt, 'a button must be a mapping with label and go');
          return { label: textAt(b.label, child(buttonAt, b, 'label')), go: c.node(b.go, { ...env, where: child(buttonAt, b, 'go') }) };
        }),
      };
    }
    if (key === 'rope_compare') {
      const spec = value.rope_compare;
      if (!isRecord(spec)) fail(keyAt, 'rope_compare must be a mapping');
      const specAt = (k: string) => child(keyAt, spec, k);
      return {
        rope_compare: {
          scores: nodesAt(spec.scores, specAt('scores')),
          weights: nodesAt(spec.weights, specAt('weights')),
          ids: c.node(spec.ids, { ...env, where: specAt('ids') }),
        },
      };
    }
    return fail(keyAt, `unknown block '${key}'`);
  };
  return sections.map((section, i): SectionItem => {
    const at = child(where, sections, i);
    if (!isRecord(section)) fail(at, 'a section must be a mapping');
    const field = (key: string) => child(at, section, key);
    const out: SectionItem = { rows: [] };
    const source = sectionSources.get(section);
    if (source !== undefined && section.rows !== undefined) fail(field('rows'), 'a section takes either src or rows');
    if (section.title !== undefined) out.title = textAt(section.title, field('title'));
    if (section.rope !== undefined) {
      const rope = section.rope;
      if (!isRecord(rope)) fail(field('rope'), 'rope must be a mapping');
      const ropeAt = (key: string) => child(field('rope'), rope, key);
      const rows = rope.rows ?? [-1, 1];
      if (!Array.isArray(rows) || rows.length !== 2) fail(ropeAt('rows'), 'rows must be [q row, k row]');
      out.rope = {
        q: nodesAt(rope.q, ropeAt('q')),
        k: nodesAt(rope.k, ropeAt('k')),
        rows: [numberAt(rows[0], ropeAt('rows')), numberAt(rows[1], ropeAt('rows'))],
        head: rope.head === undefined ? 1 : numberAt(rope.head, ropeAt('head')),
        theta: numberAt(rope.theta, ropeAt('theta')),
        dh: numberAt(rope.dh, ropeAt('dh')),
        positions: rope.positions === undefined ? ROPE_POSITIONS : numberAt(rope.positions, ropeAt('positions')),
      };
      if (!Number.isInteger(out.rope.dh) || out.rope.dh < 2 || out.rope.dh % 2 !== 0) fail(ropeAt('dh'), 'dh must be an even integer');
      if (!Number.isInteger(out.rope.positions) || out.rope.positions < 2) fail(ropeAt('positions'), 'positions must be an integer of at least 2');
    }
    const rows = source === undefined ? section.rows : source.rows;
    const rowsField = source === undefined ? field('rows') : { file: source.file, path: 'rows', obj: undefined };
    if (!Array.isArray(rows)) fail(rowsField, 'a section needs src or rows');
    out.rows = rows.map((row, r): SectionRow => {
      const rowAt = child(rowsField, rows, r);
      if (Array.isArray(row)) return { cells: [row.map((b, k) => block(b, child(rowAt, row, k)))] };
      if (!isRecord(row) || !Array.isArray(row.cells)) fail(rowAt, 'a row must be a list of blocks or a mapping with cells');
      const cells = row.cells;
      const columns = row.columns;
      if (columns !== undefined && (!Array.isArray(columns) || columns.length !== cells.length)) fail(child(rowAt, row, 'columns'), 'columns must list one width per cell');
      return {
        columns: columns as (number | string)[] | undefined,
        cells: cells.map((cell, k) => {
          const cellAt = child(child(rowAt, row, 'cells'), cells, k);
          if (!Array.isArray(cell)) fail(cellAt, 'a cell must be a list of blocks');
          return cell.map((b, n) => block(b, child(cellAt, cell, n)));
        }),
      };
    });
    return out;
  });
}
