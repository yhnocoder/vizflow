import { Node } from '../../autograd';
import { checkFigure, figureNames, figureTag } from '../../ui/figure-spec';
import { DslError } from '../source';
import { numberTex } from '../tex';
import { describe, isDtype, isView, viewShape } from './expressions';
import { Modules } from './instance';
import type { Env, TensorView, Value, ViewIndex } from './types';
import { isNum } from './util';

export interface Macro {
  args: number;
  body: string;
}

const MACRO_DEPTH = 32;

function bracedEnd(text: string, open: number): number {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    if (text[i] === '\\') i++;
    else if (text[i] === '{') depth++;
    else if (text[i] === '}' && --depth === 0) return i;
  }
  return -1;
}

export function parsePreamble(text: string, file: string): Map<string, Macro> {
  const macros = new Map<string, Macro>();
  const pattern = /\\newcommand\s*\{\\([A-Za-z]+)\}\s*(?:\[(\d+)\])?\s*/g;
  for (let found = pattern.exec(text); found !== null; found = pattern.exec(text)) {
    const open = found.index + found[0].length;
    const close = text[open] === '{' ? bracedEnd(text, open) : -1;
    if (close < 0) throw new DslError(`${file}: \\newcommand{\\${found[1]}} needs a braced body`);
    macros.set(found[1], { args: Number(found[2] ?? 0), body: text.slice(open + 1, close) });
    pattern.lastIndex = close + 1;
  }
  return macros;
}

function splitFilter(source: string): { expr: string; filter: string | undefined } {
  let depth = 0;
  let quote = '';
  for (let i = 0; i < source.length; i++) {
    const c = source[i];
    if (quote !== '') {
      if (c === quote) quote = '';
    } else if (c === '"' || c === "'") quote = c;
    else if (c === '[' || c === '(') depth++;
    else if (c === ']' || c === ')') depth--;
    else if (c === '|' && depth === 0) return { expr: source.slice(0, i).trim(), filter: source.slice(i + 1).trim() };
  }
  return { expr: source.trim(), filter: undefined };
}

function pmatrix(lines: string[][]): string {
  return `\\begin{pmatrix}${lines.map((line) => line.map((cell) => ` ${cell} `).join('&')).join('\\\\')}\\end{pmatrix}`;
}

export abstract class Texts extends Modules {
  macros = new Map<string, Macro>();
  readonly tokenLists = new Set<Node>();

  expandTex(tex: string, env: Env): string {
    return this.expandMacros(this.viz(tex, env), env);
  }

  viz(text: string, env: Env): string {
    return text.replace(/\\viz\{([^{}]+)\}/g, (_match, ref: string, offset: number, whole: string) => {
      const value = this.vizValue(ref.trim(), env);
      return /\\[A-Za-z]+$/.test(whole.slice(0, offset)) && /^[A-Za-z]/.test(value) ? ` ${value}` : value;
    });
  }

  vizValue(ref: string, env: Env): string {
    const found = this.reference(ref, env);
    if (found instanceof Node) return found.tex;
    if (isNum(found)) return numberTex(found.toy);
    return this.fail(env, `\\viz{${ref}} names ${describe(found)}; \\viz takes a tensor`);
  }

  expandMacros(text: string, env: Env, depth = 0): string {
    if (this.macros.size === 0) return text;
    if (depth > MACRO_DEPTH) this.fail(env, 'macros expand too deeply');
    let out = '';
    let i = 0;
    let changed = false;
    while (i < text.length) {
      const found = /^\\([A-Za-z]+)/.exec(text.slice(i));
      const macro = found === null ? undefined : this.macros.get(found[1]);
      if (found === null || macro === undefined) {
        out += found === null ? text[i] : found[0];
        i += found === null ? 1 : found[0].length;
        continue;
      }
      let at = i + found[0].length;
      const args: string[] = [];
      for (let k = 0; k < macro.args; k++) {
        while (/\s/.test(text[at] ?? '')) at++;
        if (text[at] !== '{') this.fail(env, `\\${found[1]} needs ${macro.args} braced arguments`);
        const close = bracedEnd(text, at);
        if (close < 0) this.fail(env, `\\${found[1]} has an unclosed argument`);
        args.push(text.slice(at + 1, close));
        at = close + 1;
      }
      out += `{${macro.body.replace(/#(\d)/g, (_m, n: string) => args[Number(n) - 1] ?? '')}}`;
      i = at;
      changed = true;
    }
    return changed ? this.expandMacros(out, env, depth + 1) : out;
  }

  text(value: unknown, env: Env): string {
    if (value === undefined) return '';
    if (typeof value !== 'string') this.fail(env, 'expected text');
    const interpolated = value.replace(/\{\{([^{}]*)\}\}/g, (_m, inner: string) => this.interpolate(inner, env));
    return this.expandMacros(this.viz(interpolated, env), env);
  }

  latex(value: unknown, env: Env): string {
    if (typeof value !== 'string') this.fail(env, 'expected LaTeX text');
    return this.expandTex(value, env);
  }

  interpolate(source: string, env: Env): string {
    const { expr, filter } = splitFilter(source);
    const found = this.reference(expr, env);
    if (filter === undefined) return this.plain(found, source, env);
    const colon = filter.indexOf(':');
    const name = (colon < 0 ? filter : filter.slice(0, colon)).trim();
    const arg = colon < 0 ? undefined : filter.slice(colon + 1).trim();
    const view = this.viewOf(found, source, env);
    if (name === 'matrix') {
      if (arg !== undefined && !/^\d+$/.test(arg)) this.fail(env, `matrix takes a number of digits, got '${arg}' in '{{${source}}}'`);
      return this.matrix(view, arg === undefined ? '' : `:${arg}`);
    }
    if (name === 'join') {
      const separator = arg === undefined ? undefined : /^(["'])(.*)\1$/.exec(arg);
      if (separator === undefined || separator === null) this.fail(env, `join takes a quoted separator, such as join: ", ", in '{{${source}}}'`);
      const shape = viewShape(view);
      if (shape.length !== 1) this.fail(env, `join needs a list or a 1-dimensional tensor in '{{${source}}}', got shape (${shape.join(', ')})`);
      const format = this.tokenLists.has(view.node) ? ':token' : '';
      return Array.from({ length: shape[0] }, (_, i) => this.cell(view, [i], format)).join(separator[2]);
    }
    if (figureNames.includes(name)) {
      if (view.index.length > 0 || view.grad) this.fail(env, `${name} takes a whole tensor in '{{${source}}}'`);
      const args = arg === undefined ? [] : arg.split(':').map((part) => part.trim());
      const problem = checkFigure(name, view.node.shape, args);
      if (problem !== undefined) this.fail(env, `${problem} in '{{${source}}}'`);
      return figureTag(name, view.node.name, args);
    }
    return this.fail(env, `unknown format '${name}' in '{{${source}}}'; formats are matrix, join and ${figureNames.join(', ')}`);
  }

  private plain(found: Value, source: string, env: Env): string {
    if (isNum(found)) return numberTex(found.toy);
    if (typeof found === 'string' || typeof found === 'boolean') return String(found);
    if (isDtype(found)) return found.dtype;
    if (found instanceof Node || isView(found)) {
      const view = this.viewOf(found, source, env);
      if (viewShape(view).length > 0) this.fail(env, `'{{${source}}}' is not a single number; add a format such as | matrix:2 or | join: ", "`);
      return this.cell(view, [], this.tokenLists.has(view.node) ? ':token' : '');
    }
    return this.fail(env, `'{{${source}}}' is ${describe(found)}, which has no value to show`);
  }

  private viewOf(found: Value, source: string, env: Env): TensorView {
    if (found instanceof Node) return { node: found, grad: false, index: [] };
    if (isView(found)) return found;
    return this.fail(env, `'{{${source}}}' is ${describe(found)}, not a tensor`);
  }

  private cell(view: TensorView, free: number[], format: string): string {
    const index: number[] = [];
    let k = 0;
    view.node.shape.forEach((size, d) => {
      const item: ViewIndex = view.index[d] ?? { start: 0, stop: size };
      index.push(typeof item === 'number' ? item : item.start + free[k++]);
    });
    const name = `${view.node.name}${view.grad ? '.grad' : ''}`;
    return index.length === 0 ? `{{${name}${format}}}` : `{{${name}[${index.join(', ')}]${format}}}`;
  }

  private matrix(view: TensorView, format: string): string {
    const shape = viewShape(view);
    if (shape.length === 0) return this.cell(view, [], format);
    const grid = (lead: number[], rows: number, cols: number, two: boolean) =>
      Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) => this.cell(view, two ? [...lead, r, c] : [...lead, c], format)));
    if (shape.length === 1) return pmatrix(grid([], 1, shape[0], false));
    if (shape.length === 2) return pmatrix(grid([], shape[0], shape[1], true));
    const leading = shape.slice(0, -2);
    const [rows, cols] = shape.slice(-2);
    const blocks: string[] = [];
    const visit = (prefix: number[]) => {
      if (prefix.length === leading.length) {
        blocks.push(`${pmatrix(grid(prefix, rows, cols, true))}_{[${prefix.join(',')}]}`);
        return;
      }
      for (let i = 0; i < leading[prefix.length]; i++) visit([...prefix, i]);
    };
    visit([]);
    return `\\left[\\,${blocks.join('\\;\\;')}\\,\\right]`;
  }
}
