import { Node } from '../../autograd';
import { trailingSubscript } from '../../explain';
import type { BackwardOverride } from '../../steps';
import { checkFigure, figureNames, figureTag } from '../../ui/figure-spec';
import { parseExpr } from '../expr';
import { child } from '../source';
import { OP_TEX, defaultTex, numberTex, opTex } from '../tex';
import { ExpressionCompiler } from './expressions';
import { type Env, NeedDim, type Num, type OpValue } from './types';
import { checkKeys, isNum, record } from './util';

function matrixTemplate(name: string, shape: readonly number[], format: string): string {
  if (shape.length === 0) return `{{${name}${format}}}`;
  const pmatrix = (lines: string[][]) => `\\begin{pmatrix}${lines.map((line) => line.map((c) => ` ${c} `).join('&')).join('\\\\')}\\end{pmatrix}`;
  const grid = (rows: number, cols: number, cell: (r: number, c: number) => string) =>
    [...Array(rows).keys()].map((r) => [...Array(cols).keys()].map((c) => `{{${name}[${cell(r, c)}]${format}}}`));
  if (shape.length === 1) return pmatrix(grid(1, shape[0], (_r, c) => `${c}`));
  if (shape.length === 2) {
    const [rows, cols] = shape;
    return pmatrix(grid(rows, cols, (r, c) => (cols === 1 ? `${r}` : rows === 1 ? `${c}` : `${r}, ${c}`)));
  }
  const middle = shape.slice(1, -1);
  const heads = middle.reduce((a, b) => a * b, 1);
  const unravel = (h: number) => middle.map((_, d) => Math.floor(h / middle.slice(d + 1).reduce((a, b) => a * b, 1)) % middle[d]);
  const blocks = [...Array(heads).keys()].map((h) => {
    const index = unravel(h);
    const block = pmatrix(grid(shape[0], shape[shape.length - 1], (r, c) => [r, ...index, c].join(', ')));
    return heads === 1 ? block : `${block}_{h=${index.map((i) => i + 1).join(',')}}`;
  });
  return heads === 1 ? blocks[0] : `\\left[\\,${blocks.join('\\;\\;')}\\,\\right]`;
}

export class Compiler extends ExpressionCompiler {
  loopTex(base: string, indices: number[]): string {
    const tex = defaultTex(base);
    return indices.length === 0 ? tex : `${tex}^{(${indices.join(',')})}`;
  }

  ofText(text: string, env: Env): string {
    const accented = text.replace(/\\(hat|bar|tilde|widehat|widetilde|overline|vec|dot)\{\\of\{([^{}]+)\}\}/g, (_match, accent: string, ref: string) => {
      const value = this.ofValue(ref.trim(), env);
      const split = trailingSubscript(value);
      return split === undefined ? `\\${accent}{${value}}` : `\\${accent}{${split.base}}_{${split.inner}}`;
    });
    return accented.replace(/\\of\{([^{}]+)\}/g, (_match, ref: string, offset: number, whole: string) => {
      const value = this.ofValue(ref.trim(), env);
      return /\\[A-Za-z]+$/.test(whole.slice(0, offset)) && /^[A-Za-z]/.test(value) ? ` ${value}` : value;
    });
  }

  ofValue(ref: string, env: Env): string {
    if (ref.endsWith('.def')) {
      const base = ref.slice(0, -4);
      const param = env.scope.params.get(base);
      if (param?.type !== 'op') this.fail(env, `\\of{${ref}}: '${base}' is not an op parameter`);
      const op = (param.value as OpValue).op;
      const def = OP_TEX[op];
      if (def === undefined) this.fail(env, `\\of{${ref}}: op '${op}' has no registered definition`);
      return def.def;
    }
    if (!env.loops.has(ref)) {
      const param = env.scope.params.get(ref);
      if (param !== undefined) {
        if (param.type === 'op') return opTex((param.value as OpValue).op);
        if (param.type === 'float') return numberTex((param.value as Num).toy);
        if (param.type === 'tex') return param.value as string;
        if (param.type === 'text') return this.ofText(param.value as string, env);
        return String((param.value as Num).toy);
      }
      const inputTex = env.scope.inputTex.get(ref);
      if (inputTex !== undefined) return inputTex;
    }
    const found = this.reference(ref, env);
    if (found instanceof Node) return found.tex;
    if (isNum(found)) return String(found.toy);
    if (typeof found === 'string') return found;
    return this.fail(env, `\\of{${ref}} does not name a node, parameter or loop variable`);
  }

  text(text: unknown, env: Env): string {
    if (text === undefined) return '';
    if (typeof text !== 'string') this.fail(env, 'expected text');
    const substituted = this.ofText(text, env);
    return substituted.replace(/\{\{([^{}]*?)\}\}/g, (_m, inner: string) => this.template(inner, env));
  }

  texts(value: unknown, env: Env, allowed: ReadonlySet<string>): BackwardOverride {
    const spec = record(value, env.where, 'a step text');
    checkKeys(spec, allowed, env.where);
    const out: BackwardOverride = {};
    for (const key of ['title', 'text', 'label'] as const) {
      if (spec[key] !== undefined) out[key] = this.text(spec[key], { ...env, where: child(env.where, spec, key) });
    }
    return out;
  }

  template(source: string, env: Env): string {
    const pipe = source.lastIndexOf('|');
    if (pipe >= 0) {
      const filter = source.slice(pipe + 1).trim();
      const match = /^matrix(?::(\d+))?$/.exec(filter);
      let target = source.slice(0, pipe).trim();
      if (match === null) {
        const [name, ...args] = filter.split(':').map((part) => part.trim());
        if (!figureNames.includes(name)) this.fail(env, `unknown template filter '${filter}' in '{{${source}}}'`);
        const node = this.node(target, env);
        const problem = checkFigure(name, node.shape, args);
        if (problem !== undefined) this.fail(env, `${problem} in '{{${source}}}'`);
        return figureTag(name, node.name, args);
      }
      let grad = '';
      if (target.endsWith('.grad')) {
        grad = '.grad';
        target = target.slice(0, -5);
      }
      const node = this.node(target, env);
      return matrixTemplate(`${node.name}${grad}`, node.shape, match[1] === undefined ? '' : `:${match[1]}`);
    }
    return `{{${this.rewriteNames(source, env)}}}`;
  }

  rewriteNames(source: string, env: Env): string {
    let out = '';
    let i = 0;
    while (i < source.length) {
      const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(i));
      const prev = source[i - 1];
      if (id === null || (prev !== undefined && /[A-Za-z0-9_.:]/.test(prev))) {
        out += source[i];
        i++;
        continue;
      }
      const ends: number[] = [];
      let j = i + id[0].length;
      ends.push(j);
      for (;;) {
        if (source[j] === '.') {
          const member = /^\.[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(j));
          if (member === null) break;
          j += member[0].length;
        } else if (source[j] === '[') {
          let depth = 0;
          let k = j;
          for (; k < source.length; k++) {
            if (source[k] === '[') depth++;
            else if (source[k] === ']' && --depth === 0) break;
          }
          if (k >= source.length) break;
          j = k + 1;
        } else {
          break;
        }
        ends.push(j);
      }
      let replaced = false;
      for (const end of ends.reverse()) {
        const candidate = source.slice(i, end);
        if (candidate === 'b') break;
        const found = this.tryNode(candidate, env);
        if (found !== undefined) {
          out += found.name;
          i = end;
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        out += id[0];
        i += id[0].length;
      }
    }
    return out;
  }

  tryNode(text: string, env: Env): Node | undefined {
    try {
      const ast = parseExpr(text);
      const found = this.resolve(ast, env, undefined, 0);
      return found instanceof Node ? found : undefined;
    } catch (error) {
      if (error instanceof NeedDim) throw error;
      return undefined;
    }
  }
}
