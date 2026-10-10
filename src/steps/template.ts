import type { Graph, Node } from '../autograd';
import { formatNumber } from '../tensor';

type Subscript = { kind: 'int'; value: number } | { kind: 'focus' } | { kind: 'node'; node: Node; row: boolean };

interface Template {
  source: string;
  node: Node;
  grad: boolean;
  subscripts: Subscript[] | undefined;
  offset: number;
  format: number | 'token' | undefined;
}

type Part = { kind: 'text'; text: string } | { kind: 'template'; template: Template; math: boolean };

const fail = (source: string, reason: string) => new Error(`cannot parse template '{{${source}}}': ${reason}`);

function lookup(g: Graph, name: string, source: string): Node {
  const node = g.node(name);
  if (node === undefined) throw fail(source, `node '${name}' does not exist`);
  return node;
}

function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '[') depth++;
    else if (text[i] === ']') depth--;
    else if (text[i] === ',' && depth === 0) {
      parts.push(text.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(text.slice(start));
  return parts;
}

function parseSubscript(g: Graph, text: string, source: string): Subscript {
  const item = text.trim();
  if (item === 'b') return { kind: 'focus' };
  if (/^-?\d+$/.test(item)) return { kind: 'int', value: Number(item) };
  const row = /^(.+?)\s*\[\s*b\s*\]$/.exec(item);
  if (row !== null) return { kind: 'node', node: lookup(g, row[1], source), row: true };
  if (item === '' || /[\[\]]/.test(item)) throw fail(source, `bad subscript '${item}'`);
  return { kind: 'node', node: lookup(g, item, source), row: false };
}

function parseTemplate(g: Graph, source: string): Template {
  let rest = source.trim();
  const nameEnd = rest.search(/[[.+\-:]/);
  const name = (nameEnd < 0 ? rest : rest.slice(0, nameEnd)).trim();
  if (name === '') throw fail(source, 'missing node name');
  rest = nameEnd < 0 ? '' : rest.slice(nameEnd);
  const node = lookup(g, name, source);
  let grad = false;
  if (rest.startsWith('.grad')) {
    grad = true;
    rest = rest.slice(5).trimStart();
  } else if (rest.startsWith('.')) {
    throw fail(source, `unknown suffix '${rest}'`);
  }
  let subscripts: Subscript[] | undefined;
  if (rest.startsWith('[')) {
    let depth = 0;
    let close = -1;
    for (let i = 0; i < rest.length; i++) {
      if (rest[i] === '[') depth++;
      else if (rest[i] === ']' && --depth === 0) {
        close = i;
        break;
      }
    }
    if (close < 0) throw fail(source, 'unclosed subscript');
    subscripts = splitTopLevel(rest.slice(1, close)).map((item) => parseSubscript(g, item, source));
    rest = rest.slice(close + 1).trimStart();
  }
  let offset = 0;
  const shift = /^([+-])\s*(\d+)\s*/.exec(rest);
  if (shift !== null) {
    offset = (shift[1] === '-' ? -1 : 1) * Number(shift[2]);
    rest = rest.slice(shift[0].length);
  }
  let format: number | 'token' | undefined;
  const fmt = /^:\s*(\d+|token)\s*$/.exec(rest);
  if (fmt !== null) {
    format = fmt[1] === 'token' ? 'token' : Number(fmt[1]);
    rest = '';
  }
  if (rest !== '') throw fail(source, `unexpected '${rest}'`);
  const template = { source, node, grad, subscripts, offset, format };
  validate(template);
  return template;
}

function validate(t: Template): void {
  if (t.grad && t.node.kind === 'label') throw fail(t.source, `label '${t.node.name}' has no gradient`);
  if (t.format === 'token' && t.node.tokens === undefined) throw fail(t.source, `node '${t.node.name}' has no tokens`);
  const shape = t.node.shape;
  const count = t.subscripts?.length ?? 0;
  if (count === 0 && t.node.value.size !== 1) throw fail(t.source, `node '${t.node.name}' with shape (${shape.join(', ')}) needs a subscript`);
  if (count !== 0 && count !== shape.length && !(count === 1 && shape.length === 2 && (shape[0] === 1 || shape[1] === 1))) {
    throw fail(t.source, `node '${t.node.name}' with shape (${shape.join(', ')}) takes ${shape.length} subscripts, got ${count}`);
  }
}

function resolveIndex(t: Template, focus: number): number[] {
  const shape = t.node.shape;
  if (t.subscripts === undefined) return shape.map(() => 0);
  const index = t.subscripts.map((s) => {
    if (s.kind === 'int') return s.value;
    if (s.kind === 'focus') return focus;
    const position = s.node.shape.map(() => 0);
    if (s.row) position[0] = focus;
    return Math.round(s.node.value.get(position));
  });
  if (index.length === 1 && shape.length === 2) return shape[0] === 1 ? [0, index[0]] : [index[0], 0];
  return index;
}

function evaluate(t: Template, focus: number, math: boolean): string {
  const tensor = t.grad ? t.node.grad : t.node.value;
  if (tensor === null) throw fail(t.source, `label '${t.node.name}' has no gradient`);
  let value: number;
  try {
    value = tensor.get(resolveIndex(t, focus));
  } catch (error) {
    throw fail(t.source, (error as Error).message);
  }
  value += t.offset;
  if (t.format === 'token') {
    const token = t.node.tokens?.[Math.round(value)];
    if (token === undefined) throw fail(t.source, `node '${t.node.name}' has no token ${Math.round(value)}`);
    return token;
  }
  if (typeof t.format === 'number') return formatNumber(value, { digits: t.format, tex: math });
  if (t.node.kind === 'label' && !t.grad) {
    const text = String(Math.round(value));
    return math ? text : text.replace('-', '−');
  }
  return formatNumber(value, { tex: math });
}

export function compileText(g: Graph, text: string): (focus: number) => string {
  const parts: Part[] = [];
  let plain = '';
  let math = false;
  let i = 0;
  while (i < text.length) {
    if (text.startsWith('\\$', i)) {
      plain += '\\$';
      i += 2;
    } else if (text[i] === '$') {
      const width = text.startsWith('$$', i) ? 2 : 1;
      plain += text.slice(i, i + width);
      i += width;
      math = !math;
    } else if (text.startsWith('{{', i)) {
      if (text[i + 2] === '{') throw new Error(`template at '${text.slice(i, i + 12)}' starts with '{{{'; leave a space before the template`);
      const close = text.indexOf('}}', i + 2);
      if (close < 0) throw new Error(`template at '${text.slice(i, i + 12)}' is not closed`);
      if (text[close + 2] === '}') throw new Error(`template '${text.slice(i, close + 3)}' ends with '}}}'; leave a space after the template`);
      if (plain) parts.push({ kind: 'text', text: plain });
      plain = '';
      parts.push({ kind: 'template', template: parseTemplate(g, text.slice(i + 2, close)), math });
      i = close + 2;
    } else {
      plain += text[i];
      i++;
    }
  }
  if (plain) parts.push({ kind: 'text', text: plain });
  return (focus) => parts.map((p) => (p.kind === 'text' ? p.text : evaluate(p.template, focus, p.math))).join('');
}
