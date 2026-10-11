import { render } from './render';

type Segment = { kind: 'html'; html: string } | { kind: 'tex'; tex: string; display: boolean };

function findClosing(source: string, from: number, delimiter: string): number {
  for (let index = from; index < source.length; index++) {
    if (source[index] === '\\') {
      index++;
    } else if (source.startsWith(delimiter, index)) {
      return index;
    }
  }
  return -1;
}

function splitTex(source: string): Segment[] {
  const segments: Segment[] = [];
  let html = '';
  let index = 0;
  while (index < source.length) {
    if (source.startsWith('\\$', index)) {
      html += '$';
      index += 2;
      continue;
    }
    if (source[index] !== '$') {
      html += source[index];
      index++;
      continue;
    }
    const delimiter = source.startsWith('$$', index) ? '$$' : '$';
    const start = index + delimiter.length;
    const end = findClosing(source, start, delimiter);
    if (end < 0) {
      html += delimiter;
      index = start;
      continue;
    }
    if (html) segments.push({ kind: 'html', html });
    html = '';
    segments.push({ kind: 'tex', tex: source.slice(start, end), display: delimiter === '$$' });
    index = end + delimiter.length;
  }
  if (html) segments.push({ kind: 'html', html });
  return segments;
}

const blockTag = /^<(?:p|div|ul|ol|pre|table|h[1-6]|blockquote|vf-figure)[\s>]/i;
const preTag = /^<pre[\s>]/i;

function dedent(source: string): string {
  const lines = source.split('\n');
  while (lines.length > 0 && lines[0].trim() === '') lines.shift();
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') lines.pop();
  const indents = lines.filter((line) => line.trim() !== '').map((line) => /^[ \t]*/.exec(line)![0].length);
  const indent = Math.min(...indents);
  return lines.map((line) => line.slice(indent)).join('\n');
}

function splitBlocks(segments: Segment[]): Segment[][] {
  const blocks: Segment[][] = [[]];
  for (const segment of segments) {
    if (segment.kind === 'tex') {
      blocks[blocks.length - 1].push(segment);
      continue;
    }
    segment.html.split(/\n[ \t]*\n\s*/).forEach((html, index) => {
      if (index > 0) blocks.push([]);
      if (html !== '') blocks[blocks.length - 1].push({ kind: 'html', html });
    });
  }
  return blocks.map(trimBlock).filter((block) => block.length > 0);
}

function trimBlock(block: Segment[]): Segment[] {
  const trimmed = block.map((segment, index) => {
    if (segment.kind === 'tex') return segment;
    let html = segment.html;
    if (index === 0) html = html.trimStart();
    if (index === block.length - 1) html = html.trimEnd();
    return { kind: 'html' as const, html };
  });
  return trimmed.filter((segment) => segment.kind === 'tex' || segment.html !== '');
}

const cjk = /[\u1100-\u11ff\u3000-\u303f\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef\u{20000}-\u{2ffff}]/u;
const cjkPunctuation = /[\u3000-\u303f\uff00-\uffef]/u;

function lineBreak(before: string, after: string): string {
  if (cjkPunctuation.test(before) || cjkPunctuation.test(after)) return '';
  return cjk.test(before) && cjk.test(after) ? '' : ' ';
}

function joinLines(block: Segment[]): Segment[] {
  return block.map((segment) => {
    if (segment.kind === 'tex') return segment;
    const html = segment.html.replace(/[ \t]*\n[ \t]*/g, (match, offset: number, whole: string) => {
      const before = [...whole.slice(0, offset)].at(-1) ?? '';
      const after = [...whole.slice(offset + match.length, offset + match.length + 2)][0] ?? '';
      return lineBreak(before, after);
    });
    return { kind: 'html', html };
  });
}

const heading = /^(#{1,6})[ \t]+/;

function inlineMarkdown(block: Segment[]): Segment[] {
  return block.map((segment) => {
    if (segment.kind === 'tex') return segment;
    const html = segment.html.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
    return { kind: 'html', html };
  });
}

function paragraphs(source: string, markdown: boolean): Segment[] {
  return splitBlocks(splitTex(dedent(source))).flatMap((block) => {
    const first = block[0];
    if (block.length === 1 && first.kind === 'tex' && first.display) return block;
    if (first.kind === 'html' && preTag.test(first.html)) return block;
    const joined = markdown ? inlineMarkdown(joinLines(block)) : joinLines(block);
    const head = joined[0];
    if (head.kind === 'html' && blockTag.test(head.html)) return joined;
    const level = markdown && head.kind === 'html' ? heading.exec(head.html) : null;
    if (level !== null && head.kind === 'html') {
      const tag = `h${level[1].length}`;
      const rest: Segment[] = [{ kind: 'html', html: head.html.slice(level[0].length) }, ...joined.slice(1)];
      return [{ kind: 'html', html: `<${tag}>` }, ...rest, { kind: 'html', html: `</${tag}>` }];
    }
    return [{ kind: 'html', html: '<p>' }, ...joined, { kind: 'html', html: '</p>' }];
  });
}

function topLevelParts(tex: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < tex.length; index++) {
    const char = tex[index];
    if (char === '\\') {
      if (depth === 0 && /^\\qquad(?![a-zA-Z])/.test(tex.slice(index))) {
        parts.push(tex.slice(start, index));
        start = index + '\\qquad'.length;
        index = start - 1;
        continue;
      }
      const word = /^\\([a-zA-Z]+)/.exec(tex.slice(index))?.[1];
      if (word === 'begin' || word === 'left') depth++;
      else if (word === 'end' || word === 'right') depth--;
      index++;
    } else if (char === '{') depth++;
    else if (char === '}') depth--;
  }
  parts.push(tex.slice(start));
  return parts.map((part) => part.trim()).filter((part) => part !== '');
}

function renderFormula(tex: string, display: boolean): HTMLElement {
  const parts = display ? topLevelParts(tex) : [tex];
  if (parts.length < 2) return render(tex, { display });
  const row = document.createElement('div');
  row.className = 'vf-display-row';
  row.append(...parts.map((part) => render(part, { display: true })));
  return row;
}

function fill(segments: Segment[], target: HTMLElement): void {
  const formulas: { tex: string; display: boolean }[] = [];
  const markup = segments
    .map((segment) => {
      if (segment.kind === 'html') return segment.html;
      formulas.push(segment);
      return `<span data-vf-tex="${formulas.length - 1}"></span>`;
    })
    .join('');
  const template = document.createElement('template');
  template.innerHTML = markup;
  for (const marker of template.content.querySelectorAll<HTMLElement>('[data-vf-tex]')) {
    const formula = formulas[Number(marker.dataset.vfTex)];
    marker.replaceWith(renderFormula(formula.tex, formula.display));
  }
  target.replaceChildren(template.content);
}

export function renderText(html: string, target: HTMLElement): void {
  fill(html.includes('\n') ? paragraphs(html, false) : splitTex(html), target);
}

export function renderMarkdown(source: string, target: HTMLElement): void {
  fill(paragraphs(source, true), target);
}
