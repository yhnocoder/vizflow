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

const blockTag = /^<(?:p|div|ul|ol|pre|table|h[1-6]|blockquote)[\s>]/i;
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

function joinLines(block: Segment[]): Segment[] {
  return block.map((segment) => (segment.kind === 'html' ? { kind: 'html', html: segment.html.replace(/\n[ \t]*/g, '') } : segment));
}

function paragraphs(source: string): Segment[] {
  return splitBlocks(splitTex(dedent(source))).flatMap((block) => {
    const first = block[0];
    if (block.length === 1 && first.kind === 'tex' && first.display) return block;
    if (first.kind === 'html' && preTag.test(first.html)) return block;
    const joined = joinLines(block);
    if (first.kind === 'html' && blockTag.test(first.html)) return joined;
    return [{ kind: 'html', html: '<p>' }, ...joined, { kind: 'html', html: '</p>' }];
  });
}

export function renderText(html: string, target: HTMLElement): void {
  const formulas: { tex: string; display: boolean }[] = [];
  const segments = html.includes('\n') ? paragraphs(html) : splitTex(html);
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
    marker.replaceWith(render(formula.tex, { display: formula.display }));
  }
  target.replaceChildren(template.content);
}
