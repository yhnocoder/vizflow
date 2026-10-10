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

export function renderText(html: string, target: HTMLElement): void {
  const formulas: { tex: string; display: boolean }[] = [];
  const markup = splitTex(html)
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
