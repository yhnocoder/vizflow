import katex from 'katex';
import katexCss from 'katex/dist/katex.min.css?raw';
import type { FormulaAdapter } from './adapter';

const styleId = 'vf-katex-style';
const fontDirectory = 'fonts/';

if (!document.getElementById(styleId)) {
  const fontBase = new URL(fontDirectory, import.meta.url).href;
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = katexCss.replaceAll('url(fonts/', `url(${fontBase}`);
  document.head.prepend(style);
}

const renderedHtml = new Map<string, string>();

export const katexAdapter: FormulaAdapter = {
  render(tex, display, target) {
    const key = `${display ? 'display' : 'inline'}:${tex}`;
    let html = renderedHtml.get(key);
    if (html === undefined) {
      html = katex.renderToString(tex, { displayMode: display, throwOnError: true });
      renderedHtml.set(key, html);
    }
    target.innerHTML = html;
  },
};
