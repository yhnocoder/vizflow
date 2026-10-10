import katex from 'katex';
import type { FormulaAdapter } from './adapter';
import './katex-style';

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
