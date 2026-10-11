import katex from 'katex';
import type { FormulaAdapter } from './adapter';
import './katex-style';

const renderedHtml = new Map<string, string>();
const ARRAY_STRETCH = '1.1';

export const katexAdapter: FormulaAdapter = {
  render(tex, display, target) {
    const key = `${display ? 'display' : 'inline'}:${tex}`;
    let html = renderedHtml.get(key);
    if (html === undefined) {
      html = katex.renderToString(tex, { displayMode: display, throwOnError: true, macros: { '\\arraystretch': ARRAY_STRETCH }, trust: (context) => context.command === '\\htmlData', strict: (code: string) => (code === 'htmlExtension' ? 'ignore' : 'warn') });
      renderedHtml.set(key, html);
    }
    target.innerHTML = html;
  },
};
