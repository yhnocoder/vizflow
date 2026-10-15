import katex from 'katex';
import type { FormulaAdapter } from './adapter';
import './katex-style';

const rendered = new Map<string, HTMLTemplateElement>();
const ARRAY_STRETCH = '1.1';

export const katexAdapter: FormulaAdapter = {
  render(tex, display, target) {
    const key = `${display ? 'display' : 'inline'}:${tex}`;
    let template = rendered.get(key);
    if (template === undefined) {
      template = document.createElement('template');
      template.innerHTML = katex.renderToString(tex, { displayMode: display, throwOnError: true, macros: { '\\arraystretch': ARRAY_STRETCH }, trust: (context) => context.command === '\\htmlData', strict: (code: string) => (code === 'htmlExtension' ? 'ignore' : 'warn') });
      rendered.set(key, template);
    }
    target.replaceChildren(template.content.cloneNode(true));
  },
};
