import katex from 'katex';
import katexCss from 'katex/dist/katex.min.css?inline';
import type { FormulaAdapter } from './adapter';

const styleId = 'vf-katex-style';
if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = katexCss;
  document.head.prepend(style);
}

const renderedHtml = new Map<string, string>();

export const katexAdapter: FormulaAdapter = {
  render(tex, display, target) {
    const key = `${display ? 'display' : 'inline'}:${tex}`;
    try {
      let html = renderedHtml.get(key);
      if (html === undefined) {
        html = katex.renderToString(tex, { displayMode: display, throwOnError: true });
        renderedHtml.set(key, html);
      }
      target.innerHTML = html;
    } catch {
      target.textContent = tex;
      target.classList.add('vf-tex-error');
    }
  },
};
