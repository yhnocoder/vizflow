import type { FormulaAdapter } from './adapter';
import { katexAdapter } from './katex';

let activeAdapter: FormulaAdapter = katexAdapter;

export function setAdapter(adapter: FormulaAdapter): void {
  activeAdapter = adapter;
}

export function render(tex: string, options: { display?: boolean } = {}): HTMLElement {
  const placeholder = document.createElement('span');
  placeholder.className = 'vf-tex';
  const showError = () => {
    placeholder.textContent = tex;
    placeholder.classList.add('vf-tex-error');
  };
  try {
    const pending = activeAdapter.render(tex, options.display ?? false, placeholder);
    if (pending instanceof Promise) pending.catch(showError);
  } catch {
    showError();
  }
  return placeholder;
}
