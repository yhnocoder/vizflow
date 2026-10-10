import type { FormulaAdapter } from './adapter';
import { katexAdapter } from './katex';

let activeAdapter: FormulaAdapter = katexAdapter;

export function setAdapter(adapter: FormulaAdapter): void {
  activeAdapter = adapter;
}

export function render(tex: string, options: { display?: boolean } = {}): HTMLElement {
  const placeholder = document.createElement('span');
  placeholder.className = 'vf-tex';
  void activeAdapter.render(tex, options.display ?? false, placeholder);
  return placeholder;
}
