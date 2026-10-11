import { createEffect } from 'solid-js';
import { renderText } from '../formula';

export function Rich(props: { text: string; class?: string }) {
  return <span class={props.class} ref={(el) => createEffect(() => renderText(props.text, el))} />;
}

export function plain(text: string): string {
  const el = document.createElement('span');
  renderText(text, el);
  for (const k of el.querySelectorAll('.katex')) {
    k.querySelectorAll('annotation').forEach((n) => n.remove());
    k.replaceWith(k.querySelector('.katex-mathml')?.textContent ?? '');
  }
  return el.textContent ?? '';
}

export const indexed = (tex: string, i: number) => `$${tex}_{${i}}$`;
