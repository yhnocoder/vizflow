import { createEffect } from 'solid-js';
import { renderText } from '../formula';

export function Rich(props: { text: string; class?: string }) {
  return <span class={props.class} ref={(el) => createEffect(() => renderText(props.text, el))} />;
}

export const plain = (text: string) => text.replace(/\$/g, '');

export const indexed = (tex: string, i: number) => `$${tex}_{${i}}$`;
