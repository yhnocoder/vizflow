export function shown(el: Element | null): string {
  const copy = el!.cloneNode(true) as Element;
  for (const k of copy.querySelectorAll('.katex')) k.replaceWith(`$${k.querySelector('annotation')?.textContent ?? ''}$`);
  return copy.textContent ?? '';
}
