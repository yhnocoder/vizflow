// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { katexAdapter, render, renderText, setAdapter, type FormulaAdapter } from '../../src/formula';

function deferredAdapter() {
  const pending: (() => void)[] = [];
  const adapter: FormulaAdapter = {
    render(tex, display, target) {
      return new Promise<void>((resolve) => {
        pending.push(() => {
          target.textContent = `${display ? 'display' : 'inline'}[${tex}]`;
          resolve();
        });
      });
    },
  };
  const flush = () => pending.splice(0).forEach((fill) => fill());
  return { adapter, flush };
}

function showLesson(container: HTMLElement): HTMLElement {
  const formula = render('\\frac{a}{b}', { display: true });
  container.append(formula);
  return formula;
}

afterEach(() => {
  setAdapter(katexAdapter);
  document.body.replaceChildren();
});

describe('render', () => {
  it('returns a placeholder span filled synchronously by KaTeX', () => {
    const element = render('x^2');
    expect(element.tagName).toBe('SPAN');
    expect(element.classList.contains('vf-tex')).toBe(true);
    expect(element.querySelector('.katex')).not.toBeNull();
    expect(element.querySelector('.katex-display')).toBeNull();
  });

  it('renders display math with katex-display', () => {
    expect(render('x^2', { display: true }).querySelector('.katex-display')).not.toBeNull();
  });

  it('returns identical markup for repeated formulas', () => {
    expect(render('\\sum_i x_i').innerHTML).toBe(render('\\sum_i x_i').innerHTML);
  });

  it('shows the raw TeX with vf-tex-error when KaTeX fails', () => {
    const element = render('\\frac{a');
    expect(element.classList.contains('vf-tex-error')).toBe(true);
    expect(element.textContent).toBe('\\frac{a');
  });

  it('lets the caller insert the placeholder before an async adapter fills it', async () => {
    const { adapter, flush } = deferredAdapter();
    setAdapter(adapter);
    const formula = showLesson(document.body);
    expect(formula.isConnected).toBe(true);
    expect(formula.textContent).toBe('');
    flush();
    await Promise.resolve();
    expect(formula.textContent).toBe('display[\\frac{a}{b}]');
  });

  it('shows the raw TeX with vf-tex-error when an async adapter rejects', async () => {
    let reject: (reason: Error) => void = () => {};
    setAdapter({
      render: () =>
        new Promise<void>((_resolve, rejectRender) => {
          reject = rejectRender;
        }),
    });
    const formula = showLesson(document.body);
    expect(formula.classList.contains('vf-tex-error')).toBe(false);
    reject(new Error('adapter failed'));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(formula.classList.contains('vf-tex-error')).toBe(true);
    expect(formula.textContent).toBe('\\frac{a}{b}');
  });

  it('shows the raw TeX with vf-tex-error when an adapter throws synchronously', () => {
    setAdapter({
      render(_tex, _display, target) {
        target.textContent = 'partial';
        throw new Error('adapter failed');
      },
    });
    const formula = render('x');
    expect(formula.classList.contains('vf-tex-error')).toBe(true);
    expect(formula.textContent).toBe('x');
  });

  it('runs the same caller code with the KaTeX adapter', () => {
    const formula = showLesson(document.body);
    expect(formula.isConnected).toBe(true);
    expect(formula.querySelector('.katex-display')).not.toBeNull();
  });

  it('injects the KaTeX stylesheet once', () => {
    expect(document.querySelectorAll('#vf-katex-style')).toHaveLength(1);
  });
});

describe('renderText', () => {
  function textWith(adapterHtml: string): HTMLElement {
    const { adapter, flush } = deferredAdapter();
    setAdapter(adapter);
    const target = document.createElement('p');
    renderText(adapterHtml, target);
    flush();
    return target;
  }

  it('replaces inline and display formulas and keeps the HTML', () => {
    const target = textWith('<b>loss</b> $L = -\\log p$ and $$\\sum_i x_i$$ end');
    expect(target.innerHTML).toBe(
      '<b>loss</b> <span class="vf-tex">inline[L = -\\log p]</span> and <span class="vf-tex">display[\\sum_i x_i]</span> end',
    );
  });

  it('renders formulas nested inside HTML elements', () => {
    const target = textWith('<em>see $x$</em>');
    expect(target.innerHTML).toBe('<em>see <span class="vf-tex">inline[x]</span></em>');
  });

  it('treats \\$ as a literal dollar sign outside formulas', () => {
    expect(textWith('costs \\$5 and \\$7').innerHTML).toBe('costs $5 and $7');
  });

  it('keeps \\$ inside a formula as TeX', () => {
    expect(textWith('$a \\$ b$').textContent).toBe('inline[a \\$ b]');
  });

  it('leaves an unclosed delimiter as text', () => {
    expect(textWith('price $5').innerHTML).toBe('price $5');
  });

  it('replaces the existing content of the target', () => {
    const target = document.createElement('p');
    target.textContent = 'old';
    renderText('new', target);
    expect(target.innerHTML).toBe('new');
  });

  it('renders with KaTeX by default', () => {
    const target = document.createElement('p');
    renderText('value $x^2$', target);
    expect(target.querySelector('.vf-tex .katex')).not.toBeNull();
  });
});
