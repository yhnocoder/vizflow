// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { graph } from '../../src/autograd';
import { katexAdapter, renderText, setAdapter } from '../../src/formula';
import { compileTemplate } from '../../src/steps';

const rendered: { tex: string; display: boolean }[] = [];

beforeEach(() => {
  rendered.length = 0;
  setAdapter({
    render(tex, display, target) {
      rendered.push({ tex, display });
      target.textContent = display ? `[${tex}]` : `(${tex})`;
    },
  });
});

afterEach(() => setAdapter(katexAdapter));

function markup(text: string): string {
  const target = document.createElement('div');
  renderText(text, target);
  for (const formula of target.querySelectorAll('.vf-tex')) formula.replaceWith(formula.textContent ?? '');
  return target.innerHTML;
}

describe('renderText without a newline', () => {
  it('keeps the text exactly as before, without adding paragraphs', () => {
    expect(markup('a  b <b>c</b> $x$ $$y$$')).toBe('a  b <b>c</b> (x) [y]');
  });
});

describe('renderText with several lines', () => {
  it('removes indentation, joins lines without a separator and wraps the block in a paragraph', () => {
    const text = String.raw`
      模型输出经过 RMSNorm 得到 $\mathbf{h}$，
      lm_head 把它投影到词表大小。
    `;
    expect(markup(text)).toBe('<p>模型输出经过 RMSNorm 得到 (\\mathbf{h})，lm_head 把它投影到词表大小。</p>');
  });

  it('joins English lines with one space and drops trailing spaces', () => {
    expect(markup('\n  The cat\n  sat on   \n  the mat.\n')).toBe('<p>The cat sat on the mat.</p>');
  });

  it('joins lines directly only when CJK characters meet across the break', () => {
    expect(markup('\n  中文\n  继续\n')).toBe('<p>中文继续</p>');
    expect(markup('\n  かな\n  カナ\n  한글\n')).toBe('<p>かなカナ한글</p>');
    expect(markup('\n  得到 logits\n  中文\n')).toBe('<p>得到 logits 中文</p>');
    expect(markup('\n  中文\n  PyTorch\n')).toBe('<p>中文 PyTorch</p>');
    expect(markup('\n  第一句。\n  PyTorch 里\n')).toBe('<p>第一句。PyTorch 里</p>');
    expect(markup('\n  softmax\n  （归一化）\n')).toBe('<p>softmax（归一化）</p>');
    expect(markup('\n  全角\n  ＡＢ\n')).toBe('<p>全角ＡＢ</p>');
  });

  it('puts a space between a line and a formula or tag on the next line', () => {
    expect(markup('\n  得到\n  $x$\n  的值\n')).toBe('<p>得到 (x) 的值</p>');
    expect(markup('\n  得到 $x$，\n  其中\n')).toBe('<p>得到 (x)，其中</p>');
  });

  it('splits blocks on blank lines and leaves a display formula block outside paragraphs', () => {
    const text = String.raw`
      第一段。

      第二段
      继续。

      $$\ell = -\log p_t$$
    `;
    expect(markup(text)).toBe('<p>第一段。</p><p>第二段继续。</p>[\\ell = -\\log p_t]');
  });

  it('keeps newlines inside a display formula', () => {
    const text = '\n  $$\\begin{aligned}\n    a &= b \\\\\n    c &= d\n  \\end{aligned}$$\n';
    markup(text);
    expect(rendered).toEqual([{ tex: '\\begin{aligned}\n  a &= b \\\\\n  c &= d\n\\end{aligned}', display: true }]);
  });

  it('does not join a TeX command with the next line inside an inline formula', () => {
    markup('\n  这是 $\\log\n  p$ 的值。\n');
    expect(rendered).toEqual([{ tex: '\\log\np', display: false }]);
  });

  it('leaves a block that starts with an HTML block tag unwrapped and joins its lines', () => {
    const text = `
      <ul>
        <li>第一项</li>
        <li>第二项</li>
      </ul>

      <div class="note">说明</div>
    `;
    expect(markup(text)).toBe('<ul> <li>第一项</li> <li>第二项</li> </ul><div class="note">说明</div>');
  });

  it('keeps the lines of a pre block', () => {
    const text = '\n  <pre>a = 1\n  b = 2</pre>\n';
    expect(markup(text)).toBe('<pre>a = 1\nb = 2</pre>');
  });

  it('wraps a block that starts with inline markup in a paragraph', () => {
    expect(markup('\n  <code>F.cross_entropy</code> 内部是\n  <code>log_softmax</code>。\n')).toBe(
      '<p><code>F.cross_entropy</code> 内部是 <code>log_softmax</code>。</p>',
    );
  });

  it('wraps text followed by a display formula in the same block in one paragraph', () => {
    expect(markup('\n  值为\n  $$x$$\n')).toBe('<p>值为 [x]</p>');
  });
});

describe('templates over several lines', () => {
  it('compile with newlines around and inside the braces', () => {
    const g = graph();
    g.input('x', [2], [0.5, -1.25]);
    const evaluate = compileTemplate(g, String.raw`
      第一个是 {{x[0]:2}}，
      第二个是 $x_2 = {{
        x[1]:2
      }}$
    `);
    expect(evaluate(0)).toContain('第一个是 0.50，');
    expect(evaluate(0)).toContain('$x_2 = -1.25$');
  });
});
