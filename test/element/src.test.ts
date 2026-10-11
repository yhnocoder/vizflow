// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import '../../src/element';

const PAGE = `
title: 一个小页面
lede: 输入是 $\\mathbf{x}$。
hint: 拖动旋转
dims: {}
inputs:
  x: { value: [1, 2] }
body:
  - y: x * 2
  - l: mean(y)
panels:
  - loss
`;

afterEach(() => {
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

function serve(files: Record<string, string>): string[] {
  const requested: string[] = [];
  vi.stubGlobal('fetch', async (url: string) => {
    requested.push(url);
    const text = files[new URL(url).pathname];
    return { ok: text !== undefined, status: text === undefined ? 404 : 200, text: async () => text ?? '' };
  });
  return requested;
}

it('loads a page description from src and mounts it with a header', async () => {
  const requested = serve({ '/pages/small.yaml': PAGE });
  const el = document.createElement('vizflow-page');
  const loaded = new Promise((resolve) => el.addEventListener('load', resolve));
  el.setAttribute('src', 'http://example.test/pages/small.yaml');
  document.body.append(el);
  await loaded;
  expect(requested).toEqual(['http://example.test/pages/small.yaml']);
  expect(el.src).toBe('http://example.test/pages/small.yaml');
  expect(el.compiled?.nodes.get('l')?.value.data[0]).toBe(3);
  expect(el.graph).toBe(el.compiled?.graph);
  expect(el.scene?.hint).toBe('拖动旋转');
  expect(el.querySelector(':scope > .vf-header .vf-title')?.textContent).toBe('一个小页面');
  expect(el.querySelector(':scope > .vf-header .vf-lede .katex')).not.toBeNull();
  expect(el.querySelector(':scope > .vf-split > .vf-side')).not.toBeNull();
  expect(el.createdSteps?.list.map((s) => s.title)).toEqual(['总览', 'y', 'l', '∂ℓ/∂y']);
});

it('shows the compile error in the element and reports it', async () => {
  serve({ '/pages/bad.yaml': 'title: t\ndims: {}\ninputs: {}\nbody:\n  - y: nope + 1\n' });
  const reported = vi.fn();
  vi.stubGlobal('reportError', reported);
  const el = document.createElement('vizflow-page');
  el.src = 'http://example.test/pages/bad.yaml';
  document.body.append(el);
  await vi.waitFor(() => expect(reported).toHaveBeenCalledTimes(1));
  expect(el.querySelector(':scope > .vf-load-error')?.textContent).toContain("bad.yaml:5:8 body[0].y: name 'nope' is not defined in the page");
  expect(el.compiled).toBeUndefined();
});
