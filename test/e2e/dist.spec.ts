import { readFile, writeFile } from 'node:fs/promises';
import { createServer, type Server } from 'node:http';
import { rmSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { expect, test } from '@playwright/test';
import { buildDistToTemp } from '../bundle/build';

const contentTypes: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.woff2': 'font/woff2',
};

const fixtureHtml = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<link rel="icon" href="data:,">
</head>
<body>
<h1 style="font-family: var(--serif)">Heading <em>italic</em></h1>
<p style="font-family: var(--sans)">Body text <b>bold</b></p>
<code style="font-family: var(--mono)">0.495</code>
<div id="formula"></div>
<script type="module">
import './dist/theme.js';
import { render } from './dist/formula.js';
document.getElementById('formula').append(render('\\\\mathcal{L} = -\\\\log p_{t} + \\\\sum_i \\\\mathbf{z}_i', { display: true }));
document.body.dataset.ready = 'true';
</script>
</body>
</html>
`;

const elementHtml = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<link rel="icon" href="data:,">
</head>
<body>
<vizflow-page></vizflow-page>
<script type="module">
import { graph } from './dist/core.js';
import { board } from './dist/scene3d.js';
import './dist/element.js';
const g = graph();
const h = g.input('h', [1, 2], [0.5, -1]);
const W = g.param('W', [3, 2], [[1, 0], [0, 1], [1, 1]]);
const t = g.label('t', [0], { tokens: ['a', 'b', 'c'] });
const z = g.matmul(h, W.T, { name: 'z' });
const logp = g.log_softmax(z, -1, { name: 'log p' });
const loss = g.nll_loss(logp, t, { name: 'loss' });
const page = document.querySelector('vizflow-page');
page.graph = g;
page.steps = {
  forward: [
    { show: [h, W.T, z], title: 'lm_head', text: String.raw\`
      $z = hW^{\\top}$，
      目标是 {{t:token}}。
    \` },
    { show: [logp, loss], title: 'loss', text: String.raw\`$$\\ell = {{loss:3}}$$\` },
  ],
};
page.scene = { layout: [board(h, W.T, z)] };
page.panels = ['loss', { tokens: t, columns: [z] }];
page.createdSteps.go(1);
document.body.dataset.ready = 'true';
</script>
</body>
</html>
`;

const faces = [
  '400 16px "Source Serif 4"',
  'italic 400 16px "Source Serif 4"',
  '400 16px "Geist"',
  '600 16px "Geist"',
  '400 16px "Geist Mono"',
  '400 16px "KaTeX_Main"',
  'italic 400 16px "KaTeX_Math"',
  '400 16px "KaTeX_Caligraphic"',
];

let workspace = '';
let server: Server;
let origin = '';

test.beforeAll(async () => {
  test.setTimeout(180_000);
  workspace = buildDistToTemp().workspace;
  await writeFile(join(workspace, 'index.html'), fixtureHtml);
  await writeFile(join(workspace, 'element.html'), elementHtml);
  server = createServer(async (request, response) => {
    const path = normalize(decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname));
    try {
      const body = await readFile(join(workspace, path));
      response.writeHead(200, { 'content-type': contentTypes[extname(path)] ?? 'application/octet-stream' });
      response.end(body);
    } catch {
      response.writeHead(404);
      response.end();
    }
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  origin = `http://127.0.0.1:${typeof address === 'object' && address ? address.port : 0}`;
});

test.afterAll(async () => {
  await new Promise<void>((resolve) => server?.close(() => resolve()) ?? resolve());
  if (workspace) rmSync(workspace, { recursive: true, force: true });
});

test('fonts referenced by the built theme and formula entries load from dist/fonts', async ({ page }) => {
  const fontResponses: { url: string; status: number }[] = [];
  const failed: string[] = [];
  const errors: string[] = [];
  page.on('response', (response) => {
    if (response.url().endsWith('.woff2')) fontResponses.push({ url: response.url(), status: response.status() });
  });
  page.on('requestfailed', (request) => failed.push(request.url()));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));

  await page.goto(`${origin}/index.html`);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('#formula .katex-display')).toHaveCount(1);
  const statuses = await page.evaluate(async (fontList) => {
    const loaded = await Promise.all(fontList.map((font) => document.fonts.load(font)));
    await document.fonts.ready;
    return loaded.map((matched) => matched.map((face) => face.status));
  }, faces);
  statuses.forEach((status, index) => expect(status, faces[index]).toContain('loaded'));

  expect(failed).toEqual([]);
  expect(errors).toEqual([]);
  const fontDirectory = `${origin}/dist/fonts/`;
  expect(fontResponses.map(({ url }) => url.slice(0, url.lastIndexOf('/') + 1))).toEqual(fontResponses.map(() => fontDirectory));
  expect(fontResponses.filter(({ status }) => status !== 200)).toEqual([]);
  const names = fontResponses.map(({ url }) => url.slice(fontDirectory.length));
  expect(names).toEqual(
    expect.arrayContaining([
      'source-serif-4-latin-400-normal.woff2',
      'source-serif-4-latin-400-italic.woff2',
      'geist-latin-400-normal.woff2',
      'geist-mono-latin-400-normal.woff2',
      'KaTeX_Main-Regular.woff2',
      'KaTeX_Math-Italic.woff2',
    ]),
  );
});

test('the standalone element entry renders a page without a build step', async ({ page }) => {
  const errors: string[] = [];
  const fontStatuses: number[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('response', (response) => {
    if (response.url().endsWith('.woff2')) fontStatuses.push(response.status());
  });
  await page.goto(`${origin}/element.html`);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('vizflow-page .vf-stage .vf-scene canvas')).toHaveCount(1);
  await expect(page.locator('vizflow-page .vf-st[aria-current="step"]')).toHaveText('lm_head');
  await expect(page.locator('vizflow-page .vf-caption-body > p')).toHaveCount(1);
  await expect(page.locator('vizflow-page .vf-caption-body > p')).toContainText(/，目标是 a。$/);
  await expect(page.locator('vizflow-page .vf-side .vf-loss-value')).toHaveCount(1);
  await page.evaluate(() => document.fonts.ready);
  expect(fontStatuses.length).toBeGreaterThan(0);
  expect(fontStatuses.filter((status) => status !== 200)).toEqual([]);
  expect(errors).toEqual([]);
});
