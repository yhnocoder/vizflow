import { spawn, type ChildProcess } from 'node:child_process';
import { rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { join } from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import { buildViteProjectToTemp } from '../bundle/build';

let workspace = '';
let project = '';
const servers: ChildProcess[] = [];

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

async function startVite(command: 'dev' | 'preview'): Promise<string> {
  const port = await freePort();
  const args = [join(project, 'node_modules/vite/bin/vite.js'), ...(command === 'dev' ? [] : ['preview']), '--host', '127.0.0.1', '--port', String(port), '--strictPort'];
  const server = spawn(process.execPath, args, { cwd: project, stdio: 'pipe' });
  servers.push(server);
  const origin = `http://127.0.0.1:${port}`;
  await new Promise<void>((resolve, reject) => {
    let output = '';
    const onData = (chunk: Buffer) => {
      output += chunk.toString();
      if (output.replace(/\x1b\[[0-9;]*m/g, '').includes(origin)) resolve();
    };
    server.stdout?.on('data', onData);
    server.stderr?.on('data', onData);
    server.once('exit', (code) => reject(new Error(`vite ${command} exited with ${code}: ${output}`)));
  });
  return origin;
}

test.beforeAll(async () => {
  test.setTimeout(300_000);
  ({ workspace, project } = buildViteProjectToTemp());
});

test.afterAll(async () => {
  for (const server of servers) server.kill();
  for (const directory of [workspace, project]) if (directory) rmSync(directory, { recursive: true, force: true });
});

async function openExample(page: Page, origin: string) {
  const fontResponses: { url: string; status: number }[] = [];
  const problems: string[] = [];
  page.on('response', (response) => {
    if (/\.(?:woff2?|ttf)(?:\?|$)/.test(response.url())) fontResponses.push({ url: response.url(), status: response.status() });
  });
  page.on('requestfailed', (request) => problems.push(`failed: ${request.url()}`));
  page.on('console', (message) => {
    if (message.type() === 'error' || message.text().includes('Multiple instances of Three.js')) problems.push(message.text());
  });
  page.on('pageerror', (error) => problems.push(error.message));
  await page.goto(`${origin}/`);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true', { timeout: 60_000 });
  await expect(page.locator('.vf-scene canvas')).toHaveCount(1);
  await expect(page.locator('.vf-st')).toHaveCount(19);
  await expect(page.locator('.vf-caption .katex').first()).toBeVisible();
  const loaded = await page.evaluate(async () => {
    await document.fonts.ready;
    const faces = ['400 16px "Source Serif 4"', '400 16px "Geist"', '400 16px "Geist Mono"', '400 16px "KaTeX_Main"'];
    const results = await Promise.all(faces.map((face) => document.fonts.load(face)));
    return results.map((matched) => matched.length > 0 && matched.every((face) => face.status === 'loaded'));
  });
  expect(loaded).toEqual([true, true, true, true]);
  expect(fontResponses.length).toBeGreaterThan(0);
  expect(fontResponses.filter(({ status }) => status !== 200)).toEqual([]);
  expect(problems).toEqual([]);
  return page.evaluate(() => ({
    check: (window as unknown as { vizflow: { g: { check: boolean } } }).vizflow.g.check,
    three: document.body.dataset.three,
  }));
}

test('the production build renders the page with fonts, one three.js and the check off', async ({ page }) => {
  test.setTimeout(120_000);
  const origin = await startVite('preview');
  expect(await openExample(page, origin)).toEqual({ check: false, three: expect.stringMatching(/^\d+$/) });
});

test('the dev server renders the page with fonts, one three.js and the check on', async ({ page }) => {
  test.setTimeout(120_000);
  const origin = await startVite('dev');
  expect(await openExample(page, origin)).toEqual({ check: true, three: expect.stringMatching(/^\d+$/) });
});
