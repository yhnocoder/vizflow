import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium } from '@playwright/test';

const usage = `usage: npm run shots -- <page> [options]

  <page>             page path under the repo root, e.g. examples/qwen3-dsl.html
  --out <dir>        output directory (default: a new temporary directory)
  --steps <list>     comma-separated steps: an index, or back / back+N counted
                     from the first backward step (default: 0)
  --hover <fx,fy>    at each step, search outward from this canvas fraction
                     for a bar tooltip (.vf-tip) and capture it
  --edge <fx,fy>     at each step, search outward from this canvas fraction
                     for an edge card (.vf-pop) and capture it
  --url <base>       use a running server instead of starting vite
  --wait <ms>        wait after each step switch (default: 2500)`;

const { values: opts, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    out: { type: 'string' },
    steps: { type: 'string', default: '0' },
    hover: { type: 'string' },
    edge: { type: 'string' },
    url: { type: 'string' },
    wait: { type: 'string', default: '2500' },
    help: { type: 'boolean', short: 'h' },
  },
});

if (opts.help || positionals.length !== 1) {
  console.log(usage);
  process.exit(opts.help ? 0 : 1);
}

function fraction(text, name) {
  const parts = text.split(',').map(Number);
  if (parts.length !== 2 || parts.some((v) => !(v >= 0 && v <= 1))) {
    console.error(`--${name} expects two fractions in [0, 1], e.g. 0.5,0.45`);
    process.exit(1);
  }
  return parts;
}

const page = positionals[0].startsWith('/') ? positionals[0].slice(1) : positionals[0];
const out = opts.out ?? mkdtempSync(join(tmpdir(), 'vizflow-shots-'));
const wait = Number(opts.wait);
const hoverAt = opts.hover && fraction(opts.hover, 'hover');
const edgeAt = opts.edge && fraction(opts.edge, 'edge');
mkdirSync(out, { recursive: true });

function freePort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function startVite() {
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  const child = spawn('npx', ['vite', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
    stdio: ['ignore', 'ignore', 'inherit'],
    detached: true,
  });
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`vite exited with code ${child.exitCode}`);
    try {
      if ((await fetch(`${base}/examples/index.html`)).ok) return { base, stop: () => process.kill(-child.pid) };
    } catch {}
    await new Promise((r) => setTimeout(r, 200));
  }
  process.kill(-child.pid);
  throw new Error('vite did not start within 60 s');
}

function resolveStep(token, backStart, count) {
  const match = /^back(?:\+(\d+))?$/.exec(token);
  const index = match ? backStart + Number(match[1] ?? 0) : Number(token);
  if (!Number.isInteger(index) || index < 0 || index >= count || (match && backStart < 0)) {
    throw new Error(`step '${token}' is out of range (page has ${count} steps, first backward step ${backStart})`);
  }
  return index;
}

const server = opts.url ? { base: opts.url.replace(/\/$/, ''), stop: () => {} } : await startVite();
const executablePath = process.env.VIZFLOW_CHROMIUM;
const browser = await chromium.launch({
  ...(executablePath ? { executablePath } : {}),
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});

try {
  const tab = await browser.newPage({ viewport: { width: 1440, height: 1000 }, colorScheme: 'light' });
  tab.on('console', (m) => { if (m.type() === 'error') console.log('console error:', m.text()); });
  tab.on('pageerror', (e) => console.log('page error:', e.message));
  await tab.goto(`${server.base}/${page}`);
  await tab.waitForSelector('body[data-ready=true]', { timeout: 60000 });
  const box = await (await tab.$('.vf-scene canvas')).boundingBox();
  const { count, backStart } = await tab.evaluate(() => ({
    count: window.vizflow.steps.list.length,
    backStart: window.vizflow.steps.list.findIndex((s) => s.dir === 'backward'),
  }));
  const steps = opts.steps.split(',').map((t) => resolveStep(t.trim(), backStart, count));

  const shoot = async (name) => {
    const path = join(out, `${name}.png`);
    await tab.screenshot({ path, clip: box });
    console.log(path);
  };
  const search = async (selector, [fx, fy]) => {
    const cx = box.x + box.width * fx;
    const cy = box.y + box.height * fy;
    for (let r = 0; r < 400; r += 5) {
      for (let a = 0; a < 16; a++) {
        await tab.mouse.move(cx + r * Math.cos((a * Math.PI) / 8), cy + r * Math.sin((a * Math.PI) / 8));
        if (await tab.$eval(selector, (el) => !el.hidden)) return tab.$eval(selector, (el) => el.textContent);
      }
    }
    return null;
  };

  for (const step of steps) {
    await tab.evaluate((s) => window.vizflow.steps.go(s), step);
    await tab.mouse.move(2, 2);
    await tab.waitForTimeout(wait);
    await shoot(`step-${step}`);
    if (edgeAt) {
      const text = await search('.vf-pop', edgeAt);
      console.log(`step ${step} edge card:`, text ?? 'not found');
      if (text) { await tab.waitForTimeout(300); await shoot(`step-${step}-edge`); }
      await tab.mouse.move(2, 2);
      await tab.waitForTimeout(300);
    }
    if (hoverAt) {
      const text = await search('.vf-tip', hoverAt);
      console.log(`step ${step} tooltip:`, text ?? 'not found');
      if (text) {
        await tab.waitForTimeout(400);
        await shoot(`step-${step}-hover`);
        const hide = await tab.addStyleTag({ content: '.vf-tip{visibility:hidden !important}' });
        await tab.waitForTimeout(200);
        await shoot(`step-${step}-hover-notip`);
        await hide.evaluate((el) => el.remove());
      }
      await tab.mouse.move(2, 2);
    }
  }
  console.log(out);
} finally {
  await browser.close();
  server.stop();
}
