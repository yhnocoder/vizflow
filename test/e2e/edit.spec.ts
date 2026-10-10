import { join } from 'node:path';
import { type Page, expect, test } from '@playwright/test';

const PAGE = '/examples/edit-lmhead.html';
const SETTLE_MS = 400;
const SAME_PX = 1.5;

type Box = { x: number; y: number };

async function shot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: `test-results/${name}.png` });
  const dir = process.env.VIZFLOW_SHOT_DIR;
  if (dir) await page.screenshot({ path: join(dir, `${name}.png`) });
}

async function open(page: Page, url = PAGE): Promise<string[]> {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 860 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(url);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('.vf-scene canvas')).toHaveCount(1);
  await page.waitForTimeout(SETTLE_MS);
  return errors;
}

async function copyConfig(page: Page): Promise<string> {
  await page.locator('.vf-edit .vf-chip', { hasText: '复制配置' }).click();
  const output = page.locator('.vf-edit-code');
  await expect(output).toBeVisible();
  return output.inputValue();
}

async function pasteAndReload(page: Page, code: string): Promise<void> {
  const pattern = /^ {2}layout: \[\n[\s\S]*?\n {2}\],\n/m;
  const indented = code
    .split('\n')
    .map((line) => `  ${line}`)
    .join('\n');
  let pasted = false;
  await page.route(/edit-lmhead\.html\?html-proxy/, async (route) => {
    const response = await route.fetch();
    const script = await response.text();
    expect(script).toMatch(pattern);
    pasted = true;
    await route.fulfill({ response, body: script.replace(pattern, `${indented}\n`) });
  });
  await page.reload();
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  expect(pasted).toBe(true);
  await page.waitForTimeout(SETTLE_MS);
}

async function positions(page: Page, selector: string): Promise<Box[]> {
  return page.locator(selector).evaluateAll((elements) =>
    elements.map((el) => {
      const r = el.getBoundingClientRect();
      return { x: r.left, y: r.top };
    }),
  );
}

function expectSame(actual: Box[], expected: Box[]): void {
  expect(actual.length).toBe(expected.length);
  actual.forEach((p, i) => {
    expect(Math.abs(p.x - expected[i].x), JSON.stringify({ p, e: expected[i] })).toBeLessThan(SAME_PX);
    expect(Math.abs(p.y - expected[i].y), JSON.stringify({ p, e: expected[i] })).toBeLessThan(SAME_PX);
  });
}

async function grabBlock(page: Page, name: string): Promise<Box> {
  const status = page.locator('.vf-edit-status');
  const anchors = await positions(page, `.vf-lb.vf-val[data-node="${name}"]`);
  const sizes = await page.locator(`.vf-lb.vf-val[data-node="${name}"]`).evaluateAll((els) => els.map((el) => el.getBoundingClientRect().width));
  for (let i = 0; i < anchors.length; i++) {
    const x = anchors[i].x + sizes[i] / 2;
    for (let dy = -6; dy >= -160; dy -= 6) {
      const y = anchors[i].y + dy;
      await page.mouse.move(x, y);
      if (!(await page.locator('.vf-scene canvas').evaluate((el) => el.classList.contains('vf-cursor-move')))) continue;
      await page.mouse.down();
      const text = (await status.textContent()) ?? '';
      if (text.startsWith(`${name}：`)) return { x, y };
      await page.mouse.up();
    }
  }
  throw new Error(`no draggable point found for ${name}`);
}

test('dragging a stack layer writes an offset that puts the block in the same place after pasting the config back', async ({ page }) => {
  const errors = await open(page);
  const watched = '.vf-lb.vf-val[data-node="log p"], .vf-lb.vf-ttl[data-node="log p"]';
  const before = await positions(page, watched);
  const start = await grabBlock(page, 'log p');
  await page.mouse.move(start.x + 60, start.y + 20, { steps: 4 });
  await page.mouse.move(start.x + 120, start.y + 40, { steps: 4 });
  await page.mouse.up();
  await page.waitForTimeout(SETTLE_MS);
  const dragged = await positions(page, watched);
  expect(Math.abs(dragged[0].x - before[0].x) + Math.abs(dragged[0].y - before[0].y)).toBeGreaterThan(10);
  const code = await copyConfig(page);
  expect(code).toContain("stack(g.node('z'), g.node('LSE'), g.node('log p'), g.node('loss'), { scales: { 'log p': { forward: 0.32, backward: 1.6 } }, offsets: { 'log p': [");
  expect(code).not.toContain('board(g.node(\'h\'), g.node(\'W.T\'), g.node(\'z\'), { at');
  await shot(page, 'edit-desktop-drag');
  await pasteAndReload(page, code);
  expectSame(await positions(page, watched), dragged);
  expect(errors).toEqual([]);
});

test('dragging a board block moves the board at, and the stack on top follows', async ({ page }) => {
  const errors = await open(page);
  const watched = '.vf-lb.vf-ttl[data-node="W.T"], .vf-lb.vf-node[data-node="LSE"]';
  const start = await grabBlock(page, 'z');
  await page.mouse.move(start.x - 80, start.y + 10, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(SETTLE_MS);
  const dragged = await positions(page, watched);
  const code = await copyConfig(page);
  expect(code).toMatch(/board\(g\.node\('h'\), g\.node\('W\.T'\), g\.node\('z'\), \{ scales: .*, at: \[-?[\d.]+, 0, -?[\d.]+\] \}\)/);
  expect(code).not.toContain('offsets');
  await pasteAndReload(page, code);
  expectSame(await positions(page, watched), dragged);
  expect(errors).toEqual([]);
});

test('changing board gap and stack level is written into the copied config', async ({ page }) => {
  const errors = await open(page);
  const gap = page.locator('.vf-edit-item[data-item="0"] input[name="gap"]');
  await expect(gap).toHaveValue('0.7');
  await gap.fill('1.2');
  await page.locator('.vf-edit-item[data-item="1"] input[name="level"]').fill('4.5');
  await page.waitForTimeout(SETTLE_MS);
  const after = await positions(page, '.vf-lb.vf-kk');
  const code = await copyConfig(page);
  expect(code).toContain('gap: 1.2');
  expect(code).toContain('level: 4.5');
  await pasteAndReload(page, code);
  await expect(page.locator('.vf-edit-item[data-item="0"] input[name="gap"]')).toHaveValue('1.2');
  expectSame(await positions(page, '.vf-lb.vf-kk'), after);
  expect(errors).toEqual([]);
});

test('dragging a block title writes a label offset that survives pasting the config back', async ({ page }) => {
  const errors = await open(page);
  const title = page.locator('.vf-lb.vf-ttl[data-node="z"]');
  const box = (await title.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 40, box.y + box.height / 2 - 30, { steps: 5 });
  await page.mouse.up();
  await page.waitForTimeout(SETTLE_MS);
  const moved = await positions(page, '.vf-lb.vf-ttl[data-node="z"]');
  const code = await copyConfig(page);
  expect(code).toMatch(/^labels: \{ z: \[-?\d+, -?\d+\] \},$/m);
  await pasteAndReload(page, code);
  expectSame(await positions(page, '.vf-lb.vf-ttl[data-node="z"]'), moved);
  expect(errors).toEqual([]);
});

test('setting the current camera angle as default writes camera.direction', async ({ page }) => {
  const errors = await open(page);
  const stage = (await page.locator('.vf-scene').boundingBox())!;
  await page.mouse.move(stage.x + stage.width - 60, stage.y + 80);
  await page.mouse.down();
  await page.mouse.move(stage.x + stage.width - 160, stage.y + 100, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(SETTLE_MS);
  await page.locator('.vf-edit .vf-chip', { hasText: '设为默认视角' }).click();
  await page.waitForTimeout(SETTLE_MS);
  const view = await positions(page, '.vf-lb.vf-ttl');
  const code = await copyConfig(page);
  const match = /^camera: \{ direction: \[(-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\] \},$/m.exec(code);
  expect(match).not.toBeNull();
  expect(match!.slice(1).map(Number)).not.toEqual([0.42, 0.78, 1]);
  await pasteAndReload(page, code);
  expectSame(await positions(page, '.vf-lb.vf-ttl'), view);
  expect(errors).toEqual([]);
});

test('the edit panel renders in light and dark themes', async ({ page }) => {
  const errors = await open(page);
  await page.locator('.vf-edit .vf-chip', { hasText: '复制配置' }).click();
  await expect(page.locator('.vf-edit-code')).toBeVisible();
  await shot(page, 'edit-desktop');
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.evaluate(() => {
    document.documentElement.dataset.theme = 'dark';
  });
  await page.waitForTimeout(SETTLE_MS);
  await shot(page, 'edit-desktop-dark');
  expect(errors).toEqual([]);
});

for (const url of ['/examples/scene-lmhead.html', '/examples/lm-head-nll.html', '/examples/softmax-ce.html']) {
  test(`${url} without edit has no edit interface`, async ({ page }) => {
    await open(page, url);
    await expect(page.locator('.vf-edit')).toHaveCount(0);
    await expect(page.locator('.vf-scene.vf-editing')).toHaveCount(0);
    const title = page.locator('.vf-lb.vf-ttl').first();
    const before = (await title.boundingBox())!;
    await page.mouse.move(before.x + before.width / 2, before.y + before.height / 2);
    await page.mouse.down();
    await page.mouse.move(before.x + 80, before.y + 60, { steps: 4 });
    await page.mouse.up();
    const after = (await title.boundingBox())!;
    expect(Math.abs(after.x - before.x) > 40 && Math.abs(after.y - before.y) > 40).toBe(false);
  });
}
