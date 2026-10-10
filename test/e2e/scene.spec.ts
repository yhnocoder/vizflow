import { join } from 'node:path';
import { type Page, expect, test } from '@playwright/test';
import { intersects, labelRects, stageRect } from './labels';

const FORWARD_2 = 2;
const BACKWARD_2 = 7;
const BACKWARD_3 = 8;
const SETTLE_MS = 400;

async function openScene(page: Page, query = ''): Promise<string[]> {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.location().url.endsWith('/favicon.ico')) errors.push(message.text());
    if (message.type() === 'warning' && message.text().startsWith('THREE.')) errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`/examples/scene-lmhead.html${query}`);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('.vf-scene canvas')).toHaveCount(1);
  await expect(page.locator('.vf-nogl')).toHaveCount(0);
  await page.waitForTimeout(SETTLE_MS);
  return errors;
}

async function shot(page: Page, file: string, name: string): Promise<void> {
  await page.screenshot({ path: file });
  const dir = process.env.VIZFLOW_SHOT_DIR;
  if (dir) await page.screenshot({ path: join(dir, `${name}.png`) });
}

async function goStep(page: Page, index: number): Promise<void> {
  await page.evaluate((i) => (window as unknown as { vizflow: { steps: { go(i: number): void } } }).vizflow.steps.go(i), index);
  await page.waitForTimeout(SETTLE_MS);
}

test('renders the lm_head scene and screenshots forward step 2 and backward step 3', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 860 });
  const errors = await openScene(page);
  await goStep(page, FORWARD_2);
  await shot(page, 'test-results/scene-forward-2.png', 'desktop-forward-2');
  await goStep(page, BACKWARD_3);
  await shot(page, 'test-results/scene-backward-3.png', 'desktop-backward-3');
  expect(errors).toEqual([]);
  expect(await page.locator('.vf-lb.vf-val:not(.vf-off)').count()).toBeGreaterThan(0);
});

for (const gap of [0.7, 1.2]) {
  test(`board gap ${gap} keeps every block inside the canvas`, async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 860 });
    const errors = await openScene(page, gap === 0.7 ? '' : `?gap=${gap}`);
    const stage = await stageRect(page);
    const rects = await labelRects(page, '.vf-lb');
    expect(rects.length).toBeGreaterThan(10);
    for (const r of rects) {
      expect(r.x, JSON.stringify(r)).toBeGreaterThanOrEqual(stage.x);
      expect(r.y, JSON.stringify(r)).toBeGreaterThanOrEqual(stage.y);
      expect(r.x + r.w, JSON.stringify(r)).toBeLessThanOrEqual(stage.x + stage.w);
      expect(r.y + r.h, JSON.stringify(r)).toBeLessThanOrEqual(stage.y + stage.h);
    }
    expect(errors).toEqual([]);
  });
}

test('hovering a z → LSE edge in a backward step shows the edge card with explain values', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 860 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await openScene(page);
  await goStep(page, BACKWARD_2);
  const lse = page.locator('.vf-lb.vf-node[data-node="LSE"]');
  const box = (await lse.boundingBox())!;
  const pop = page.locator('.vf-pop');
  let edgeId: string | null = null;
  for (let dx = -36; dx <= 0 && edgeId === null; dx += 4) {
    for (let dy = 8; dy <= 220 && edgeId === null; dy += 4) {
      await page.mouse.move(box.x + box.width / 2 + dx, box.y + box.height + dy);
      const id = await pop.evaluate((el) => ((el as HTMLElement).hidden ? null : el.querySelector('[data-edge-id]')?.getAttribute('data-edge-id') ?? null));
      if (id !== null && id.startsWith('LSE:0:')) edgeId = id;
    }
  }
  expect(edgeId).not.toBeNull();
  const shown = await pop.locator('.vf-eq').evaluateAll((els) => els.map((el) => (el as HTMLElement).dataset.value));
  const expected = await page.evaluate((id) => {
    const w = (window as unknown as {
      vizflow: {
        explainer: { edges(): { id: string }[]; local(e: unknown): number; upstream(e: unknown): number; contribution(e: unknown): number };
        formatNumber(v: number): string;
      };
    }).vizflow;
    const e = w.explainer.edges().find((edge) => edge.id === id)!;
    return [w.formatNumber(w.explainer.local(e)), w.formatNumber(w.explainer.upstream(e)), w.formatNumber(w.explainer.contribution(e))];
  }, edgeId);
  expect(shown).toEqual(expected);
  expect(await pop.locator('.vf-lab').allTextContents()).toEqual(['局部导数', '上游梯度', '沿这条边传回']);
  await shot(page, 'test-results/scene-edge-card.png', 'card-edge-z-to-LSE');
});

test('clicking another logits bar changes the target and the loss', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 860 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await openScene(page);
  const readLoss = () => page.evaluate(() => (window as unknown as { vizflow: { g: { loss(): { value: { data: Float64Array } } } } }).vizflow.g.loss().value.data[0]);
  const readTarget = () => page.evaluate(() => (window as unknown as { vizflow: { nodes: { t: { index(i: number): number } } } }).vizflow.nodes.t.index(0));
  const before = await readLoss();
  expect(await readTarget()).toBe(0);
  const label = page.locator('.vf-lb.vf-val[data-node="z"][data-index="0,1"]');
  const box = (await label.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height + 10);
  await expect(page.locator('.vf-tip')).toBeVisible();
  await expect(page.locator('.vf-tip')).toContainText('点击设为目标 token');
  await shot(page, 'test-results/scene-tip.png', 'tooltip-hover-z2');
  await page.mouse.down();
  await page.mouse.up();
  expect(await readTarget()).toBe(1);
  expect(await readLoss()).not.toBe(before);
});

test('hovering a block title shows the node card', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 860 });
  await openScene(page);
  await page.locator('.vf-lb.vf-ttl[data-node="z"]').hover();
  const pop = page.locator('.vf-pop');
  await expect(pop).toBeVisible();
  await expect(pop.locator('.vf-pop-title')).toHaveText('z');
  await expect(pop.locator('dt').first()).toHaveText('shape');
  await shot(page, 'test-results/scene-node-card.png', 'card-node-z');
});

test('movable labels do not overlap at 375px width', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 700 });
  const errors = await openScene(page);
  for (const step of [0, 5, BACKWARD_3]) {
    await goStep(page, step);
    const rects = await labelRects(page, '.vf-lb.vf-ttl, .vf-lb.vf-node');
    expect(rects.length).toBeGreaterThan(1);
    for (let i = 0; i < rects.length; i++) {
      for (let j = i + 1; j < rects.length; j++) {
        expect(intersects(rects[i], rects[j]), `step ${step}: ${JSON.stringify(rects[i])} vs ${JSON.stringify(rects[j])}`).toBe(false);
      }
    }
  }
  await shot(page, 'test-results/scene-375.png', 'mobile-375-backward-3');
  expect(errors).toEqual([]);
});

for (const [width, height] of [[1280, 860], [375, 700]]) {
  test(`no two visible labels overlap in the overview at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    const errors = await openScene(page);
    await goStep(page, 0);
    const rects = await labelRects(page, '.vf-lb');
    expect(rects.length).toBeGreaterThan(10);
    for (let i = 0; i < rects.length; i++) {
      for (let j = i + 1; j < rects.length; j++) {
        expect(intersects(rects[i], rects[j]), `${JSON.stringify(rects[i])} vs ${JSON.stringify(rects[j])}`).toBe(false);
      }
    }
    expect(errors).toEqual([]);
  });
}

test('mounting and disposing the scene twenty times still renders', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 860 });
  const errors = await openScene(page);
  for (let i = 0; i < 20; i++) {
    await page.evaluate(() => (window as unknown as { vizflow: { remount(): void } }).vizflow.remount());
  }
  await page.waitForTimeout(SETTLE_MS);
  await expect(page.locator('.vf-scene canvas')).toHaveCount(1);
  await expect(page.locator('.vf-nogl')).toHaveCount(0);
  const painted = await page.locator('.vf-scene canvas').evaluate((canvas) => {
    const gl = (canvas as HTMLCanvasElement).getContext('webgl2') ?? (canvas as HTMLCanvasElement).getContext('webgl');
    return gl !== null && !gl.isContextLost();
  });
  expect(painted).toBe(true);
  expect(await page.locator('.vf-lb.vf-val:not(.vf-off)').count()).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});
