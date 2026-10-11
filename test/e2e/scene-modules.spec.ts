import { join } from 'node:path';
import { type Page, expect, test } from '@playwright/test';
import { stageRect } from './labels';

const OVERVIEW = 0;
const FORWARD_1 = 1;
const FORWARD_2 = 2;
const FORWARD_3 = 3;
const FORWARD_5 = 5;
const SETTLE_MS = 900;

async function openScene(page: Page, query = ''): Promise<string[]> {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
    if (message.type() === 'warning' && message.text().startsWith('THREE.')) errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 860 });
  await page.goto(`/examples/scene-modules.html${query}`);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('.vf-scene canvas')).toHaveCount(1);
  await page.waitForTimeout(SETTLE_MS);
  return errors;
}

async function shot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: `test-results/scene-modules-${name}.png` });
  const dir = process.env.VIZFLOW_SHOT_DIR;
  if (dir) await page.screenshot({ path: join(dir, `modules-${name}.png`) });
}

async function goStep(page: Page, index: number, wait = SETTLE_MS): Promise<void> {
  await page.evaluate((i) => (window as unknown as { vizflow: { steps: { go(i: number): void } } }).vizflow.steps.go(i), index);
  if (wait > 0) await page.waitForTimeout(wait);
}

async function nextFrames(page: Page): Promise<void> {
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}

async function zSpan(page: Page): Promise<{ left: number; right: number; mid: number }> {
  const first = (await page.locator('.vf-lb.vf-val[data-node="z"][data-index="0,0"]').boundingBox())!;
  const last = (await page.locator('.vf-lb.vf-val[data-node="z"][data-index="0,4"]').boundingBox())!;
  return { left: first.x, right: last.x, mid: (first.x + last.x) / 2 };
}

async function moduleState(page: Page): Promise<Record<string, { shown: boolean; on: boolean }>> {
  return page.locator('.vf-lb.vf-mod').evaluateAll((els) =>
    Object.fromEntries(els.map((el) => [(el as HTMLElement).dataset.module!, { shown: !el.classList.contains('vf-off'), on: el.classList.contains('vf-on') }])),
  );
}

async function titleColors(page: Page): Promise<{ title: string; fg: string }> {
  return page.evaluate(() => {
    const scene = document.querySelector('.vf-scene')!;
    const probe = (value: string) => {
      const span = document.createElement('span');
      span.style.color = value;
      scene.appendChild(span);
      const color = getComputedStyle(span).color;
      span.remove();
      return color;
    };
    const title = document.querySelector('.vf-lb.vf-mod[data-module="lm_head"] .vf-mod-title')!;
    return { title: getComputedStyle(title).color, fg: probe('var(--fg)') };
  });
}

async function moduleBox(page: Page, title: string): Promise<{ x: number; y: number }> {
  const box = (await page.locator(`.vf-lb.vf-mod[data-module="${title}"]`).boundingBox())!;
  return { x: box.x, y: box.y };
}

test('follow moves the camera with the step and the overview returns to the same view', async ({ page }) => {
  const errors = await openScene(page);
  const overview = await moduleBox(page, 'lm_head');
  await goStep(page, FORWARD_2);
  await shot(page, 'forward-2');
  await goStep(page, OVERVIEW);
  const back = await moduleBox(page, 'lm_head');
  expect(Math.abs(back.x - overview.x)).toBeLessThan(1);
  expect(Math.abs(back.y - overview.y)).toBeLessThan(1);
  await goStep(page, FORWARD_3);
  const shown = await moduleState(page);
  expect(shown.log_softmax.shown).toBe(true);
  expect(shown['NLL Loss'].shown).toBe(true);
  await shot(page, 'follow-forward-3');
  expect(errors).toEqual([]);
});

test('without follow the framing stays the same across steps', async ({ page }) => {
  const errors = await openScene(page, '?follow=0');
  const overview = await zSpan(page);
  await goStep(page, FORWARD_2);
  const step = await zSpan(page);
  expect(Math.abs(step.left - overview.left)).toBeLessThan(1);
  expect(Math.abs(step.right - overview.right)).toBeLessThan(1);
  expect(errors).toEqual([]);
});

test('reduced motion moves the camera to the new framing at once', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const errors = await openScene(page);
  await goStep(page, FORWARD_2, 0);
  await nextFrames(page);
  const immediate = await zSpan(page);
  await page.waitForTimeout(SETTLE_MS);
  const settled = await zSpan(page);
  expect(Math.abs(immediate.left - settled.left)).toBeLessThan(1);
  expect(Math.abs(immediate.right - settled.right)).toBeLessThan(1);
  expect(errors).toEqual([]);
});

test('reset view returns to the framing of the current step', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const errors = await openScene(page);
  await goStep(page, FORWARD_2);
  const framed = await zSpan(page);
  const stage = await stageRect(page);
  await page.mouse.move(stage.x + stage.w / 2, stage.y + stage.h / 2);
  await page.mouse.down({ button: 'right' });
  await page.mouse.move(stage.x + stage.w / 2 + 120, stage.y + stage.h / 2 + 40, { steps: 6 });
  await page.mouse.up({ button: 'right' });
  await page.waitForTimeout(SETTLE_MS);
  const panned = await zSpan(page);
  expect(Math.abs(panned.mid - framed.mid)).toBeGreaterThan(40);
  await page.getByRole('button', { name: '复位视角' }).click();
  await page.waitForTimeout(SETTLE_MS);
  const reset = await zSpan(page);
  expect(Math.abs(reset.left - framed.left)).toBeLessThan(1);
  expect(Math.abs(reset.right - framed.right)).toBeLessThan(1);
  await page.mouse.move(stage.x + stage.w / 2, stage.y + stage.h / 2);
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(SETTLE_MS);
  await goStep(page, FORWARD_1);
  await goStep(page, FORWARD_2);
  const reframed = await zSpan(page);
  expect(Math.abs(reframed.left - framed.left)).toBeLessThan(1);
  expect(Math.abs(reframed.right - framed.right)).toBeLessThan(1);
  expect(errors).toEqual([]);
});

test('module labels stay visible on every step and use the foreground color', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const errors = await openScene(page, '?follow=0');
  for (const step of [FORWARD_1, FORWARD_3, FORWARD_5]) {
    await goStep(page, step);
    const state = await moduleState(page);
    expect(Object.values(state).every((s) => s.shown && !s.on)).toBe(true);
    const colors = await titleColors(page);
    expect(colors.title).toBe(colors.fg);
  }
  await shot(page, 'forward-5');
  await page.getByRole('button', { name: '切换深色' }).click();
  await goStep(page, FORWARD_1);
  const colors = await titleColors(page);
  expect(colors.title).toBe(colors.fg);
  await shot(page, 'dark-forward-1');
  expect(errors).toEqual([]);
});

test('nested and stacked brace labels sit outside the graph on their sides', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const errors = await openScene(page, '?follow=0');
  await goStep(page, FORWARD_5);
  const box = async (title: string) => (await page.locator(`.vf-lb.vf-mod[data-module="${title}"]`).boundingBox())!;
  const lmHead = await box('lm_head');
  const logits = await box('logits');
  const logSoftmax = await box('log_softmax');
  const nll = await box('NLL Loss');
  const loss = await box('ℓ');
  const h = (await page.locator('.vf-lb.vf-ttl[data-node="h"]').boundingBox())!;
  expect(lmHead.x + lmHead.width).toBeLessThan(logits.x + logits.width);
  expect(logits.x + logits.width).toBeLessThan(h.x + h.width);
  expect(nll.x).toBeGreaterThan(logSoftmax.x);
  expect(nll.x).toBeGreaterThan(loss.x);
  await shot(page, 'overview-braces');
  expect(errors).toEqual([]);
});
