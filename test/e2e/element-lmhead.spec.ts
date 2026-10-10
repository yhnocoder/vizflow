import { expect, test, type Page } from '@playwright/test';

interface Win {
  vizflow: {
    page: HTMLElement & { createdSteps?: { go(i: number): void; list: unknown[] } };
  };
}

async function open(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/examples/element-lmhead.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  return errors;
}

test('the element lays out scene, steps and panels like lm-head-nll', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  const errors = await open(page);
  await expect(page.locator('vizflow-page > .vf-split > .vf-main > .vf-stage .vf-scene canvas')).toHaveCount(1);
  await expect(page.locator('vizflow-page > .vf-split > .vf-main > .vf-steps .vf-st')).toHaveCount(11);
  await expect(page.locator('vizflow-page > .vf-split > aside.vf-side .vf-loss-value')).toHaveText('0.503');
  const stage = await page.locator('vizflow-page .vf-stage').boundingBox();
  const side = await page.locator('vizflow-page .vf-side').boundingBox();
  expect(stage!.height).toBeGreaterThanOrEqual(400);
  expect(side!.x).toBeGreaterThan(stage!.x + stage!.width);
  expect(errors).toEqual([]);
});

test('createdSteps drives the step bar', async ({ page }) => {
  const errors = await open(page);
  await page.evaluate(() => (window as unknown as Win).vizflow.page.createdSteps!.go(3));
  await expect(page.locator('.vf-st[aria-current="step"]')).toHaveText('LSE');
  await expect(page.locator('.vf-caption h3')).toHaveText('log-sum-exp');
  expect(errors).toEqual([]);
});

test('removing the element disposes the scene and the panels', async ({ page }) => {
  const errors = await open(page);
  await page.evaluate(() => (window as unknown as Win).vizflow.page.remove());
  await expect(page.locator('.vf-scene')).toHaveCount(0);
  await expect(page.locator('.vf-panels')).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as Win).vizflow.page.createdSteps)).toBeUndefined();
  expect(errors).toEqual([]);
});
