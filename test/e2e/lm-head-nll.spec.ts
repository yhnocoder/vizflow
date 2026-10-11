import { join } from 'node:path';
import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { expectLabelsApartAndInside, waitForStableLabels, waitForStep } from './labels';

const DEFAULT_LOSS = 0.5027410756235495;
const FORWARD_2 = 2;
const BACKWARD_3 = 8;
const STEP_NAMES = ['overview', 'forward-1', 'forward-2', 'forward-3', 'forward-4', 'forward-5', 'backward-1', 'backward-2', 'backward-3', 'backward-4', 'backward-5'];

interface Win {
  g: { loss(): { value: { data: Float64Array } } };
  steps: { go(i: number): void; list: unknown[] };
  nodes: { t: { index(i: number): number } };
}

async function open(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' || (message.type() === 'warning' && message.text().startsWith('THREE.'))) errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/examples/lm-head-nll.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('.vf-scene canvas')).toHaveCount(1);
  await page.evaluate(() => document.fonts.ready);
  await waitForStableLabels(page);
  return errors;
}

async function goStep(page: Page, index: number): Promise<void> {
  await page.evaluate((i) => (window as unknown as { vizflow: Win }).vizflow.steps.go(i), index);
  await waitForStep(page, index);
}

const readLoss = (page: Page) => page.evaluate(() => (window as unknown as { vizflow: Win }).vizflow.g.loss().value.data[0]);
const readTarget = (page: Page) => page.evaluate(() => (window as unknown as { vizflow: Win }).vizflow.nodes.t.index(0));
const lossText = (page: Page) => page.locator('.vf-loss-value');
const caption = (page: Page) => page.locator('.vf-caption-body');

async function shot(page: Page, info: TestInfo, name: string): Promise<void> {
  await page.screenshot({ path: info.outputPath(`${name}.png`), fullPage: true });
  const dir = process.env.VIZFLOW_SHOT_DIR;
  if (dir) await page.screenshot({ path: join(dir, `${name}.png`), fullPage: true });
}

test('every step renders without console errors', async ({ page }, info) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1280, height: 900 });
  const errors = await open(page);
  const count = await page.evaluate(() => (window as unknown as { vizflow: Win }).vizflow.steps.list.length);
  expect(count).toBe(STEP_NAMES.length);
  const labels = await page.locator('.vf-st').evaluateAll((els) =>
    els.map((el) => {
      const copy = el.cloneNode(true) as Element;
      for (const k of copy.querySelectorAll('.katex')) k.replaceWith(`$${k.querySelector('annotation')?.textContent ?? ''}$`);
      return copy.textContent;
    }),
  );
  expect(labels).toEqual(['整张图', 'h, Wᵀ', 'z = hWᵀ', 'LSE', 'log p', 'ℓ', '$\\partial \\log\\mathbf{p}$', '$\\partial \\mathrm{LSE}$', '$\\partial \\mathbf{z}$', '∂Wᵀ', '$\\partial \\mathbf{h}$']);
  for (let i = 0; i < count; i++) {
    await goStep(page, i);
    await expect(page.locator('.vf-st[aria-current="step"]')).toHaveCount(1);
    await expect(page.locator('.vf-caption .vf-tex-error')).toHaveCount(0);
    await expectLabelsApartAndInside(page, `step ${i} at 1280px`);
    await page.screenshot({ path: info.outputPath(`step-${i}-${STEP_NAMES[i]}.png`) });
  }
  expect(errors).toEqual([]);
});

test('default input gives the reference loss', async ({ page }) => {
  await open(page);
  expect(Math.abs((await readLoss(page)) - DEFAULT_LOSS)).toBeLessThan(1e-12);
  await expect(lossText(page)).toHaveText('0.503');
  await expect(caption(page)).toContainText('0.503');
  await expect(page.locator('.vf-sub')).toContainText('log p = −0.503');
  await goStep(page, FORWARD_2);
  await expect(caption(page)).toContainText('2.37');
});

test('dragging an h slider changes the loss and the caption', async ({ page }) => {
  await open(page);
  await goStep(page, FORWARD_2);
  const before = await caption(page).textContent();
  await page.getByRole('slider', { name: 'h₁' }).fill('2');
  await expect(lossText(page)).not.toHaveText('0.503');
  await expect(caption(page)).not.toHaveText(before!);
  await expect(caption(page)).toContainText('3.45');
  expect(await readLoss(page)).not.toBeCloseTo(DEFAULT_LOSS, 6);
});

test('updating W lowers the loss and reset restores it', async ({ page }) => {
  await open(page);
  const before = await readLoss(page);
  await page.getByRole('button', { name: /^更新一步 W/ }).click();
  const after = await readLoss(page);
  expect(after).toBeLessThan(before);
  await expect(page.locator('.vf-spark path')).toHaveCount(1);
  await page.getByRole('button', { name: '重置' }).click();
  await expect(lossText(page)).toHaveText('0.503');
});

test('clicking a logits bar changes the target', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await open(page);
  expect(await readTarget(page)).toBe(0);
  const label = page.locator('.vf-lb.vf-val[data-node="z"][data-index="0,2"]');
  const box = (await label.boundingBox())!;
  let found = false;
  for (let dy = 4; dy <= 60 && !found; dy += 4) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height + dy);
    found = await page.locator('.vf-tip', { hasText: '点击设为目标 token' }).isVisible();
  }
  expect(found).toBe(true);
  await page.mouse.down();
  await page.mouse.up();
  expect(await readTarget(page)).toBe(2);
  await expect(page.locator('tr.vf-on .vf-tok')).toHaveText('␣sofa');
  await expect(lossText(page)).not.toHaveText('0.503');
});

for (const step of [0, FORWARD_2, BACKWARD_3]) {
  test(`review screenshots for step ${step}`, async ({ page }, info) => {
    const name = STEP_NAMES[step];
    await page.setViewportSize({ width: 1280, height: 900 });
    const errors = await open(page);
    await goStep(page, step);
    await expectLabelsApartAndInside(page, `step ${step} at 1280px`);
    await shot(page, info, `${name}-1280`);

    await page.setViewportSize({ width: 375, height: 812 });
    await waitForStableLabels(page);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    await expectLabelsApartAndInside(page, `step ${step} at 375px`);
    const formulas = await page.locator('.vf-caption .katex-display').evaluateAll((els) =>
      els.map((el) => ({ overflow: getComputedStyle(el).overflowX, right: el.getBoundingClientRect().right, card: el.closest('.vf-stepper')!.getBoundingClientRect().right })),
    );
    for (const f of formulas) {
      expect(f.overflow).toBe('auto');
      expect(f.right).toBeLessThanOrEqual(f.card);
    }
    await shot(page, info, `${name}-375`);

    await page.setViewportSize({ width: 1280, height: 900 });
    await page.emulateMedia({ colorScheme: 'dark' });
    await waitForStableLabels(page);
    await shot(page, info, `${name}-dark`);
    expect(errors).toEqual([]);
  });
}
