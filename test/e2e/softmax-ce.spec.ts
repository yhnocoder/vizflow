import { type Page, expect, test } from '@playwright/test';
import { expectLabelsApartAndInside, waitForStableLabels, waitForStep } from './labels';

type Mode = 'single' | 'batch';

interface PageNode {
  name: string;
  value: { data: Float64Array };
  grad: { data: Float64Array } | null;
}

interface PageMode {
  g: { nodes: PageNode[]; loss(): PageNode };
  steps: { list: unknown[]; current: number; focus: number; go(i: number): void; setFocus(b: number): void };
}

declare global {
  interface Window {
    vizflow: { modes: Record<Mode, PageMode>; current(): Mode };
  }
}

const SCREENSHOT_DIR = process.env.VIZFLOW_SCREENSHOTS ?? 'test-results/softmax-ce';
const DEFAULT_LOGITS = [
  [2.0, 1.0, 0.2, -0.6, -1.4],
  [-0.5, 1.8, 0.3, 0.9, -1.0],
  [0.4, -0.2, 1.5, 0.1, 0.6],
];
const DEFAULT_TARGETS = [0, 1, 2];
const FORWARD_STEP = 4;
const BACKWARD_STEP = 9;

function softmax(row: number[]): number[] {
  const u = row.map(Math.exp);
  const s = u.reduce((a, b) => a + b, 0);
  return u.map((v) => v / s);
}

const crossEntropy = (row: number[], t: number) => -Math.log(softmax(row)[t]);
const SINGLE_LOSS = crossEntropy(DEFAULT_LOGITS[0], DEFAULT_TARGETS[0]);
const BATCH_LOSS = DEFAULT_LOGITS.reduce((s, row, b) => s + crossEntropy(row, DEFAULT_TARGETS[b]), 0) / DEFAULT_LOGITS.length;

async function open(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' || (message.type() === 'warning' && message.text().startsWith('THREE.'))) errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/examples/softmax-ce.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('.vf-scene canvas')).toHaveCount(1);
  await expect(page.locator('.vf-nogl')).toHaveCount(0);
  await page.evaluate(() => document.fonts.ready);
  await waitForStableLabels(page);
  return errors;
}

async function switchTo(page: Page, mode: Mode): Promise<void> {
  await page.getByRole('tab', { name: mode === 'single' ? /单样本/ : /批量/ }).click();
  await expect.poll(() => page.evaluate(() => window.vizflow.current())).toBe(mode);
  await expect(page.locator('.vf-scene canvas')).toHaveCount(1);
  await waitForStableLabels(page);
}

async function goStep(page: Page, mode: Mode, index: number): Promise<void> {
  await page.evaluate(([m, i]) => window.vizflow.modes[m as Mode].steps.go(i as number), [mode, index] as const);
  await waitForStep(page, index);
}

const readLoss = (page: Page, mode: Mode) => page.evaluate((m) => window.vizflow.modes[m as Mode].g.loss().value.data[0], mode);

const snapshot = (page: Page, mode: Mode) =>
  page.evaluate(
    (m) =>
      window.vizflow.modes[m as Mode].g.nodes.map((n) => ({
        name: n.name,
        value: Array.from(n.value.data),
        grad: n.grad === null ? null : Array.from(n.grad.data),
      })),
    mode,
  );

for (const scheme of ['light', 'dark'] as const) {
  for (const width of [1280, 375]) {
    if (scheme === 'dark' && width !== 1280) continue;
    test(`steps through both modes without console errors (${scheme}, ${width}px)`, async ({ page }) => {
      test.setTimeout(90_000);
      await page.emulateMedia({ colorScheme: scheme, reducedMotion: 'reduce' });
      await page.setViewportSize({ width, height: width === 375 ? 812 : 900 });
      const errors = await open(page);
      const tag = `${width === 375 ? '375' : 'desktop'}${scheme === 'dark' ? '-dark' : ''}`;
      for (const mode of ['single', 'batch'] as const) {
        if (mode === 'batch') await switchTo(page, mode);
        const count = await page.evaluate((m) => window.vizflow.modes[m as Mode].steps.list.length, mode);
        expect(count).toBe(11);
        for (let i = 0; i < count; i++) {
          await goStep(page, mode, i);
          await expect(page.locator('.vf-st[aria-current="step"]')).toHaveCount(1);
          await expectLabelsApartAndInside(page, `${mode} step ${i} at ${width}px`);
          if (i === 0 || i === FORWARD_STEP || i === BACKWARD_STEP) {
            const name = i === 0 ? 'overview' : i === FORWARD_STEP ? 'forward' : 'backward';
            await page.screenshot({ path: `${SCREENSHOT_DIR}/${mode}-${name}-${tag}.png`, fullPage: true });
          }
        }
        await goStep(page, mode, 0);
      }
      if (width === 375) {
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
        expect(overflow).toBeLessThanOrEqual(0);
      }
      expect(errors).toEqual([]);
    });
  }
}

test('default inputs reproduce the reference losses', async ({ page }) => {
  await open(page);
  expect(await readLoss(page, 'single')).toBeCloseTo(SINGLE_LOSS, 12);
  await expect(page.locator('.vf-loss-value')).toHaveText(SINGLE_LOSS.toFixed(3));
  await switchTo(page, 'batch');
  expect(await readLoss(page, 'batch')).toBeCloseTo(BATCH_LOSS, 12);
  await expect(page.locator('.vf-loss-value')).toHaveText(BATCH_LOSS.toFixed(3));
});

test('batch gradient of X equals (P - Y) / B', async ({ page }) => {
  await open(page);
  await switchTo(page, 'batch');
  const { P, gX, t } = await page.evaluate(() => {
    const g = window.vizflow.modes.batch.g;
    const byName = (name: string) => g.nodes.find((n) => n.name === name)!;
    return { P: Array.from(byName('P').value.data), gX: Array.from(byName('X').grad!.data), t: Array.from(byName('t').value.data) };
  });
  const B = t.length;
  const V = P.length / B;
  expect(gX.length).toBe(P.length);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < V; j++) {
      const expected = (P[b * V + j] - (j === t[b] ? 1 : 0)) / B;
      expect(gX[b * V + j]).toBeCloseTo(expected, 12);
    }
  }
  expect(t).toEqual(DEFAULT_TARGETS);
});

test('switching samples changes highlight and caption but no node values', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await open(page);
  await switchTo(page, 'batch');
  await goStep(page, 'batch', BACKWARD_STEP);
  const before = await snapshot(page, 'batch');
  const caption = await page.locator('.vf-caption-body').textContent();
  const jacobian = await page.locator('.vf-wmat').textContent();
  await expect(page.getByRole('button', { name: '样本 1' })).toHaveAttribute('aria-current', 'true');
  await page.getByRole('button', { name: '样本 2' }).click();
  expect(await page.evaluate(() => window.vizflow.modes.batch.steps.focus)).toBe(1);
  await expect(page.getByRole('button', { name: '样本 2' })).toHaveAttribute('aria-current', 'true');
  await expect(page.locator('.vf-caption-body')).not.toHaveText(caption!);
  await expect(page.locator('.vf-caption-body')).toContainText('样本 2');
  await expect(page.locator('.vf-wmat')).not.toHaveText(jacobian!);
  await expect(page.locator('.vf-wmat .vf-frame.vf-rt')).toHaveCSS('grid-row-start', '3');
  await expect(page.locator('tr.vf-on .vf-tok-id')).toHaveText('2');
  expect(await snapshot(page, 'batch')).toEqual(before);
  await page.getByRole('button', { name: '样本 3' }).click();
  expect(await snapshot(page, 'batch')).toEqual(before);
});

test('slider, target choice and a gradient step change the loss as expected', async ({ page }) => {
  await open(page);
  const x = [...DEFAULT_LOGITS[0]];
  x[0] = -1;
  await page.getByRole('slider', { name: 'x₁' }).fill('-1');
  expect(await readLoss(page, 'single')).toBeCloseTo(crossEntropy(x, 0), 12);
  await expect(page.locator('.vf-loss-value')).toHaveText(crossEntropy(x, 0).toFixed(3));
  await page.locator('tbody tr').nth(1).getByRole('radio').check();
  expect(await readLoss(page, 'single')).toBeCloseTo(crossEntropy(x, 1), 12);
  await expect(page.locator('.vf-sub')).toContainText('目标类 2');
  await expect(page.locator('.vf-wmat .vf-frame.vf-rt')).toHaveCSS('grid-row-start', '3');
  const p = softmax(x);
  const stepped = x.map((v, j) => v - (p[j] - (j === 1 ? 1 : 0)));
  await page.getByRole('button', { name: /^梯度下降一步/ }).click();
  expect(await readLoss(page, 'single')).toBeCloseTo(crossEntropy(stepped, 1), 12);
  expect(crossEntropy(stepped, 1)).toBeLessThan(crossEntropy(x, 1));
  await expect(page.locator('.vf-spark path')).toHaveCount(1);
  await page.getByRole('button', { name: '重置' }).click();
  expect(await readLoss(page, 'single')).toBeCloseTo(SINGLE_LOSS, 12);

  await switchTo(page, 'batch');
  await page.getByRole('button', { name: '样本 3' }).click();
  await page.locator('tbody tr').nth(0).getByRole('radio').check();
  const targets = [0, 1, 0];
  const batchLoss = (rows: number[][]) => rows.reduce((s, row, b) => s + crossEntropy(row, targets[b]), 0) / rows.length;
  expect(await readLoss(page, 'batch')).toBeCloseTo(batchLoss(DEFAULT_LOGITS), 12);
  await page.getByRole('slider', { name: 'X₂' }).fill('2');
  const rows = DEFAULT_LOGITS.map((row) => [...row]);
  rows[2][1] = 2;
  expect(await readLoss(page, 'batch')).toBeCloseTo(batchLoss(rows), 12);
  const after = rows.map((row, b) => {
    const pb = softmax(row);
    return row.map((v, j) => v - (pb[j] - (j === targets[b] ? 1 : 0)) / rows.length);
  });
  await page.getByRole('button', { name: /^梯度下降一步/ }).click();
  expect(await readLoss(page, 'batch')).toBeCloseTo(batchLoss(after), 12);
  expect(batchLoss(after)).toBeLessThan(batchLoss(rows));
});

test('each mode keeps its own state across tab switches', async ({ page }) => {
  await open(page);
  await goStep(page, 'single', 3);
  await switchTo(page, 'batch');
  await expect(page.locator('.vf-st[aria-current="step"]')).toHaveText('总览');
  await switchTo(page, 'single');
  await expect(page.locator('.vf-st[aria-current="step"]')).toHaveText('S');
  await expect(page.locator('.vf-scene')).toHaveCount(1);
  await expect(page.locator('.vf-stepper')).toHaveCount(1);
});
