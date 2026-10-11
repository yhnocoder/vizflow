import { type Page, expect } from '@playwright/test';

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export const LABEL_SAFE_MARGIN = 12;
const TOLERANCE = 0.5;

export const intersects = (a: Rect, b: Rect) => a.x < b.x + b.w - TOLERANCE && b.x < a.x + a.w - TOLERANCE && a.y < b.y + b.h - TOLERANCE && b.y < a.y + a.h - TOLERANCE;

export async function stageRect(page: Page): Promise<Rect> {
  return page.locator('.vf-scene').evaluate((el) => {
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
  });
}

export async function labelRects(page: Page, selector = '.vf-lb'): Promise<(Rect & { text: string })[]> {
  return page.locator(selector).evaluateAll((elements) =>
    elements
      .filter((el) => !el.classList.contains('vf-off'))
      .map((el) => {
        const r = el.getBoundingClientRect();
        let left = r.left;
        let right = r.right;
        for (const inner of el.querySelectorAll('*')) {
          const b = inner.getBoundingClientRect();
          if (b.width === 0) continue;
          left = Math.min(left, b.left);
          right = Math.max(right, b.right);
        }
        return { x: left, y: r.top, w: right - left, h: r.height, text: el.textContent ?? '' };
      }),
  );
}

export async function waitForStableLabels(page: Page): Promise<void> {
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  let previous = '';
  await expect
    .poll(
      async () => {
        const current = JSON.stringify(await labelRects(page));
        const stable = current === previous;
        previous = current;
        return stable;
      },
      { intervals: [100] },
    )
    .toBe(true);
}

export async function waitForStep(page: Page, index: number): Promise<void> {
  await expect(page.locator('.vf-st').nth(index)).toHaveAttribute('aria-current', 'step');
  await waitForStableLabels(page);
}

export async function expectLabelsApartAndInside(page: Page, context: string): Promise<void> {
  const stage = await stageRect(page);
  const rects = await labelRects(page);
  expect(rects.length, context).toBeGreaterThan(0);
  const m = LABEL_SAFE_MARGIN - TOLERANCE;
  for (const r of rects) {
    const where = `${context}: ${JSON.stringify(r)} in ${JSON.stringify(stage)}`;
    expect(r.x, where).toBeGreaterThanOrEqual(stage.x + m);
    expect(r.y, where).toBeGreaterThanOrEqual(stage.y + m);
    expect(r.x + r.w, where).toBeLessThanOrEqual(stage.x + stage.w - m);
    expect(r.y + r.h, where).toBeLessThanOrEqual(stage.y + stage.h - m);
  }
  for (let i = 0; i < rects.length; i++) {
    for (let j = i + 1; j < rects.length; j++) {
      expect(intersects(rects[i], rects[j]), `${context}: ${JSON.stringify(rects[i])} vs ${JSON.stringify(rects[j])}`).toBe(false);
    }
  }
}
