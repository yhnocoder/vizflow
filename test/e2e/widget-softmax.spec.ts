import { expect, test, type Page } from '@playwright/test';

async function open(page: Page) {
  const requests: string[] = [];
  const errors: string[] = [];
  page.on('request', (request) => requests.push(request.url()));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/examples/widget-softmax.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  return { requests, errors };
}

const lossText = (page: Page) => page.locator('.vf-loss-value').textContent();
const captionText = (page: Page) => page.locator('.vf-caption-body').textContent();

test('widget page does not load three.js', async ({ page }) => {
  const { requests, errors } = await open(page);
  await expect(page.locator('.vf-card')).toHaveCount(4);
  expect(requests.filter((url) => /three/i.test(url))).toEqual([]);
  expect(errors).toEqual([]);
});

test('dragging a slider updates the loss readout and the caption together', async ({ page }) => {
  await open(page);
  await expect(page.locator('.vf-caption-body')).toContainText('0.495');
  const before = await lossText(page);
  expect(before).toBe('0.495');
  const slider = page.getByRole('slider', { name: 'x₁' });
  await slider.fill('-1');
  await expect(page.locator('.vf-loss-value')).not.toHaveText(before!);
  const after = await lossText(page);
  await expect(page.locator('.vf-caption-body')).toContainText(after!);
  expect(await captionText(page)).not.toContain('0.495');
  await expect(page.locator('.vf-sub')).toContainText('目标类 1');
});

test('a gradient step lowers the loss and draws the curve', async ({ page }) => {
  await open(page);
  await expect(page.locator('.vf-spark')).toHaveCount(0);
  const before = Number(await lossText(page));
  await page.getByRole('button', { name: /梯度下降一步（η = 1）/ }).click();
  await expect(page.locator('.vf-spark path')).toHaveCount(1);
  const after = Number(await lossText(page));
  expect(after).toBeLessThan(before);
  await page.getByRole('button', { name: '重置' }).click();
  await expect(page.locator('.vf-loss-value')).toHaveText('0.495');
  await expect(page.locator('.vf-spark')).toHaveCount(0);
});

test('choosing another target updates the table and the matrix frame', async ({ page }) => {
  await open(page);
  await page.locator('tbody tr').nth(2).getByRole('radio').check();
  await expect(page.locator('tr.vf-on .vf-tok')).toHaveText('␣sofa');
  const frame = page.locator('.vf-wmat .vf-frame.vf-rt');
  await expect(frame).toHaveCount(1);
  await expect(frame).toHaveCSS('grid-row-start', '4');
  await expect(page.locator('.vf-card', { hasText: 'Jacobian' }).locator('.vf-note')).toContainText('t=3');
});

test('arrow button and keyboard switch steps', async ({ page }) => {
  await open(page);
  const current = page.locator('.vf-st[aria-current="step"]');
  await expect(current).toHaveText('整张图');
  await expect(page.getByRole('button', { name: '上一步' })).toBeDisabled();
  await page.getByRole('button', { name: '下一步' }).click();
  await expect(current).toHaveText('x');
  await expect(page.locator('.vf-caption h3')).toHaveText('logits');
  await page.keyboard.press('ArrowRight');
  await expect(current).toHaveText('p');
  await page.keyboard.press('ArrowLeft');
  await expect(current).toHaveText('x');
  await page.getByRole('button', { name: '重置' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(current).toHaveText('x');
  await page.getByRole('slider', { name: 'x₂' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(current).toHaveText('x');
  await page.locator('.vf-st').last().click();
  await expect(page.getByRole('button', { name: '下一步' })).toBeDisabled();
});

test('no horizontal overflow at 375px', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await open(page);
  const overflow = await page.evaluate(() => {
    const root = document.documentElement;
    return { scroll: root.scrollWidth, client: root.clientWidth };
  });
  expect(overflow.scroll).toBeLessThanOrEqual(overflow.client);
  await page.screenshot({ path: 'test-results/widget-softmax-375.png', fullPage: true });
});

for (const scheme of ['light', 'dark'] as const) {
  test(`screenshot in ${scheme} mode`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await page.setViewportSize({ width: 1280, height: 900 });
    const { errors } = await open(page);
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: `test-results/widget-softmax-${scheme}.png`, fullPage: true });
    expect(errors).toEqual([]);
  });
}
