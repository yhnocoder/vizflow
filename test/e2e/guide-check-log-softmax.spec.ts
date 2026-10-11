import { expect, test } from '@playwright/test';

const INITIAL = [2.0, 1.0, 0.2, -0.6, -1.4];

function nllAt(values: number[], target: number): string {
  const max = Math.max(...values);
  const lse = max + Math.log(values.reduce((s, v) => s + Math.exp(v - max), 0));
  return (lse - values[target]).toFixed(3);
}

test('log_softmax widget', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/examples/guide-check-log-softmax.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');

  const lossValue = page.locator('.vf-loss-value');
  const initialLoss = nllAt(INITIAL, 0);
  await expect(lossValue).toHaveText(initialLoss);

  const moved = [...INITIAL];
  moved[0] = -1;
  await page.getByRole('slider', { name: 'x₁' }).fill('-1');
  await expect(lossValue).toHaveText(nllAt(moved, 0));
  expect(nllAt(moved, 0)).not.toBe(initialLoss);

  const buttons = page.locator('.vf-st');
  await expect(buttons).toHaveCount(16);
  const labels = await buttons.allTextContents();
  expect(labels.filter((l) => l.trim().startsWith('∂'))).toHaveLength(7);

  await buttons.last().click();
  await expect(page.locator('.vf-st[aria-current="step"]')).toHaveText(/∂x/);
  await expect(page.locator('.vf-caption h3')).toContainText('∂ℓ/∂x');

  const before = Number(await lossValue.textContent());
  await page.getByRole('button', { name: /梯度下降一步/ }).click();
  await expect.poll(async () => Number(await lossValue.textContent())).toBeLessThan(before);

  await page.getByRole('button', { name: '重置' }).click();
  await expect(lossValue).toHaveText(initialLoss);

  expect(errors).toEqual([]);
});
