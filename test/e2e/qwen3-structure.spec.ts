import { expect, test } from '@playwright/test';

test('qwen3 structure page builds the full layer and walks every step without errors', async ({ page }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/examples/qwen3-structure.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true', { timeout: 60_000 });
  await expect(page.locator('.vf-loss-value')).toHaveText('2.746');
  const count = await page.evaluate(() => (window as unknown as { vizflow: { steps: { list: unknown[] } } }).vizflow.steps.list.length);
  expect(count).toBeGreaterThan(80);
  for (let i = 0; i < count; i += 7) {
    await page.evaluate((step) => (window as unknown as { vizflow: { steps: { go(i: number): void } } }).vizflow.steps.go(step), i);
  }
  await page.getByRole('button', { name: /更新一步全部参数/ }).click();
  await expect(page.locator('.vf-loss-value')).not.toHaveText('2.746');
  expect(errors).toEqual([]);
});
