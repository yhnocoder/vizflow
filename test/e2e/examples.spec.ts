import { expect, test } from '@playwright/test';

test('example index loads the core entry', async ({ page }) => {
  await page.goto('/examples/index.html');
  await expect(page.locator('#status')).toHaveText(/^core exports: \d+$/);
});
