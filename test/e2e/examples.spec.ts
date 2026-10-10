import { expect, test } from '@playwright/test';

test('example index loads the core entry', async ({ page }) => {
  await page.goto('/examples/index.html');
  await expect(page.locator('#status')).toHaveText(/^core exports: \d+$/);
});

test('example index links to every example page', async ({ page, request }) => {
  await page.goto('/examples/index.html');
  const links = await page.locator('a').evaluateAll((anchors) => anchors.map((anchor) => (anchor as HTMLAnchorElement).href));
  const pages = links.map((link) => new URL(link).pathname).sort();
  expect(pages).toEqual(
    ['edit-lmhead', 'guide-check-log-softmax', 'lm-head-nll', 'scene-lmhead', 'softmax-ce', 'theme', 'widget-softmax'].map((name) => `/examples/${name}.html`),
  );
  for (const link of links) expect((await request.get(link)).status(), link).toBe(200);
});
