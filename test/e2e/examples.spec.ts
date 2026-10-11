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
    ['auto-layout-qwen3', 'edit-lmhead', 'element-lmhead', 'guide-check-log-softmax', 'lm-head-nll', 'qwen3-structure', 'scene-lmhead', 'scene-modules', 'softmax-ce', 'theme', 'widget-softmax'].map((name) => `/examples/${name}.html`),
  );
  for (const link of links) expect((await request.get(link)).status(), link).toBe(200);
});
