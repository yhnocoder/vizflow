import { expect, test } from '@playwright/test';

interface Win {
  vizflow: { steps: { list: unknown[]; go(i: number): void } };
}

test('qwen3 YAML page compiles in the browser and walks every step without errors', async ({ page }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error' || m.text().startsWith('vizflow dsl:')) errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/examples/qwen3/qwen3-dsl.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true', { timeout: 60_000 });
  await expect(page.locator('vizflow-page > .vf-header .vf-title')).toHaveText('Qwen3 一层：从 token 到损失');
  await expect(page.locator('vizflow-page .vf-stage .vf-scene canvas')).toHaveCount(1);
  await expect(page.locator('.vf-loss-value')).toHaveText('2.777');
  const count = await page.evaluate(() => (window as unknown as Win).vizflow.steps.list.length);
  expect(count).toBeGreaterThan(80);
  for (let i = 0; i < count; i++) {
    await page.evaluate((step) => (window as unknown as Win).vizflow.steps.go(step), i);
  }
  await page.getByRole('button', { name: /更新一步全部参数/ }).click();
  await expect(page.locator('.vf-loss-value')).not.toHaveText('2.777');
  expect(errors).toEqual([]);
});

test('a page whose src cannot be compiled shows the error in the element', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  await page.goto('/examples/index.html');
  await page.evaluate(async (element) => {
    await import(element);
    const el = document.createElement('vizflow-page');
    el.setAttribute('src', 'missing-page.yaml');
    document.body.append(el);
  }, '/src/element/index.ts');
  await expect(page.locator('vizflow-page > .vf-load-error')).toContainText('missing-page.yaml');
  expect(errors.join('\n')).toContain('missing-page.yaml');
});
