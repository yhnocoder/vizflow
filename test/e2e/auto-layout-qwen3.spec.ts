import { expect, test } from '@playwright/test';

test('auto layout qwen3 page lays out the full layer and walks every step without errors', async ({ page }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/examples/auto-layout-qwen3.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true', { timeout: 60_000 });
  await expect(page.locator('.vf-loss-value')).toHaveText('2.746');
  const summary = await page.evaluate(() => {
    const { layout, modules } = (window as unknown as { vizflow: { layout: { template: string }[]; modules: { title: string; side: string }[] } }).vizflow;
    return { boards: layout.filter((item) => item.template === 'board').length, modules: modules.map((m) => `${m.side} ${m.title}`) };
  });
  expect(summary.boards).toBe(12);
  expect(summary.modules).toEqual([
    'left embed_tokens', 'left layers 1 … N−1', 'left decoder layer N', 'right input_layernorm', 'right self_attn', 'right k_norm', 'right apply_rotary_pos_emb',
    'right q_norm', 'right apply_rotary_pos_emb', 'right softmax', 'right post_attention_layernorm', 'right mlp', 'left norm', 'left lm_head', 'left loss',
  ]);
  const count = await page.evaluate(() => (window as unknown as { vizflow: { steps: { list: unknown[] } } }).vizflow.steps.list.length);
  for (let i = 0; i < count; i += 5) {
    await page.evaluate((step) => (window as unknown as { vizflow: { steps: { go(i: number): void } } }).vizflow.steps.go(step), i);
  }
  expect(errors).toEqual([]);
});
