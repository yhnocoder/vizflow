import { chromium } from '@playwright/test';
const [, , base, out] = process.argv;
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, colorScheme: 'light' });
await page.goto(base + '/examples/qwen3-dsl.html');
await page.waitForSelector('body[data-ready=true]', { timeout: 60000 });
const b = await (await page.$('.vf-scene canvas')).boundingBox();
const back = await page.evaluate(() => window.vizflow.steps.list.findIndex((s) => s.dir === 'backward'));
await page.evaluate((s) => window.vizflow.steps.go(s), back + 6);
await page.waitForTimeout(2500);
for (let y = 0.5; y < 0.62; y += 0.004) {
  await page.mouse.move(b.x + b.width * 0.215, b.y + b.height * y);
  const t = await page.$eval('.vf-pop', (p) => (p.hidden ? '' : p.textContent));
  if (t.includes('局部导数')) { console.log(t.slice(-40)); break; }
}
await page.waitForTimeout(300);
await page.screenshot({ path: out, clip: b });
await browser.close();
