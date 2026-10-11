import { chromium } from '@playwright/test';
const [, , url, out, step, fx, fy] = process.argv;
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, colorScheme: 'light' });
page.on('console', (m) => { if (m.type() === 'error') console.log('console', m.text()); });
await page.goto(url);
await page.waitForSelector('body[data-ready=true]', { timeout: 60000 });
await page.evaluate((s) => window.vizflow.steps.go(Number(s)), step);
await page.waitForTimeout(2500);
const canvas = await page.$('.vf-scene canvas');
const b = await canvas.boundingBox();
let found = false;
const cx = b.x + b.width * Number(fx), cy = b.y + b.height * Number(fy);
outer: for (let r = 0; r < 400; r += 6) {
  for (let a = 0; a < 16; a++) {
    const x = cx + r * Math.cos((a * Math.PI) / 8), y = cy + r * Math.sin((a * Math.PI) / 8);
    await page.mouse.move(x, y);
    const shown = await page.$eval('.vf-tip', (t) => !t.hidden);
    if (shown) { found = true; console.log('hit', x, y, await page.$eval('.vf-tip', (t) => t.textContent)); break outer; }
  }
}
await page.waitForTimeout(500);
await page.screenshot({ path: out, clip: b });
await page.addStyleTag({ content: '.vf-tip{visibility:hidden !important}' });
await page.waitForTimeout(200);
await page.screenshot({ path: out.replace('.png', '-notip.png'), clip: b });
console.log('found', found);
await browser.close();
