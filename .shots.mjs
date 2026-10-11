import { chromium } from '@playwright/test';
const [, , base, dir] = process.argv;
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, colorScheme: 'light' });
page.on('console', (m) => { if (m.type() === 'error') console.log('console', m.text()); });
await page.goto(base + '/examples/qwen3-dsl.html');
await page.waitForSelector('body[data-ready=true]', { timeout: 60000 });
const canvas = await page.$('.vf-scene canvas');
const b = await canvas.boundingBox();
const go = async (s) => { await page.evaluate((s) => window.vizflow.steps.go(s), s); await page.mouse.move(2, 2); await page.waitForTimeout(2500); };
const scan = async (sel, fx, fy) => {
  const cx = b.x + b.width * fx, cy = b.y + b.height * fy;
  for (let r = 0; r < 400; r += 5) for (let a = 0; a < 16; a++) {
    await page.mouse.move(cx + r * Math.cos((a * Math.PI) / 8), cy + r * Math.sin((a * Math.PI) / 8));
    if (await page.$eval(sel, (t) => !t.hidden)) return true;
  }
  return false;
};
const last = (await page.evaluate(() => window.vizflow.steps.list.length)) - 1;
await go(0);
await page.screenshot({ path: `${dir}/overview.png`, clip: b });
const back = await page.evaluate(() => window.vizflow.steps.list.findIndex((s) => s.dir === 'backward'));
await go(back + 6);
await page.screenshot({ path: `${dir}/backward.png`, clip: b });
if (process.argv[4] === 'full') {
  if (await scan('.vf-pop', 0.5, 0.45)) { await page.waitForTimeout(300); await page.screenshot({ path: `${dir}/edgecard.png`, clip: b }); }
  await go(8);
  if (await scan('.vf-tip', 0.2, 0.6)) {
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${dir}/hover.png`, clip: b });
    await page.addStyleTag({ content: '.vf-tip{visibility:hidden !important}' });
    await page.waitForTimeout(200);
    await page.screenshot({ path: `${dir}/hover-notip.png`, clip: b });
    await page.addStyleTag({ content: '.vf-tip{visibility:visible !important}' });
  }
  await page.mouse.move(2, 2);
  const labels = await page.$$('.vf-labels > *');
  for (const l of labels) {
    const t = await l.textContent();
    const vis = await l.isVisible();
    if (vis && /^n1/.test(t.trim())) { await l.hover(); break; }
  }
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${dir}/nodecard.png`, clip: b });
  await go(12);
  const bars = await page.$$('.vf-tb');
  await bars[5].hover();
  await page.waitForTimeout(400);
  const st = await (await page.$('.vf-stepper')).boundingBox();
  await page.screenshot({ path: `${dir}/stepper.png`, clip: { x: st.x - 10, y: st.y - 220, width: st.width + 20, height: 340 } });
}
console.log('done', last);
await browser.close();
