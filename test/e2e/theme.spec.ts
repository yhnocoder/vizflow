import { expect, test } from '@playwright/test';

const faces = [
  '400 16px "Source Serif 4"',
  '500 16px "Source Serif 4"',
  'italic 400 16px "Source Serif 4"',
  '400 16px "Geist"',
  '500 16px "Geist"',
  '600 16px "Geist"',
  '400 16px "Geist Mono"',
  '500 16px "Geist Mono"',
];

const tokens = {
  light: { bg: '#f5f4ed', surface: '#faf9f5', fg: '#141413', neg: '#6a9bcc' },
  dark: { bg: '#262624', surface: '#30302e', fg: '#faf9f5', neg: '#7eaad6' },
};

for (const scheme of ['light', 'dark'] as const) {
  test(`theme page renders with local fonts in ${scheme} mode`, async ({ page }) => {
    const requests: string[] = [];
    page.on('request', (request) => requests.push(request.url()));
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto('/examples/theme.html');
    await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');

    const loaded = await page.evaluate(async (fontList) => {
      const results = await Promise.all(fontList.map((font) => document.fonts.load(font)));
      await document.fonts.ready;
      return results.map((matched, index) => ({
        font: fontList[index],
        status: matched.map((face) => face.status),
      }));
    }, faces);
    for (const { font, status } of loaded) expect(status, font).toEqual(['loaded']);

    const variables = await page.evaluate(() => {
      const style = getComputedStyle(document.documentElement);
      return Object.fromEntries(['bg', 'surface', 'fg', 'neg'].map((name) => [name, style.getPropertyValue(`--${name}`).trim()]));
    });
    expect(variables).toEqual(tokens[scheme]);

    expect(requests.filter((url) => /fonts\.(googleapis|gstatic)\.com/.test(url))).toEqual([]);
    const fontFiles = requests.filter((url) => url.endsWith('.woff2')).map((url) => url.slice(url.lastIndexOf('/') + 1));
    expect(fontFiles).toEqual(expect.arrayContaining(['geist-latin-400-normal.woff2', 'geist-mono-latin-500-normal.woff2', 'source-serif-4-latin-400-italic.woff2']));
    expect(requests.some((url) => url.startsWith('data:'))).toBe(false);
    expect(await page.locator('#vf-theme-style').count()).toBe(1);

    await page.screenshot({ path: `test-results/theme-${scheme}.png`, fullPage: true });
  });
}

test('data-theme overrides the system color scheme', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/examples/theme.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  const readBg = () => page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--bg').trim());
  expect(await readBg()).toBe('#262624');
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
  expect(await readBg()).toBe('#f5f4ed');
  await page.emulateMedia({ colorScheme: 'light' });
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
  expect(await readBg()).toBe('#262624');
});

test('system dark mode and data-theme="dark" define identical tokens', async ({ page }) => {
  const names = ['--bg', '--surface', '--ink', '--fg', '--muted', '--line', '--line-strong', '--pos', '--neg', 'color-scheme'];
  const readTokens = () =>
    page.evaluate((list) => {
      const style = getComputedStyle(document.documentElement);
      return Object.fromEntries(list.map((name) => [name, style.getPropertyValue(name).trim()]));
    }, names);
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/examples/theme.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  const systemDark = await readTokens();
  await page.emulateMedia({ colorScheme: 'light' });
  const systemLight = await readTokens();
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
  const forcedDark = await readTokens();
  for (const name of names) expect(forcedDark[name], name).toBe(systemDark[name]);
  expect(systemDark['--bg']).not.toBe(systemLight['--bg']);
});

test('split layout becomes one column below 980px', async ({ page }) => {
  await page.goto('/examples/theme.html');
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  const columns = () => page.locator('.vf-split').evaluate((element) => getComputedStyle(element).gridTemplateColumns.split(' ').length);
  await page.setViewportSize({ width: 1280, height: 800 });
  expect(await columns()).toBe(2);
  await page.setViewportSize({ width: 975, height: 800 });
  expect(await columns()).toBe(1);
});
