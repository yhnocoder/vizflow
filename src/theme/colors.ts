export type LinearRgb = [number, number, number];

export interface ThemeColors {
  bg: LinearRgb;
  surface: LinearRgb;
  ink: LinearRgb;
  fg: LinearRgb;
  muted: LinearRgb;
  line: LinearRgb;
  lineStrong: LinearRgb;
  pos: LinearRgb;
  neg: LinearRgb;
}

const variables: Record<keyof ThemeColors, string> = {
  bg: '--bg',
  surface: '--surface',
  ink: '--ink',
  fg: '--fg',
  muted: '--muted',
  line: '--line',
  lineStrong: '--line-strong',
  pos: '--pos',
  neg: '--neg',
};

function srgbToLinear(channel: number): number {
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
}

function parseHexColor(name: string, value: string): LinearRgb {
  const match = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(value.trim());
  if (!match) throw new Error(`readTheme: ${name} must be a #rgb or #rrggbb color, got "${value}"`);
  const digits = match[1].length === 3 ? [...match[1]].map((digit) => digit + digit) : match[1].match(/../g)!;
  const [r, g, b] = digits.map((pair) => srgbToLinear(parseInt(pair, 16) / 255));
  return [r, g, b];
}

export function readTheme(): ThemeColors {
  const style = getComputedStyle(document.documentElement);
  const entries = Object.entries(variables).map(([key, name]) => [key, parseHexColor(name, style.getPropertyValue(name))]);
  return Object.fromEntries(entries) as ThemeColors;
}

export function onThemeChange(fn: () => void): () => void {
  const scheme = matchMedia('(prefers-color-scheme: dark)');
  scheme.addEventListener('change', fn);
  const observer = new MutationObserver(fn);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  return () => {
    scheme.removeEventListener('change', fn);
    observer.disconnect();
  };
}
