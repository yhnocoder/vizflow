// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { onThemeChange, readTheme } from '../../src/theme';

type ChangeListener = () => void;

const schemeListeners = new Set<ChangeListener>();

beforeEach(() => {
  vi.stubGlobal('matchMedia', (query: string) => ({
    media: query,
    matches: false,
    addEventListener: (_type: string, listener: ChangeListener) => schemeListeners.add(listener),
    removeEventListener: (_type: string, listener: ChangeListener) => schemeListeners.delete(listener),
  }));
});

afterEach(() => {
  vi.unstubAllGlobals();
  schemeListeners.clear();
  document.documentElement.removeAttribute('style');
  document.documentElement.removeAttribute('data-theme');
});

const lightTokens = {
  '--bg': '#f5f4ed',
  '--surface': '#faf9f5',
  '--ink': '#141413',
  '--fg': '#141413',
  '--muted': '#6f6d65',
  '--line': '#e6e3d8',
  '--line-strong': '#b4b1a6',
  '--pos': '#d97757',
  '--neg': '#6a9bcc',
  '--rank0': '#9a62b8',
  '--rank1': '#3f9a6b',
};

function setTokens(tokens: Record<string, string>) {
  for (const [name, value] of Object.entries(tokens)) document.documentElement.style.setProperty(name, value);
}

function srgbToLinear(byte: number): number {
  const channel = byte / 255;
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
}

describe('theme style injection', () => {
  it('adds one style element to the head', () => {
    expect(document.head.querySelectorAll('#vf-theme-style')).toHaveLength(1);
  });
});

describe('readTheme', () => {
  it('converts every token to linear RGB', () => {
    setTokens(lightTokens);
    const theme = readTheme();
    expect(Object.keys(theme).sort()).toEqual(
      ['bg', 'fg', 'ink', 'line', 'lineStrong', 'muted', 'neg', 'pos', 'rank0', 'rank1', 'surface'].sort(),
    );
    expect(theme.pos[0]).toBeCloseTo(srgbToLinear(0xd9), 10);
    expect(theme.pos[1]).toBeCloseTo(srgbToLinear(0x77), 10);
    expect(theme.pos[2]).toBeCloseTo(srgbToLinear(0x57), 10);
    expect(theme.lineStrong).toEqual([0xb4, 0xb1, 0xa6].map(srgbToLinear));
  });

  it('uses the linear segment for dark channels', () => {
    setTokens({ ...lightTokens, '--ink': '#0a0000' });
    expect(readTheme().ink).toEqual([10 / 255 / 12.92, 0, 0]);
  });

  it('accepts three digit hex colors', () => {
    setTokens({ ...lightTokens, '--bg': '#fff' });
    expect(readTheme().bg).toEqual([1, 1, 1]);
  });

  it('names the variable when a token is not a hex color', () => {
    setTokens({ ...lightTokens, '--neg': 'oklch(0.7 0.1 240)' });
    expect(() => readTheme()).toThrow('readTheme: --neg must be a #rgb or #rrggbb color, got "oklch(0.7 0.1 240)"');
  });
});

describe('onThemeChange', () => {
  it('fires on a color scheme change and on data-theme changes until unsubscribed', async () => {
    const fn = vi.fn();
    const unsubscribe = onThemeChange(fn);
    schemeListeners.forEach((listener) => listener());
    expect(fn).toHaveBeenCalledTimes(1);
    document.documentElement.setAttribute('data-theme', 'dark');
    await Promise.resolve();
    expect(fn).toHaveBeenCalledTimes(2);
    unsubscribe();
    expect(schemeListeners.size).toBe(0);
    document.documentElement.setAttribute('data-theme', 'light');
    await Promise.resolve();
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it('ignores other attribute changes on the root element', async () => {
    const fn = vi.fn();
    const unsubscribe = onThemeChange(fn);
    document.documentElement.setAttribute('lang', 'en');
    await Promise.resolve();
    expect(fn).not.toHaveBeenCalled();
    unsubscribe();
  });
});
