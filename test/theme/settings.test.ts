// @vitest-environment jsdom
import { beforeAll, describe, expect, it, vi } from 'vitest';
import type * as Theme from '../../src/theme';
import { displacementMap } from '../../src/theme/glass';

let theme: typeof Theme;

beforeAll(async () => {
  vi.stubGlobal('matchMedia', (query: string) => ({ media: query, matches: false, addEventListener: () => {}, removeEventListener: () => {} }));
  localStorage.setItem('vizflow.settings', JSON.stringify({ theme: 'catppuccin-latte', background: 0 }));
  theme = await import('../../src/theme');
});

const root = () => document.documentElement;

describe('theme settings', () => {
  it('applies the saved theme on import', () => {
    expect(theme.getTheme()).toBe('catppuccin-latte');
    expect(root().getAttribute('data-vf-theme')).toBe('catppuccin-latte');
  });

  it('saves the chosen theme and removes the attribute for the default theme', () => {
    theme.setTheme('default');
    expect(root().hasAttribute('data-vf-theme')).toBe(false);
    expect(JSON.parse(localStorage.getItem('vizflow.settings')!)).toEqual({ theme: 'default', background: 0 });
  });

  it('adds the backdrop layer for liquid glass and removes it again', () => {
    theme.setTheme('liquid-glass');
    expect(root().getAttribute('data-vf-theme')).toBe('liquid-glass');
    expect(document.querySelectorAll('body > .vf-backdrop')).toHaveLength(1);
    expect(root().hasAttribute('data-vf-refract')).toBe(false);
    theme.setTheme('default');
    expect(document.querySelector('.vf-backdrop')).toBeNull();
  });

  it('follows a theme change made in another tab', () => {
    localStorage.setItem('vizflow.settings', JSON.stringify({ theme: 'catppuccin-latte', background: 0 }));
    dispatchEvent(new StorageEvent('storage', { key: 'vizflow.settings' }));
    expect(theme.getTheme()).toBe('catppuccin-latte');
    theme.setTheme('default');
  });

  it('rejects unknown theme names and non-image backgrounds', async () => {
    expect(() => theme.setTheme('dark' as Theme.ThemeName)).toThrow('setTheme: unknown theme "dark", expected one of default, catppuccin-latte, liquid-glass');
    await expect(theme.setBackgroundImage(new Blob(['x'], { type: 'text/plain' }))).rejects.toThrow('setBackgroundImage: expected an image file, got type "text/plain"');
  });
});

describe('displacementMap', () => {
  const width = 40;
  const height = 30;
  const map = displacementMap(width, height, 6, 10);
  const pixel = (x: number, y: number) => Array.from(map.slice((y * width + x) * 4, (y * width + x) * 4 + 4));

  it('leaves the center untouched', () => {
    expect(pixel(20, 15)).toEqual([128, 128, 128, 255]);
  });

  it('points every edge toward the center', () => {
    expect(pixel(0, 15)[0]).toBeGreaterThan(200);
    expect(pixel(width - 1, 15)[0]).toBeLessThan(56);
    expect(pixel(20, 0)[1]).toBeGreaterThan(200);
    expect(pixel(20, height - 1)[1]).toBeLessThan(56);
  });

  it('leaves pixels outside the rounded corner untouched', () => {
    expect(pixel(0, 0)).toEqual([128, 128, 128, 255]);
  });
});
