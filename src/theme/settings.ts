import { loadBackgroundBlob, saveBackgroundBlob } from './background-store';
import { syncGlass } from './glass';

export const themeNames = ['default', 'catppuccin-latte', 'liquid-glass'] as const;

export type ThemeName = (typeof themeNames)[number];

export interface RevealOrigin {
  x: number;
  y: number;
}

interface StoredSettings {
  theme: ThemeName;
  background: number;
}

const storageKey = 'vizflow.settings';
const themeAttribute = 'data-vf-theme';
const backdropAttribute = 'data-vf-backdrop';
const FADE_MS = 480;
const defaults: StoredSettings = { theme: 'default', background: 0 };

let applied: StoredSettings = { ...defaults };
let backdrop: HTMLDivElement | undefined;
let backdropUrl: string | undefined;
let image: Blob | undefined;
let backgroundLoad = 0;

function isThemeName(value: unknown): value is ThemeName {
  return themeNames.includes(value as ThemeName);
}

function readStored(): StoredSettings {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(storageKey) ?? 'null');
    if (typeof raw !== 'object' || raw === null) return { ...defaults };
    const { theme, background } = raw as Partial<StoredSettings>;
    return {
      theme: isThemeName(theme) ? theme : defaults.theme,
      background: typeof background === 'number' && Number.isFinite(background) ? background : 0,
    };
  } catch {
    return { ...defaults };
  }
}

function writeStored(settings: StoredSettings): void {
  try {
    localStorage.setItem(storageKey, JSON.stringify(settings));
  } catch {
    return;
  }
}

function reducedMotion(): boolean {
  return matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function animate(update: () => void, origin?: RevealOrigin): void {
  const root = document.documentElement;
  if (reducedMotion()) {
    update();
    return;
  }
  if (origin !== undefined && typeof document.startViewTransition === 'function') {
    const radius = Math.hypot(Math.max(origin.x, innerWidth - origin.x), Math.max(origin.y, innerHeight - origin.y));
    root.style.setProperty('--vf-reveal-x', `${origin.x}px`);
    root.style.setProperty('--vf-reveal-y', `${origin.y}px`);
    root.style.setProperty('--vf-reveal-r', `${Math.ceil(radius)}px`);
    root.classList.add('vf-theme-revealing');
    const transition = document.startViewTransition(update);
    void transition.finished.finally(() => root.classList.remove('vf-theme-revealing'));
    return;
  }
  root.classList.add('vf-theme-fading');
  update();
  setTimeout(() => root.classList.remove('vf-theme-fading'), FADE_MS);
}

function syncBackdrop(): void {
  const root = document.documentElement;
  const needed = applied.theme === 'liquid-glass' || backdropUrl !== undefined;
  root.toggleAttribute(backdropAttribute, backdropUrl !== undefined);
  if (!needed) {
    backdrop?.remove();
    backdrop = undefined;
    return;
  }
  if (backdrop === undefined) {
    backdrop = document.createElement('div');
    backdrop.className = 'vf-backdrop';
    backdrop.setAttribute('aria-hidden', 'true');
  }
  if (!backdrop.isConnected) document.body.prepend(backdrop);
  backdrop.classList.toggle('vf-backdrop-image', backdropUrl !== undefined);
  backdrop.style.backgroundImage = backdropUrl === undefined ? '' : `url("${backdropUrl}")`;
}

function applyTheme(theme: ThemeName): void {
  applied.theme = theme;
  const root = document.documentElement;
  if (theme === 'default') root.removeAttribute(themeAttribute);
  else root.setAttribute(themeAttribute, theme);
  syncBackdrop();
  syncGlass(theme === 'liquid-glass');
}

function showBackground(blob: Blob | undefined): void {
  image = blob;
  if (backdropUrl !== undefined) URL.revokeObjectURL(backdropUrl);
  backdropUrl = blob === undefined ? undefined : URL.createObjectURL(blob);
  syncBackdrop();
}

async function applyBackground(version: number, animated: boolean): Promise<void> {
  const load = ++backgroundLoad;
  applied.background = version;
  const blob = version === 0 ? undefined : await loadBackgroundBlob().catch(() => undefined);
  if (load !== backgroundLoad) return;
  if (animated) animate(() => showBackground(blob));
  else showBackground(blob);
}

export function getTheme(): ThemeName {
  return applied.theme;
}

export function setTheme(theme: ThemeName, origin?: RevealOrigin): void {
  if (!isThemeName(theme)) throw new Error(`setTheme: unknown theme "${String(theme)}", expected one of ${themeNames.join(', ')}`);
  writeStored({ ...readStored(), theme });
  if (theme === applied.theme) return;
  animate(() => applyTheme(theme), origin);
}

export function getBackgroundImage(): Promise<Blob | undefined> {
  return Promise.resolve(image);
}

export async function setBackgroundImage(next: Blob | null): Promise<void> {
  if (next !== null && !next.type.startsWith('image/')) throw new Error(`setBackgroundImage: expected an image file, got type "${next.type}"`);
  const saved = await saveBackgroundBlob(next ?? undefined).then(
    () => true,
    () => false,
  );
  const version = next === null ? 0 : Date.now();
  if (saved) writeStored({ ...readStored(), background: version });
  backgroundLoad++;
  applied.background = version;
  animate(() => showBackground(next ?? undefined));
}

function syncFromStorage(event: StorageEvent): void {
  if (event.key !== storageKey && event.key !== null) return;
  const stored = readStored();
  if (stored.theme !== applied.theme) animate(() => applyTheme(stored.theme));
  if (stored.background !== applied.background) void applyBackground(stored.background, true);
}

function start(): void {
  const stored = readStored();
  applyTheme(stored.theme);
  if (stored.background !== 0) void applyBackground(stored.background, false);
  addEventListener('storage', syncFromStorage);
}

if (document.body === null) document.addEventListener('DOMContentLoaded', start, { once: true });
else start();
