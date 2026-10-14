import { renderText } from '../formula';
import type { SceneController } from './controller';

export const JUMP_PLACEHOLDER = '跳到模块…';
export const VIEW_MENU = '视图';

function chip(text: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.className = 'vf-chip';
  button.type = 'button';
  button.textContent = text;
  return button;
}

export function buildChrome(root: HTMLDivElement, hint: string | undefined): void {
  const key = document.createElement('div');
  key.className = 'vf-key';
  key.innerHTML = `<span><i class="vf-sw" style="background:var(--pos)"></i>正</span><span><i class="vf-sw" style="background:var(--neg)"></i>负</span>`;
  root.appendChild(key);
  if (hint !== undefined) {
    const el = document.createElement('div');
    el.className = 'vf-hint';
    el.textContent = hint;
    root.appendChild(el);
  }
}

function buildMenu(toggle: HTMLButtonElement, label: string, items: HTMLButtonElement[]): HTMLDivElement {
  const wrap = document.createElement('div');
  wrap.className = 'vf-jump';
  toggle.setAttribute('aria-label', label);
  toggle.setAttribute('aria-haspopup', 'menu');
  toggle.setAttribute('aria-expanded', 'false');
  const menu = document.createElement('div');
  menu.className = 'vf-jump-menu';
  menu.setAttribute('role', 'menu');
  menu.hidden = true;
  const close = () => {
    menu.hidden = true;
    toggle.setAttribute('aria-expanded', 'false');
  };
  for (const item of items) menu.appendChild(item);
  toggle.addEventListener('click', () => {
    menu.hidden = !menu.hidden;
    toggle.setAttribute('aria-expanded', String(!menu.hidden));
  });
  wrap.addEventListener('focusout', (ev) => {
    if (!wrap.contains(ev.relatedTarget as globalThis.Node | null)) close();
  });
  wrap.append(toggle, menu);
  return wrap;
}

function menuItem(role: 'menuitem' | 'menuitemcheckbox'): HTMLButtonElement {
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'vf-jump-item';
  item.setAttribute('role', role);
  return item;
}

function buildJump(c: SceneController): HTMLDivElement {
  const toggle = chip(JUMP_PLACEHOLDER);
  const items = (c.options.modules ?? []).map((module) => {
    const item = menuItem('menuitem');
    renderText(module.title, item);
    item.addEventListener('click', () => {
      toggle.focus();
      const brace = c.moduleBraces.braces.find((b) => b.module === module);
      if (brace === undefined) return;
      c.rig.jumpTo((brace.y0 + brace.y1) / 2);
      c.refresh();
    });
    return item;
  });
  return buildMenu(toggle, '跳到模块', items);
}

function buildView(c: SceneController): HTMLDivElement {
  const option = (text: string, toggle: () => boolean) => {
    const item = menuItem('menuitemcheckbox');
    item.textContent = text;
    item.setAttribute('aria-checked', 'true');
    item.addEventListener('click', () => {
      item.setAttribute('aria-checked', String(toggle()));
      c.refresh();
    });
    return item;
  };
  const items: HTMLButtonElement[] = [];
  if (c.follow) items.push(option('远处淡出', () => c.toggleFar()));
  if (c.options.steps !== undefined) items.push(option('遮挡半透明', () => c.toggleSeeThrough()));
  items.push(option('只画最前一行', () => c.toggleFrontRow()));
  return buildMenu(chip(VIEW_MENU), VIEW_MENU, items);
}

export function buildTools(c: SceneController): void {
  const tools = document.createElement('div');
  tools.className = 'vf-stage-tools';
  if (c.follow && (c.options.modules ?? []).length > 0) tools.appendChild(buildJump(c));
  tools.appendChild(buildView(c));
  const reset = chip('复位视角');
  reset.addEventListener('click', () => c.resetView());
  tools.appendChild(reset);
  c.root.appendChild(tools);
}
