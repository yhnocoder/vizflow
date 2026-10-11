import { renderText } from '../formula';
import type { SceneController } from './controller';
import { parallelLegend } from './parallel';

export const KEY_FRAME_TEXT = '墨线框 = 被取用的行或标签';
export const JUMP_PLACEHOLDER = '跳到模块…';

export interface OverviewNav {
  prev: HTMLButtonElement;
  next: HTMLButtonElement;
  text: HTMLSpanElement;
}

function chip(text: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.className = 'vf-chip';
  button.type = 'button';
  button.textContent = text;
  return button;
}

export function buildChrome(root: HTMLDivElement, hint: string | undefined, parallel = false): void {
  const key = document.createElement('div');
  key.className = 'vf-key';
  key.innerHTML = `<span><i class="vf-sw" style="background:var(--pos)"></i>正</span><span><i class="vf-sw" style="background:var(--neg)"></i>负</span><span>${KEY_FRAME_TEXT}</span>`;
  root.appendChild(key);
  if (parallel) {
    const ranks = document.createElement('div');
    ranks.className = 'vf-key vf-ranks';
    ranks.innerHTML = parallelLegend();
    root.appendChild(ranks);
  }
  if (hint !== undefined) {
    const el = document.createElement('div');
    el.className = 'vf-hint';
    el.textContent = hint;
    root.appendChild(el);
  }
}

function buildJump(c: SceneController): HTMLDivElement {
  const modules = c.options.modules ?? [];
  const jump = document.createElement('div');
  jump.className = 'vf-jump';
  const toggle = chip(JUMP_PLACEHOLDER);
  toggle.setAttribute('aria-label', '跳到模块');
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
  modules.forEach((module) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'vf-jump-item';
    item.setAttribute('role', 'menuitem');
    renderText(module.title, item);
    item.addEventListener('click', () => {
      close();
      const brace = c.moduleBraces.braces.find((b) => b.module === module);
      if (brace === undefined) return;
      c.rig.jumpTo((brace.y0 + brace.y1) / 2);
      c.refresh();
    });
    menu.appendChild(item);
  });
  toggle.addEventListener('click', () => {
    menu.hidden = !menu.hidden;
    toggle.setAttribute('aria-expanded', String(!menu.hidden));
  });
  jump.addEventListener('focusout', (ev) => {
    if (!jump.contains(ev.relatedTarget as globalThis.Node | null)) close();
  });
  jump.append(toggle, menu);
  return jump;
}

export function buildTools(c: SceneController): OverviewNav | undefined {
  const tools = document.createElement('div');
  tools.className = 'vf-stage-tools';
  const modules = c.options.modules ?? [];
  if (c.follow && modules.length > 0) tools.appendChild(buildJump(c));
  if (c.follow) {
    const far = chip('远处淡出');
    far.setAttribute('aria-pressed', 'true');
    far.addEventListener('click', () => {
      far.setAttribute('aria-pressed', String(c.toggleFar()));
      c.refresh();
    });
    tools.appendChild(far);
  }
  if (c.options.steps !== undefined) {
    const see = chip('遮挡半透明');
    see.setAttribute('aria-pressed', 'true');
    see.addEventListener('click', () => {
      see.setAttribute('aria-pressed', String(c.toggleSeeThrough()));
      c.refresh();
    });
    tools.appendChild(see);
  }
  const front = chip('只画最前一行');
  front.setAttribute('aria-pressed', 'true');
  front.addEventListener('click', () => {
    front.setAttribute('aria-pressed', String(c.toggleFrontRow()));
    c.refresh();
  });
  tools.appendChild(front);
  if (c.follow && modules.length > 0 && c.options.steps !== undefined) {
    const foldAll = chip('全部折叠');
    foldAll.addEventListener('click', () => c.steps?.setAllFolded?.(true));
    const unfoldAll = chip('全部展开');
    unfoldAll.addEventListener('click', () => c.steps?.setAllFolded?.(false));
    tools.append(foldAll, unfoldAll);
  }
  const reset = chip('复位视角');
  reset.addEventListener('click', () => c.resetView());
  tools.appendChild(reset);
  c.root.appendChild(tools);
  if (!c.follow || c.options.steps === undefined) return undefined;
  const steps = c.options.steps;
  const nav = document.createElement('div');
  nav.className = 'vf-ovnav';
  const button = (text: string, label: string, go: () => void) => {
    const b = document.createElement('button');
    b.className = 'vf-nav';
    b.type = 'button';
    b.textContent = text;
    b.setAttribute('aria-label', label);
    b.addEventListener('click', go);
    return b;
  };
  const prev = button('←', '上一步', () => steps.prev());
  const next = button('→', '下一步', () => steps.next());
  const text = document.createElement('span');
  text.className = 'vf-ovnav-text';
  nav.append(prev, next, text);
  c.root.appendChild(nav);
  return { prev, next, text };
}

export function updateNav(c: SceneController, nav: OverviewNav): void {
  const steps = c.options.steps;
  const info = c.currentInfo();
  if (steps === undefined || info === undefined) return;
  nav.prev.disabled = steps.current === 0;
  nav.next.disabled = steps.current === steps.list.length - 1;
  const where = steps.modules.find((m) => m.id === info.module)?.title;
  const key = `${steps.current}:${info.label}:${where ?? ''}`;
  if (nav.text.dataset.key === key) return;
  nav.text.dataset.key = key;
  const small = document.createElement('small');
  renderText(`${info.dir === 'forward' ? '前向' : '反向'}${where ? ` · ${where}` : ''}`, small);
  const label = document.createElement('span');
  renderText(info.label, label);
  nav.text.replaceChildren(small, ...label.childNodes);
}
