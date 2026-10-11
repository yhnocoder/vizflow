import { render, renderText } from '../formula';
import type { StepInfo } from '../steps';
import type { SceneController } from './controller';

type PartState = 'off' | 'on' | 'now' | 'grad';

export class FormulaPanel {
  private readonly el: HTMLDivElement;
  private readonly title: HTMLDivElement;
  private readonly body: HTMLDivElement;
  private key: string | undefined;

  constructor(private readonly c: SceneController) {
    this.el = document.createElement('div');
    this.el.className = 'vf-formula';
    this.el.hidden = true;
    this.title = document.createElement('div');
    this.title.className = 'vf-formula-title';
    this.body = document.createElement('div');
    this.body.className = 'vf-formula-tex';
    this.el.append(this.title, this.body);
    c.root.appendChild(this.el);
  }

  private owner(info: StepInfo): { title: string; tex: string } | undefined {
    if (info.formula !== undefined) return info.formula;
    const modules = this.c.steps?.modules ?? [];
    for (let m = modules.find((x) => x.id === info.module); m !== undefined; m = m.parent === null ? undefined : modules.find((x) => x.id === m!.parent)) {
      if (m.formula !== undefined) return { title: m.title, tex: m.formula };
    }
    return undefined;
  }

  update(): void {
    const steps = this.c.steps;
    const info = this.c.currentInfo();
    const forward = info?.dir === 'forward';
    const base = info === undefined || forward ? info : (steps?.list.find((s) => s.dir === 'forward' && s.id === info.forwardId) ?? info);
    const owner = steps === undefined || base === undefined || this.c.lod || info?.kind === 'overview' || info?.kind === 'inputs' ? undefined : this.owner(base);
    this.el.hidden = owner === undefined;
    if (owner === undefined || steps === undefined || info === undefined) return;
    const key = `${owner.title}\n${owner.tex}`;
    if (this.key !== key) {
      this.key = key;
      renderText(owner.title, this.title);
      this.body.replaceChildren(render(owner.tex, { display: true }));
    }
    const reached = new Set(forward ? steps.list.slice(0, steps.current + 1).flatMap((step) => step.marks ?? []) : []);
    const now = new Set(forward ? (info.marks ?? []) : []);
    const grad = new Set(forward ? [] : (base?.marks ?? []));
    for (const part of this.body.querySelectorAll<HTMLElement>('[data-vf-step]')) {
      const marks = (part.dataset.vfStep ?? '').split('.');
      const state: PartState = !forward ? (marks.some((m) => grad.has(m)) ? 'grad' : 'on') : marks.some((m) => now.has(m)) ? 'now' : marks.some((m) => reached.has(m)) ? 'on' : 'off';
      if (part.dataset.vfState === 'off' && state !== 'off' && !this.c.reduceMotion.matches) {
        part.classList.remove('vf-f-in');
        void part.offsetWidth;
        part.classList.add('vf-f-in');
      }
      part.dataset.vfState = state;
    }
  }
}
