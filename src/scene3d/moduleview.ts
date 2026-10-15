import { renderText } from '../formula';
import type { ThemeColors } from '../theme';
import type { SceneController } from './controller';
import { boundsOf, framingPoints } from './framing';
import type { Label } from './labels';
import type { Vec3 } from './layout';
import { BRACE_X } from './levels';
import { LineBatch } from './lines';
import { type Brace, type SceneModule, BRACE_OPACITY, BRACE_THICK_OFFSET, blockSpan, braceOffset, braceScale, bracePoints, braceTip, layoutBraces, levelBraces } from './modules';

export class ModuleBraces {
  private readonly ink: LineBatch;
  private labels: Label[] = [];
  braces: Brace[] = [];

  constructor(private readonly c: SceneController) {
    this.ink = new LineBatch(c.theme.ink);
    c.scene.add(this.ink.object);
  }

  mountLabels(modules: SceneModule[]): void {
    this.labels = modules.map((module) => this.moduleLabel(module));
  }

  private foldable(module: SceneModule): string | undefined {
    const steps = this.c.steps;
    if (module.id === undefined || steps?.setFolded === undefined || steps.folded === undefined) return undefined;
    if (steps.modules.find((m) => m.id === module.id)?.foldable !== true) return undefined;
    return module.id;
  }

  private moduleLabel(module: SceneModule): Label {
    const side = module.side ?? 'left';
    const id = this.foldable(module);
    const label = this.c.labels.add('mod', `vf-mod-${side === 'outer' ? 'right vf-mod-thick' : side}${id === undefined ? '' : ' vf-foldable'}`);
    label.anchor = side === 'left' ? 'right-middle' : 'left-middle';
    label.el.dataset.module = module.title;
    if (id !== undefined) label.el.addEventListener('click', () => this.c.steps?.setFolded?.(id, this.c.steps.folded?.(id) !== true));
    return label;
  }

  private moduleContent(label: Label, module: SceneModule): void {
    const id = this.foldable(module);
    const folded = id === undefined ? undefined : this.c.steps?.folded?.(id);
    const action = folded === undefined ? '' : folded ? '▸ 展开' : '▾ 折叠';
    const brief = this.c.lod;
    const subtitle = brief ? undefined : module.subtitle;
    const act = brief ? '' : action;
    label.setContent(`${module.title}\n${subtitle ?? ''}\n${act}`, () => {
      const fragment = document.createDocumentFragment();
      const title = document.createElement('span');
      title.className = 'vf-mod-title';
      renderText(module.title, title);
      fragment.appendChild(title);
      if (subtitle !== undefined) {
        const sub = document.createElement('span');
        sub.className = 'vf-mod-sub';
        renderText(subtitle, sub);
        fragment.appendChild(sub);
      }
      if (act !== '') {
        const el = document.createElement('span');
        el.className = 'vf-mod-act';
        el.textContent = act;
        fragment.appendChild(el);
      }
      return fragment;
    });
  }

  update(): void {
    const c = this.c;
    const modules = c.options.modules ?? [];
    if (modules.length === 0) return;
    const levels = c.levelLayout;
    if (levels !== undefined) {
      const xs = c.renderables.flatMap((r) => r.block.elements.map((e) => e.x));
      const shown = modules.filter((m) => !(c.lod && (m.depth ?? 0) > 0));
      this.braces = levelBraces(shown, levels.ranges, { xMin: Math.min(...xs), xMax: Math.max(...xs), ...levels.braces }, c.lod ? { ...BRACE_X, outer: 0 } : BRACE_X);
    } else {
      const { min, max } = boundsOf(framingPoints(c.renderables));
      this.braces = layoutBraces(
        modules,
        c.renderables.map((r) => blockSpan(r.block)),
        { xMin: min[0], xMax: max[0] },
      );
    }
    modules.forEach((module, i) => {
      const label = this.labels[i];
      const brace = this.braces.find((b) => b.module === module);
      label.visible = brace !== undefined;
      if (brace === undefined) return;
      this.moduleContent(label, brace.module);
      if (label.pos === undefined) label.pos = braceTip(brace);
    });
  }

  place(): void {
    if (this.braces.length === 0) {
      this.ink.setPolylines([]);
      return;
    }
    const view = this.c.rig.viewDirection();
    const direction: Vec3 = [view.x, view.y, view.z];
    const modules = this.c.options.modules ?? [];
    const scale = braceScale(this.c.camera.zoom);
    const polylines = this.braces.flatMap((brace) => {
      const offset = braceOffset(brace, direction);
      const label = this.labels[modules.indexOf(brace.module)];
      label.pos = braceTip(brace, offset, scale);
      label.lane = brace.side;
      const points = bracePoints(brace, offset, scale);
      const line = { points, opacity: BRACE_OPACITY };
      if (brace.thick !== true) return [line];
      return [line, { ...line, points: points.map((v, i) => (i % 3 === 0 ? v + BRACE_THICK_OFFSET * scale : v)) }];
    });
    this.ink.setPolylines(polylines);
  }

  setTheme(theme: ThemeColors): void {
    this.ink.setColor(theme.ink);
  }

  dispose(): void {
    this.ink.dispose();
  }
}
