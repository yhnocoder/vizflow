import { describe, expect, it } from 'vitest';
import { AVOID_STEP, avoidOverlaps, intersects } from '../../src/scene3d/avoid';
import { barColor, mix, nodeColor, outlineOpacity } from '../../src/scene3d/colors';
import { HEIGHT_CAP, MIN_HEIGHT, barHeight, niceTicks, scaleFor } from '../../src/scene3d/scales';
import { edgeStyle } from '../../src/scene3d/style';
import { TRANSITION_MS, Transition, easeOut } from '../../src/scene3d/transition';
import type { ThemeColors } from '../../src/theme';

const theme: ThemeColors = {
  bg: [0.9, 0.9, 0.9],
  surface: [1, 1, 1],
  ink: [0, 0, 0],
  fg: [0, 0, 0],
  muted: [0.5, 0.5, 0.5],
  line: [0.8, 0.8, 0.8],
  lineStrong: [0.6, 0.6, 0.6],
  pos: [1, 0.4, 0.2],
  neg: [0.2, 0.5, 1],
};

describe('scales', () => {
  it('maps the initial maximum to 1.6 units', () => {
    expect(scaleFor([0.5, -2, 1])).toBeCloseTo(0.8, 12);
    expect(scaleFor([])).toBe(1.6 / 1e-6);
  });

  it('clips at the cap and keeps a minimum height', () => {
    expect(barHeight(1, 1, true)).toEqual({ height: 1, clipped: false });
    expect(barHeight(3, 1, true)).toEqual({ height: HEIGHT_CAP, clipped: true });
    expect(barHeight(-3, 1, true)).toEqual({ height: -HEIGHT_CAP, clipped: true });
    expect(barHeight(0, 1, true)).toEqual({ height: MIN_HEIGHT, clipped: false });
    expect(barHeight(-1e-6, 1, true)).toEqual({ height: -MIN_HEIGHT, clipped: false });
    expect(barHeight(3, 1, false)).toEqual({ height: 0, clipped: false });
  });

  it('chooses three round ticks covering the data', () => {
    const q = [-0.32, -1.07, -2.36, -3.7, -2.9];
    expect(niceTicks(q, scaleFor(q))).toEqual([-4, -2, 0]);
    const gq = [-1, 0, 0, 0, 0];
    expect(niceTicks(gq, scaleFor(gq))).toEqual([-1, -0.5, 0]);
    const z = [2.37, 1.62, 0.33, -0.95, -0.21];
    const ticks = niceTicks(z, scaleFor(z));
    expect(ticks).toContain(0);
    expect(ticks.length).toBeGreaterThanOrEqual(2);
    expect(ticks.every((v) => Math.abs(v) * scaleFor(z) <= HEIGHT_CAP + 1e-9)).toBe(true);
  });
});

describe('colors', () => {
  it('mixes linearly', () => {
    expect(mix([0, 0, 0], [1, 1, 1], 0.25)).toEqual([0.25, 0.25, 0.25]);
  });

  it('picks pos, neg or line-strong and dims inactive and unfocused bars', () => {
    expect(barColor(1, theme, { active: true, focused: true })).toEqual(theme.pos);
    expect(barColor(-1, theme, { active: true, focused: true })).toEqual(theme.neg);
    expect(barColor(1e-12, theme, { active: true, focused: true })).toEqual(theme.lineStrong);
    expect(barColor(1, theme, { active: false, focused: true })).toEqual(mix(theme.pos, theme.surface, 0.55));
    expect(barColor(1, theme, { active: false, focused: false })).toEqual(mix(mix(theme.pos, theme.surface, 0.55), theme.surface, 0.45));
  });

  it('colors nodes by direction and sign of the gradient', () => {
    expect(nodeColor(1, theme, { dir: 'forward', revealed: true, active: true, focused: true })).toEqual(theme.bg);
    expect(nodeColor(-2, theme, { dir: 'backward', revealed: true, active: true, focused: true })).toEqual(theme.neg);
    expect(nodeColor(2, theme, { dir: 'backward', revealed: true, active: false, focused: true })).toEqual(mix(theme.pos, theme.surface, 0.55));
    expect(nodeColor(2, theme, { dir: 'backward', revealed: false, active: true, focused: true })).toEqual(mix(theme.bg, theme.surface, 0.5));
  });

  it('uses 0.9 for active outlines and 0.35 otherwise', () => {
    expect(outlineOpacity(true, true)).toBe(0.9);
    expect(outlineOpacity(true, false)).toBe(0.35);
    expect(outlineOpacity(false, false)).toBe(0.35);
  });
});

describe('transition', () => {
  it('eases out over 200ms', () => {
    expect(easeOut(0)).toBe(0);
    expect(easeOut(0.5)).toBeCloseTo(0.875, 12);
    expect(easeOut(1)).toBe(1);
    expect(easeOut(2)).toBe(1);
    const transition = new Transition(2, () => false);
    transition.retarget([1, -1], 1000);
    expect(transition.active).toBe(true);
    expect(transition.step(1000 + TRANSITION_MS / 2)).toBe(true);
    expect(transition.current[0]).toBeCloseTo(0.875, 12);
    expect(transition.current[1]).toBeCloseTo(-0.875, 12);
    expect(transition.step(1000 + TRANSITION_MS)).toBe(true);
    expect(transition.active).toBe(false);
    expect(Array.from(transition.current)).toEqual([1, -1]);
    expect(transition.step(1000 + TRANSITION_MS * 2)).toBe(false);
  });

  it('jumps to the target when motion is reduced', () => {
    const transition = new Transition(1, () => true);
    transition.retarget([2], 0);
    expect(transition.active).toBe(false);
    expect(transition.current[0]).toBe(2);
  });

  it('ignores retargets with the same values', () => {
    const transition = new Transition(1, () => false);
    transition.retarget([1], 0);
    transition.step(0 + TRANSITION_MS);
    transition.retarget([1], 500);
    expect(transition.active).toBe(false);
  });
});

describe('label avoidance', () => {
  const box = (x: number, y: number, w = 40, h = 16) => ({ x, y, w, h });

  it('detects intersections', () => {
    expect(intersects(box(0, 0), box(39, 15))).toBe(true);
    expect(intersects(box(0, 0), box(40, 0))).toBe(false);
  });

  it('moves the lower-priority movable label to the nearest free offset', () => {
    const result = avoidOverlaps([
      { box: box(0, 0), movable: true, hideOnOverlap: false, priority: 3 },
      { box: box(10, 4), movable: true, hideOnOverlap: false, priority: 2 },
    ]);
    expect(result[0]).toEqual({ dx: 0, dy: 0, hidden: false });
    expect(result[1]).toEqual({ dx: 0, dy: 3 * AVOID_STEP, hidden: false });
    const up = avoidOverlaps([
      { box: box(0, 0), movable: true, hideOnOverlap: false, priority: 3 },
      { box: box(10, -6), movable: true, hideOnOverlap: false, priority: 2 },
    ]);
    expect(up[1]).toEqual({ dx: 0, dy: -3 * AVOID_STEP, hidden: false });
  });

  it('tries the other directions when up is blocked', () => {
    const result = avoidOverlaps([
      { box: box(0, 0, 40, 100), movable: false, hideOnOverlap: false, priority: 4 },
      { box: box(10, 40), movable: true, hideOnOverlap: false, priority: 2 },
    ]);
    expect(result[1].dx).toBe(8 * AVOID_STEP);
    expect(result[1].dy).toBe(0);
  });

  it('hides value labels that overlap a movable label but not ones that overlap fixed labels', () => {
    const result = avoidOverlaps([
      { box: box(0, 0), movable: false, hideOnOverlap: true, priority: 1 },
      { box: box(5, 5), movable: true, hideOnOverlap: false, priority: 3 },
      { box: box(200, 200), movable: false, hideOnOverlap: true, priority: 1 },
      { box: box(205, 205), movable: false, hideOnOverlap: false, priority: 4 },
    ]);
    expect(result[0].hidden).toBe(true);
    expect(result[2].hidden).toBe(false);
  });

  it('leaves a movable label in place when no offset is free', () => {
    const result = avoidOverlaps([
      { box: box(-200, -200, 400, 400), movable: false, hideOnOverlap: false, priority: 4 },
      { box: box(0, 0), movable: true, hideOnOverlap: false, priority: 3 },
    ]);
    expect(result[1]).toEqual({ dx: 0, dy: 0, hidden: false });
  });
});

describe('edge style', () => {
  const base = {
    dir: 'forward' as const,
    overview: false,
    contract: false,
    toLoss: false,
    zeroGrad: false,
    focused: true,
    revealedFrom: true,
    revealedTo: true,
    activeFrom: false,
    activeTo: false,
  };

  it('follows the forward rules', () => {
    expect(edgeStyle({ ...base, revealedTo: false })).toEqual({ opacity: 0.1, dashed: true });
    expect(edgeStyle({ ...base, overview: true })).toEqual({ opacity: 0.42, dashed: false });
    expect(edgeStyle({ ...base, overview: true, toLoss: true })).toEqual({ opacity: 0.9, dashed: false });
    expect(edgeStyle({ ...base, activeTo: true })).toEqual({ opacity: 0.95, dashed: false });
    expect(edgeStyle({ ...base, activeTo: true, contract: true })).toEqual({ opacity: 0.75, dashed: false });
    expect(edgeStyle(base)).toEqual({ opacity: 0.3, dashed: false });
  });

  it('follows the backward rules', () => {
    const backward = { ...base, dir: 'backward' as const };
    expect(edgeStyle({ ...backward, revealedFrom: false })).toEqual({ opacity: 0.1, dashed: false });
    expect(edgeStyle({ ...backward, revealedFrom: false, zeroGrad: true })).toEqual({ opacity: 0.1, dashed: true });
    expect(edgeStyle({ ...backward, zeroGrad: true, activeFrom: true })).toEqual({ opacity: 0.3, dashed: true });
    expect(edgeStyle({ ...backward, activeFrom: true })).toEqual({ opacity: 0.95, dashed: false });
    expect(edgeStyle({ ...backward, activeFrom: true, contract: true })).toEqual({ opacity: 0.8, dashed: false });
    expect(edgeStyle(backward)).toEqual({ opacity: 0.3, dashed: false });
  });

  it('dims rows that are not in focus', () => {
    expect(edgeStyle({ ...base, activeTo: true, focused: false }).opacity).toBeCloseTo(0.95 * 0.4, 12);
  });
});
