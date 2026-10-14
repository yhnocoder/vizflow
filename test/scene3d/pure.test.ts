import { describe, expect, it } from 'vitest';
import { AVOID_STEP, avoidOverlaps, intersects } from '../../src/scene3d/avoid';
import { barColor, mix, nodeColor, outlineOpacity } from '../../src/scene3d/colors';
import { HEIGHT_CAP, MIN_HEIGHT, barHeight, capFor, niceTicks, scaleFor } from '../../src/scene3d/scales';
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
  it('maps the initial maximum to 0.8 of the cap', () => {
    expect(scaleFor([0.5, -2, 1], 'fp32')).toBeCloseTo(0.48, 12);
    expect(scaleFor([0.5, -2, 1], 'fp32', 0.85)).toBeCloseTo(0.34, 12);
    expect(scaleFor([], 'fp32')).toBe(1);
  });

  it('picks the cap by shape', () => {
    expect(capFor('bar', 2)).toBe(1.2);
    expect(capFor('bar', 1)).toBe(1.0);
    expect(capFor('tile', 2)).toBe(0.85);
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
    expect(niceTicks(q, 'fp32', scaleFor(q, 'fp32'))).toEqual([-4, -2, 0]);
    const gq = [-1, 0, 0, 0, 0];
    expect(niceTicks(gq, 'fp32', scaleFor(gq, 'fp32'))).toEqual([-1, -0.5, 0]);
    const z = [2.37, 1.62, 0.33, -0.95, -0.21];
    const ticks = niceTicks(z, 'fp32', scaleFor(z, 'fp32'));
    expect(ticks).toContain(0);
    expect(ticks.length).toBeGreaterThanOrEqual(2);
    expect(ticks.every((v) => Math.abs(v) * scaleFor(z, 'fp32') <= HEIGHT_CAP + 1e-9)).toBe(true);
  });

  it('ignores infinite and saturated values such as a causal mask', () => {
    const masked = [0.5, -Infinity, -2, -3.3895313892515355e38];
    expect(scaleFor(masked, 'bf16')).toBeCloseTo(0.48, 12);
    expect(niceTicks(masked, 'bf16', scaleFor(masked, 'bf16'))).toEqual(niceTicks([0.5, -2], 'bf16', scaleFor([0.5, -2], 'bf16')));
    expect(barHeight(-Infinity, 0.8, true)).toEqual({ height: -HEIGHT_CAP, clipped: true });
  });
});

describe('colors', () => {
  it('mixes linearly', () => {
    expect(mix([0, 0, 0], [1, 1, 1], 0.25)).toEqual([0.25, 0.25, 0.25]);
  });

  it('picks pos, neg or line-strong and dims inactive and unfocused bars', () => {
    expect(barColor(1, theme, { active: true, far: false, focused: true })).toEqual(theme.pos);
    expect(barColor(-1, theme, { active: true, far: false, focused: true })).toEqual(theme.neg);
    expect(barColor(1e-12, theme, { active: true, far: false, focused: true })).toEqual(theme.lineStrong);
    expect(barColor(1, theme, { active: false, far: false, focused: true })).toEqual(mix(theme.pos, theme.surface, 0.55));
    expect(barColor(1, theme, { active: false, far: true, focused: true })).toEqual(mix(theme.pos, theme.surface, 0.88));
    expect(barColor(1, theme, { active: false, far: false, focused: false })).toEqual(mix(mix(theme.pos, theme.surface, 0.55), theme.surface, 0.45));
  });

  it('colors nodes by direction and sign of the gradient', () => {
    const node = { far: false, focused: true };
    expect(nodeColor(1, theme, { ...node, dir: 'forward', revealed: true, active: true })).toEqual(theme.bg);
    expect(nodeColor(1, theme, { ...node, dir: 'forward', revealed: true, active: false })).toEqual(mix(theme.bg, theme.surface, 0.6));
    expect(nodeColor(-2, theme, { ...node, dir: 'backward', revealed: true, active: true })).toEqual(theme.neg);
    expect(nodeColor(0, theme, { ...node, dir: 'backward', revealed: true, active: true })).toEqual(theme.lineStrong);
    expect(nodeColor(2, theme, { ...node, dir: 'backward', revealed: true, active: false })).toEqual(mix(theme.pos, theme.surface, 0.55));
    expect(nodeColor(2, theme, { ...node, dir: 'backward', revealed: false, active: true })).toEqual(mix(theme.bg, theme.surface, 0.5));
    expect(nodeColor(2, theme, { ...node, far: true, dir: 'backward', revealed: true, active: false })).toEqual(mix(theme.bg, theme.surface, 0.85));
  });

  it('uses 0.85 for active outlines, 0.32 for idle and 0.08 far away', () => {
    expect(outlineOpacity({ active: true, far: false, tile: false })).toBe(0.85);
    expect(outlineOpacity({ active: false, far: false, tile: false })).toBe(0.32);
    expect(outlineOpacity({ active: false, far: true, tile: false })).toBe(0.08);
    expect(outlineOpacity({ active: true, far: false, tile: true })).toBeCloseTo(0.85 * 0.85, 12);
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

  it('hides lower-priority fixed labels that overlap a placed label and keeps the rest', () => {
    const result = avoidOverlaps([
      { box: box(0, 0), movable: false, hideOnOverlap: true, priority: 1 },
      { box: box(5, 5), movable: true, hideOnOverlap: false, priority: 3 },
      { box: box(200, 200), movable: false, hideOnOverlap: true, priority: 1 },
      { box: box(205, 205), movable: false, hideOnOverlap: true, priority: 2 },
    ]);
    expect(result[0].hidden).toBe(true);
    expect(result[2].hidden).toBe(true);
    expect(result[3].hidden).toBe(false);
  });

  it('keeps a movable label in place when no offset is free', () => {
    const result = avoidOverlaps([
      { box: box(-200, -200, 400, 400), movable: false, hideOnOverlap: false, priority: 4 },
      { box: box(0, 0), movable: true, hideOnOverlap: false, priority: 3 },
      { box: box(300, 0), movable: true, hideOnOverlap: false, priority: 2 },
      { box: box(305, 5), movable: true, hideOnOverlap: false, priority: 1 },
    ]);
    expect(result[1]).toEqual({ dx: 0, dy: 0, hidden: false });
    expect(result[2]).toEqual({ dx: 0, dy: 0, hidden: false });
    expect(result[3].hidden).toBe(false);
  });

  it('keeps a label with a user offset visible where it overlaps', () => {
    const result = avoidOverlaps([
      { box: box(0, 0), movable: true, hideOnOverlap: false, priority: 3 },
      { box: box(5, 5), movable: false, hideOnOverlap: false, priority: 1 },
    ]);
    expect(result[1]).toEqual({ dx: 0, dy: 0, hidden: false });
  });

  it('keeps labels inside the bounds: moves movable ones in and hides fixed ones outside', () => {
    const bounds = box(12, 12, 200, 100);
    const result = avoidOverlaps([
      { box: box(190, 50), movable: true, hideOnOverlap: false, priority: 4 },
      { box: box(195, 20), movable: false, hideOnOverlap: true, priority: 1 },
      { box: box(50, 50), movable: false, hideOnOverlap: true, priority: 1 },
    ], bounds);
    expect(result[0].hidden).toBe(false);
    expect(190 + result[0].dx + 40).toBeLessThanOrEqual(212);
    expect(result[1].hidden).toBe(true);
    expect(result[2].hidden).toBe(false);
  });

  it('keeps a movable label that cannot be moved into the bounds', () => {
    const result = avoidOverlaps([{ box: box(300, -40), movable: true, hideOnOverlap: false, priority: 3 }], box(0, 0, 100, 100));
    expect(result[0]).toEqual({ dx: 0, dy: 0, hidden: false });
  });
});

describe('edge style', () => {
  const base = {
    dir: 'forward' as const,
    overview: false,
    contract: false,
    zeroGrad: false,
    focused: true,
    far: false,
    current: false,
    revealedFrom: true,
    revealedTo: true,
  };

  it('follows the forward rules', () => {
    expect(edgeStyle({ ...base, revealedTo: false })).toBeNull();
    expect(edgeStyle({ ...base, overview: true })).toEqual({ opacity: 0.35 });
    expect(edgeStyle({ ...base, overview: true, contract: true })).toEqual({ opacity: 0.2 });
    expect(edgeStyle({ ...base, current: true })).toEqual({ opacity: 0.85 });
    expect(edgeStyle({ ...base, current: true, contract: true })).toEqual({ opacity: 0.6 });
    expect(edgeStyle(base)).toEqual({ opacity: 0.13 });
    expect(edgeStyle({ ...base, far: true })).toBeNull();
  });

  it('follows the backward rules', () => {
    const backward = { ...base, dir: 'backward' as const };
    expect(edgeStyle({ ...backward, revealedFrom: false })).toBeNull();
    expect(edgeStyle({ ...backward, revealedFrom: false, zeroGrad: true })).toBeNull();
    expect(edgeStyle({ ...backward, zeroGrad: true, current: true })).toEqual({ opacity: 0.35 });
    expect(edgeStyle({ ...backward, current: true })).toEqual({ opacity: 0.9 });
    expect(edgeStyle({ ...backward, current: true, contract: true })).toEqual({ opacity: 0.65 });
    expect(edgeStyle(backward)).toEqual({ opacity: 0.13 });
  });

  it('dims rows that are not in focus', () => {
    expect(edgeStyle({ ...base, current: true, focused: false })!.opacity).toBeCloseTo(0.85 * 0.4, 12);
  });
});
