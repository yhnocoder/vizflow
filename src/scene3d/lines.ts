import { BufferAttribute, BufferGeometry, Color, LineSegments, ShaderMaterial } from 'three';
import type { LinearRgb } from '../theme';

export interface Polyline {
  points: ArrayLike<number>;
  opacity: number;
  dashed?: boolean;
}

export const DASH = 0.14;
export const DASH_GAP = 0.1;

function dashes(p: ArrayLike<number>): number[] {
  const out: number[] = [];
  let travelled = 0;
  for (let i = 0; i + 5 < p.length; i += 3) {
    const length = Math.hypot(p[i + 3] - p[i], p[i + 4] - p[i + 1], p[i + 5] - p[i + 2]);
    const at = (t: number) => [0, 1, 2].map((k) => p[i + k] + ((p[i + 3 + k] - p[i + k]) * t) / length);
    const period = DASH + DASH_GAP;
    for (let n = Math.floor(travelled / period); n * period < travelled + length; n++) {
      const start = Math.max(0, n * period - travelled);
      const end = Math.min(length, n * period + DASH - travelled);
      if (end > start) out.push(...at(start), ...at(end));
    }
    travelled += length;
  }
  return out;
}

const vertexShader = `
attribute float lineOpacity;
varying float vOpacity;
void main() {
  vOpacity = lineOpacity;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const blendedShader = `
uniform vec3 color;
varying float vOpacity;
void main() {
  gl_FragColor = vec4(color, vOpacity);
  #include <colorspace_fragment>
}
`;

const mixedShader = `
uniform vec3 color;
uniform vec3 background;
varying float vOpacity;
void main() {
  gl_FragColor = vec4(mix(background, color, vOpacity), 1.0);
  #include <colorspace_fragment>
}
`;

export class Segments {
  positions = new Float32Array(0);
  opacity = new Float32Array(0);
  count = 0;

  clear(): void {
    this.count = 0;
  }

  private reserve(vertices: number): void {
    if (this.count + vertices <= this.opacity.length) return;
    const capacity = Math.max(this.count + vertices, this.opacity.length * 2);
    const positions = new Float32Array(capacity * 3);
    positions.set(this.positions.subarray(0, this.count * 3));
    const opacity = new Float32Array(capacity);
    opacity.set(this.opacity.subarray(0, this.count));
    this.positions = positions;
    this.opacity = opacity;
  }

  add(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, opacity: number): void {
    this.reserve(2);
    const v = this.count;
    const p = this.positions;
    const at = v * 3;
    p[at] = x0;
    p[at + 1] = y0;
    p[at + 2] = z0;
    p[at + 3] = x1;
    p[at + 4] = y1;
    p[at + 5] = z1;
    this.opacity[v] = opacity;
    this.opacity[v + 1] = opacity;
    this.count += 2;
  }

  addPolyline(p: ArrayLike<number>, opacity: number): void {
    for (let i = 0; i + 5 < p.length; i += 3) this.add(p[i], p[i + 1], p[i + 2], p[i + 3], p[i + 4], p[i + 5], opacity);
  }

  addShifted(pairs: ArrayLike<number>, x: number, y: number, z: number, opacity: number): void {
    for (let i = 0; i + 5 < pairs.length; i += 6) this.add(pairs[i] + x, pairs[i + 1] + y, pairs[i + 2] + z, pairs[i + 3] + x, pairs[i + 4] + y, pairs[i + 5] + z, opacity);
  }
}

export class LineBatch {
  readonly object: LineSegments;
  private readonly geometry = new BufferGeometry();
  private readonly material: ShaderMaterial;
  private capacity = 0;

  constructor(color: LinearRgb, background?: LinearRgb, depthWrite = background !== undefined) {
    const mixed = background !== undefined;
    this.material = new ShaderMaterial({
      uniforms: { color: { value: new Color(...color) }, background: { value: new Color(...(background ?? color)) } },
      vertexShader,
      fragmentShader: mixed ? mixedShader : blendedShader,
      transparent: !mixed,
      depthWrite,
    });
    this.object = new LineSegments(this.geometry, this.material);
    this.object.frustumCulled = false;
  }

  setColor(color: LinearRgb, background?: LinearRgb): void {
    (this.material.uniforms.color.value as Color).setRGB(...color);
    if (background !== undefined) (this.material.uniforms.background.value as Color).setRGB(...background);
  }

  private ensure(vertices: number): void {
    if (vertices <= this.capacity && this.capacity > 0) return;
    this.capacity = Math.max(vertices, 2);
    this.geometry.setAttribute('position', new BufferAttribute(new Float32Array(this.capacity * 3), 3));
    this.geometry.setAttribute('lineOpacity', new BufferAttribute(new Float32Array(this.capacity), 1));
  }

  setSegments(parts: readonly Segments[]): void {
    let vertices = 0;
    for (const part of parts) vertices += part.count;
    this.ensure(vertices);
    const position = this.geometry.getAttribute('position') as BufferAttribute;
    const opacity = this.geometry.getAttribute('lineOpacity') as BufferAttribute;
    const positions = position.array as Float32Array;
    const opacities = opacity.array as Float32Array;
    let v = 0;
    for (const part of parts) {
      positions.set(part.positions.subarray(0, part.count * 3), v * 3);
      opacities.set(part.opacity.subarray(0, part.count), v);
      v += part.count;
    }
    this.geometry.setDrawRange(0, v);
    position.needsUpdate = true;
    opacity.needsUpdate = true;
  }

  setPolylines(lines: Polyline[]): void {
    const dashed = lines.map((line) => (line.dashed === true ? dashes(line.points) : undefined));
    let vertices = 0;
    lines.forEach((line, i) => (vertices += dashed[i] === undefined ? Math.max(0, line.points.length / 3 - 1) * 2 : dashed[i]!.length / 3));
    this.ensure(vertices);
    const position = this.geometry.getAttribute('position') as BufferAttribute;
    const opacity = this.geometry.getAttribute('lineOpacity') as BufferAttribute;
    const positions = position.array as Float32Array;
    let v = 0;
    lines.forEach((line, i) => {
      const pieces = dashed[i];
      if (pieces !== undefined) {
        positions.set(pieces, v * 3);
        for (let k = 0; k < pieces.length / 3; k++) opacity.setX(v + k, line.opacity);
        v += pieces.length / 3;
        return;
      }
      const p = line.points;
      for (let j = 0; j + 5 < p.length; j += 3) {
        for (let k = 0; k < 6; k++) positions[v * 3 + k] = p[j + k];
        opacity.setX(v, line.opacity);
        opacity.setX(v + 1, line.opacity);
        v += 2;
      }
    });
    this.geometry.setDrawRange(0, v);
    position.needsUpdate = true;
    opacity.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
