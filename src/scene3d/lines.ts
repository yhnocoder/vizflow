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

export class LineBatch {
  readonly object: LineSegments;
  private readonly geometry = new BufferGeometry();
  private readonly material: ShaderMaterial;
  private capacity = 0;

  constructor(color: LinearRgb, background?: LinearRgb) {
    const mixed = background !== undefined;
    this.material = new ShaderMaterial({
      uniforms: { color: { value: new Color(...color) }, background: { value: new Color(...(background ?? color)) } },
      vertexShader,
      fragmentShader: mixed ? mixedShader : blendedShader,
      transparent: !mixed,
      depthWrite: mixed,
    });
    this.object = new LineSegments(this.geometry, this.material);
    this.object.frustumCulled = false;
  }

  setColor(color: LinearRgb, background?: LinearRgb): void {
    (this.material.uniforms.color.value as Color).setRGB(...color);
    if (background !== undefined) (this.material.uniforms.background.value as Color).setRGB(...background);
  }

  setPolylines(lines: Polyline[]): void {
    const dashed = lines.map((line) => (line.dashed === true ? dashes(line.points) : undefined));
    let vertices = 0;
    lines.forEach((line, i) => (vertices += dashed[i] === undefined ? Math.max(0, line.points.length / 3 - 1) * 2 : dashed[i]!.length / 3));
    if (vertices > this.capacity || this.capacity === 0) {
      this.capacity = Math.max(vertices, 2);
      this.geometry.setAttribute('position', new BufferAttribute(new Float32Array(this.capacity * 3), 3));
      this.geometry.setAttribute('lineOpacity', new BufferAttribute(new Float32Array(this.capacity), 1));
    }
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
