import { BufferAttribute, BufferGeometry, Color, LineSegments, ShaderMaterial } from 'three';
import type { LinearRgb } from '../theme';
import { DASH_GAP, DASH_LENGTH } from './style';

export interface Polyline {
  points: ArrayLike<number>;
  opacity: number;
  dashed: boolean;
}

const vertexShader = `
attribute float lineOpacity;
attribute float dashed;
attribute float along;
varying float vOpacity;
varying float vDashed;
varying float vAlong;
void main() {
  vOpacity = lineOpacity;
  vDashed = dashed;
  vAlong = along;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = `
uniform vec3 color;
varying float vOpacity;
varying float vDashed;
varying float vAlong;
void main() {
  if (vDashed > 0.5 && mod(vAlong, ${(DASH_LENGTH + DASH_GAP).toFixed(3)}) > ${DASH_LENGTH.toFixed(3)}) discard;
  gl_FragColor = vec4(color, vOpacity);
  #include <colorspace_fragment>
}
`;

export class LineBatch {
  readonly object: LineSegments;
  private readonly geometry = new BufferGeometry();
  private readonly material: ShaderMaterial;
  private capacity = 0;

  constructor(color: LinearRgb) {
    this.material = new ShaderMaterial({
      uniforms: { color: { value: new Color(...color) } },
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
    });
    this.object = new LineSegments(this.geometry, this.material);
    this.object.frustumCulled = false;
  }

  setColor(color: LinearRgb): void {
    (this.material.uniforms.color.value as Color).setRGB(...color);
  }

  setPolylines(lines: Polyline[]): void {
    let vertices = 0;
    for (const line of lines) vertices += Math.max(0, line.points.length / 3 - 1) * 2;
    if (vertices > this.capacity || this.capacity === 0) {
      this.capacity = Math.max(vertices, 2);
      this.geometry.setAttribute('position', new BufferAttribute(new Float32Array(this.capacity * 3), 3));
      this.geometry.setAttribute('lineOpacity', new BufferAttribute(new Float32Array(this.capacity), 1));
      this.geometry.setAttribute('dashed', new BufferAttribute(new Float32Array(this.capacity), 1));
      this.geometry.setAttribute('along', new BufferAttribute(new Float32Array(this.capacity), 1));
    }
    const position = this.geometry.getAttribute('position') as BufferAttribute;
    const opacity = this.geometry.getAttribute('lineOpacity') as BufferAttribute;
    const dashed = this.geometry.getAttribute('dashed') as BufferAttribute;
    const along = this.geometry.getAttribute('along') as BufferAttribute;
    const positions = position.array as Float32Array;
    let v = 0;
    for (const line of lines) {
      const p = line.points;
      let distance = 0;
      for (let i = 0; i + 5 < p.length; i += 3) {
        const next = distance + Math.hypot(p[i + 3] - p[i], p[i + 4] - p[i + 1], p[i + 5] - p[i + 2]);
        positions[v * 3] = p[i];
        positions[v * 3 + 1] = p[i + 1];
        positions[v * 3 + 2] = p[i + 2];
        positions[v * 3 + 3] = p[i + 3];
        positions[v * 3 + 4] = p[i + 4];
        positions[v * 3 + 5] = p[i + 5];
        opacity.setX(v, line.opacity);
        opacity.setX(v + 1, line.opacity);
        dashed.setX(v, line.dashed ? 1 : 0);
        dashed.setX(v + 1, line.dashed ? 1 : 0);
        along.setX(v, distance);
        along.setX(v + 1, next);
        distance = next;
        v += 2;
      }
    }
    this.geometry.setDrawRange(0, v);
    position.needsUpdate = true;
    opacity.needsUpdate = true;
    dashed.needsUpdate = true;
    along.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
