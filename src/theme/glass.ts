const svgNamespace = 'http://www.w3.org/2000/svg';
const glassSelector = '.vf-card, .vf-stepper, .vf-settings-section, .vf-settings-link, [data-vf-glass]';
const GLASS_BEZEL = 26;
const GLASS_SCALE = 64;
const GLASS_BLUR = 1.2;
const GLASS_SATURATE = 1.6;
const CHANNEL_SCALES = [1, 0.93, 0.86] as const;
const CHANNEL_KEEP = ['1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0', '0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0', '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0'] as const;

interface Lens {
  filter: SVGFilterElement | undefined;
  width: number;
  height: number;
  radius: number;
}

let svg: SVGSVGElement | undefined;
let mutations: MutationObserver | undefined;
let resizes: ResizeObserver | undefined;
let scanFrame = 0;
let filterCount = 0;
const lenses = new Map<HTMLElement, Lens>();

function refractionSupported(): boolean {
  const brands = (navigator as Navigator & { userAgentData?: { brands: { brand: string }[] } }).userAgentData?.brands ?? [];
  return brands.some((entry) => entry.brand === 'Chromium');
}

export function displacementMap(width: number, height: number, radius: number, bezel: number): Uint8ClampedArray<ArrayBuffer> {
  const data = new Uint8ClampedArray(width * height * 4);
  const halfWidth = width / 2;
  const halfHeight = height / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const px = x + 0.5 - halfWidth;
      const py = y + 0.5 - halfHeight;
      const qx = Math.abs(px) - (halfWidth - radius);
      const qy = Math.abs(py) - (halfHeight - radius);
      let inset: number;
      let nx = 0;
      let ny = 0;
      if (qx > 0 && qy > 0) {
        const length = Math.hypot(qx, qy);
        inset = radius - length;
        nx = (-Math.sign(px) * qx) / length;
        ny = (-Math.sign(py) * qy) / length;
      } else if (qx > qy) {
        inset = radius - qx;
        nx = -Math.sign(px);
      } else {
        inset = radius - qy;
        ny = -Math.sign(py);
      }
      const depth = Math.min(Math.max(inset / bezel, 0), 1);
      const strength = inset < 0 ? 0 : (1 - depth) ** 2;
      const i = (y * width + x) * 4;
      data[i] = Math.round(128 + strength * nx * 127);
      data[i + 1] = Math.round(128 + strength * ny * 127);
      data[i + 2] = 128;
      data[i + 3] = 255;
    }
  }
  return data;
}

function mapUrl(width: number, height: number, radius: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const bezel = Math.min(GLASS_BEZEL, width / 4, height / 4);
  canvas.getContext('2d')?.putImageData(new ImageData(displacementMap(width, height, Math.min(radius, width / 2, height / 2), bezel), width, height), 0, 0);
  return canvas.toDataURL();
}

function svgElement<K extends keyof SVGElementTagNameMap>(tag: K, attributes: Record<string, string | number>): SVGElementTagNameMap[K] {
  const el = document.createElementNS(svgNamespace, tag);
  for (const [name, value] of Object.entries(attributes)) el.setAttribute(name, String(value));
  return el;
}

function buildFilter(map: string, width: number, height: number): SVGFilterElement {
  const region = { x: 0, y: 0, width, height };
  const filter = svgElement('filter', { id: `vf-glass-${++filterCount}`, ...region, filterUnits: 'userSpaceOnUse', primitiveUnits: 'userSpaceOnUse', 'color-interpolation-filters': 'sRGB' });
  filter.append(
    svgElement('feGaussianBlur', { in: 'SourceGraphic', stdDeviation: GLASS_BLUR, result: 'soft' }),
    svgElement('feImage', { href: map, ...region, preserveAspectRatio: 'none', result: 'map' }),
  );
  CHANNEL_SCALES.forEach((factor, channel) => {
    filter.append(
      svgElement('feDisplacementMap', { in: 'soft', in2: 'map', scale: GLASS_SCALE * factor, xChannelSelector: 'R', yChannelSelector: 'G', result: `shift${channel}` }),
      svgElement('feColorMatrix', { in: `shift${channel}`, type: 'matrix', values: CHANNEL_KEEP[channel], result: `channel${channel}` }),
    );
  });
  filter.append(
    svgElement('feBlend', { in: 'channel0', in2: 'channel1', mode: 'screen', result: 'rg' }),
    svgElement('feBlend', { in: 'rg', in2: 'channel2', mode: 'screen', result: 'rgb' }),
    svgElement('feColorMatrix', { in: 'rgb', type: 'saturate', values: GLASS_SATURATE }),
  );
  return filter;
}

async function refract(el: HTMLElement): Promise<void> {
  const lens = lenses.get(el);
  if (lens === undefined) return;
  const width = Math.round(el.offsetWidth);
  const height = Math.round(el.offsetHeight);
  const radius = parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0;
  if (width < 2 || height < 2) return;
  if (width === lens.width && height === lens.height && radius === lens.radius) return;
  Object.assign(lens, { width, height, radius });
  const map = mapUrl(width, height, radius);
  const image = new Image();
  image.src = map;
  await image.decode().catch(() => undefined);
  if (svg === undefined || lenses.get(el) !== lens || lens.width !== width || lens.height !== height || lens.radius !== radius) return;
  const filter = buildFilter(map, width, height);
  svg.append(filter);
  el.style.setProperty('--vf-glass-filter', `url(#${filter.id})`);
  lens.filter?.remove();
  lens.filter = filter;
}

function release(el: HTMLElement): void {
  const lens = lenses.get(el);
  if (lens === undefined) return;
  lens.filter?.remove();
  resizes?.unobserve(el);
  el.style.removeProperty('--vf-glass-filter');
  lenses.delete(el);
}

function scan(): void {
  scanFrame = 0;
  if (svg === undefined) return;
  for (const el of [...lenses.keys()]) if (!el.isConnected || !el.matches(glassSelector)) release(el);
  for (const el of document.querySelectorAll<HTMLElement>(glassSelector)) {
    if (lenses.has(el)) continue;
    lenses.set(el, { filter: undefined, width: 0, height: 0, radius: -1 });
    resizes?.observe(el);
  }
}

function scheduleScan(): void {
  if (scanFrame === 0) scanFrame = requestAnimationFrame(scan);
}

function start(): void {
  svg = svgElement('svg', { 'aria-hidden': 'true', width: 0, height: 0 });
  svg.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden';
  document.body.append(svg);
  resizes = new ResizeObserver((entries) => entries.forEach((entry) => void refract(entry.target as HTMLElement)));
  mutations = new MutationObserver(scheduleScan);
  mutations.observe(document.body, { childList: true, subtree: true });
  scan();
}

function stop(): void {
  if (scanFrame !== 0) cancelAnimationFrame(scanFrame);
  scanFrame = 0;
  mutations?.disconnect();
  for (const el of [...lenses.keys()]) release(el);
  resizes?.disconnect();
  svg?.remove();
  svg = mutations = resizes = undefined;
}

export function syncGlass(on: boolean): void {
  const active = on && refractionSupported();
  document.documentElement.toggleAttribute('data-vf-refract', active);
  if (active && svg === undefined) start();
  if (!active && svg !== undefined) stop();
}
