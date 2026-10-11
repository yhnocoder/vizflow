import { trailingSubscript } from '../explain/elem';

export const OP_TEX: Readonly<Record<string, { tex: string; def: string }>> = {
  sigmoid: { tex: '\\sigma', def: '\\sigma(v)=\\frac{1}{1+e^{-v}}' },
  relu: { tex: '\\mathrm{ReLU}', def: '\\mathrm{ReLU}(v)=\\max(0,v)' },
  gelu: { tex: '\\mathrm{GELU}', def: '\\mathrm{GELU}(v)=v\\,\\Phi(v)' },
  silu: { tex: '\\mathrm{SiLU}', def: '\\mathrm{SiLU}(v)=v\\,\\sigma(v)' },
};

export function opTex(op: string): string {
  return OP_TEX[op]?.tex ?? `\\mathrm{${op.replace(/_/g, '\\_')}}`;
}

function bold(letters: string): string {
  return `\\mathbf{${letters}}`;
}

export function defaultTex(name: string): string {
  const lowerDigits = /^([a-z])(\d+)$/.exec(name);
  if (lowerDigits !== null) return `${bold(lowerDigits[1])}_{${lowerDigits[2]}}`;
  const underscored = /^([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9]+)$/.exec(name);
  if (underscored !== null) {
    const sub = underscored[2];
    return addSubscript(defaultTex(underscored[1]), /^\d+$/.test(sub) || sub.length === 1 ? sub : `\\mathrm{${sub}}`);
  }
  if (/^[A-Z]$/.test(name)) return name;
  if (/^[A-Z][A-Z0-9]+$/.test(name)) return `\\mathrm{${name}}`;
  const upper = /^([A-Z])([A-Za-z0-9]+)$/.exec(name);
  if (upper !== null) return `${upper[1]}_{${upper[2]}}`;
  const lowerWord = /^([a-z]+)(\d*)$/.exec(name);
  if (lowerWord !== null) return lowerWord[2] === '' ? bold(lowerWord[1]) : `${bold(lowerWord[1])}_{${lowerWord[2]}}`;
  return `\\mathrm{${name.replace(/_/g, '\\_')}}`;
}

export function addSubscript(tex: string, sub: string): string {
  const merged = trailingSubscript(tex);
  if (merged !== undefined) return `${merged.base}_{${merged.inner},${sub}}`;
  return `${tex}_{${sub}}`;
}

export function addMark(tex: string, mark: string): string {
  if (mark.startsWith('_{') && mark.endsWith('}')) return addSubscript(tex, mark.slice(2, -1));
  if (/^_[^{]$/.test(mark)) return addSubscript(tex, mark.slice(1));
  if (/^[\^_]/.test(mark)) return `${tex}${mark}`;
  return addSubscript(tex, mark);
}

function hasTopLevelSubscript(tex: string): boolean {
  let depth = 0;
  for (let i = 0; i < tex.length; i++) {
    if (tex[i] === '{') depth++;
    else if (tex[i] === '}') depth--;
    else if (tex[i] === '_' && depth === 0 && tex[i - 1] !== '\\') return true;
  }
  return false;
}

export function elemOf(tex: string): string {
  const plain = tex.replace(/\\(?:mathbf|boldsymbol)\{([^{}]*)\}/g, (_m, inner: string) => (inner.length === 1 || inner.startsWith('\\') ? `{${inner}}` : `\\mathrm{${inner}}`));
  if (trailingSubscript(plain) !== undefined || !hasTopLevelSubscript(plain)) return plain;
  return `{${plain}}`;
}

export function numberTex(value: number): string {
  if (value === 0 || !Number.isFinite(value)) return String(value);
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  if (Math.abs(exponent) < 4) return String(value);
  const mantissa = Number((value / 10 ** exponent).toPrecision(12));
  const power = `10^{${exponent}}`;
  if (mantissa === 1) return power;
  if (mantissa === -1) return `-${power}`;
  return `${mantissa}\\times ${power}`;
}
