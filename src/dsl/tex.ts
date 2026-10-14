import { trailingSubscript } from '../explain';

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
  if (mark.startsWith('^{') && mark.endsWith('}')) return addSuperscript(tex, mark.slice(2, -1));
  if (/^[\^_]/.test(mark)) return `${tex}${mark}`;
  return addSubscript(tex, mark);
}

function hasTopLevelSuperscript(tex: string): boolean {
  let depth = 0;
  for (let i = 0; i < tex.length; i++) {
    const c = tex[i];
    if (c === '\\') i++;
    else if (c === '{') depth++;
    else if (c === '}') depth--;
    else if (depth === 0 && (c === '^' || c === "'")) return true;
  }
  return false;
}

const TOP_LEVEL_OPERATOR_COMMANDS = ['\\cdot', '\\times', '\\pm', '\\mp'];

function hasTopLevelOperator(tex: string): boolean {
  let depth = 0;
  for (let i = 0; i < tex.length; i++) {
    const c = tex[i];
    if (c === '\\') {
      if (depth === 0 && TOP_LEVEL_OPERATOR_COMMANDS.some((command) => tex.startsWith(command, i) && !/[A-Za-z]/.test(tex[i + command.length] ?? ''))) return true;
      i++;
    } else if (c === '{') depth++;
    else if (c === '}') depth--;
    else if (depth === 0 && (c === '+' || c === '-' || c === '=' || c === ',')) return true;
  }
  return false;
}

function addSuperscript(tex: string, sup: string): string {
  if (hasTopLevelOperator(tex)) return `\\left(${tex}\\right)^{${sup}}`;
  if (tex.endsWith('}')) {
    let depth = 0;
    for (let i = tex.length - 1; i >= 0; i--) {
      if (tex[i] === '}' && tex[i - 1] !== '\\') depth++;
      else if (tex[i] === '{' && tex[i - 1] !== '\\' && --depth === 0) {
        if (tex[i - 1] === '^') return `${tex.slice(0, i)}{${tex.slice(i + 1, -1)},${sup}}`;
        break;
      }
    }
  }
  if (hasTopLevelSuperscript(tex)) return `{${tex}}^{${sup}}`;
  return `${tex}^{${sup}}`;
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

function bracedArgument(tex: string, open: number): number {
  if (tex[open] !== '{') throw new Error('\\step needs two braced arguments: \\step{id}{...}');
  let depth = 0;
  for (let i = open; i < tex.length; i++) {
    if (tex[i] === '\\') i++;
    else if (tex[i] === '{') depth++;
    else if (tex[i] === '}' && --depth === 0) return i;
  }
  throw new Error('\\step has an unclosed brace');
}

export function markSteps(tex: string, mark: (id: string) => string): string {
  const pattern = /\\step(?![A-Za-z])\s*/g;
  let out = '';
  let from = 0;
  for (let found = pattern.exec(tex); found !== null; found = pattern.exec(tex)) {
    const idOpen = found.index + found[0].length;
    const idClose = bracedArgument(tex, idOpen);
    let bodyOpen = idClose + 1;
    while (/\s/.test(tex[bodyOpen] ?? '')) bodyOpen++;
    const bodyClose = bracedArgument(tex, bodyOpen);
    const id = tex.slice(idOpen + 1, idClose).trim();
    const body = markSteps(tex.slice(bodyOpen + 1, bodyClose), mark);
    out += `${tex.slice(from, found.index)}\\htmlData{vf-step=${mark(id)}}{${body}}`;
    from = bodyClose + 1;
    pattern.lastIndex = from;
  }
  return out + tex.slice(from);
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
