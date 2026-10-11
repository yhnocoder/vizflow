import { EXACT_LIMIT } from './tensor';

export function defaultDigits(v: number): number {
  const a = Math.abs(v);
  if (a >= 100) return 0;
  if (a >= 10) return 1;
  if (a >= 1) return 2;
  return 3;
}

export function formatNumber(v: number, options: { digits?: number; tex?: boolean } = {}): string {
  if (v === Infinity || v === -Infinity) {
    const sign = v < 0 ? (options.tex ? '-' : '−') : '';
    return `${sign}${options.tex ? '\\infty' : '∞'}`;
  }
  if (Math.abs(v) > EXACT_LIMIT) {
    const [mantissa, exponent] = v.toExponential(options.digits ?? 2).split('e');
    const power = Number(exponent);
    return options.tex ? `${mantissa}\\times 10^{${power}}` : `${mantissa}e${power}`.replace('-', '−');
  }
  let s = v.toFixed(options.digits ?? defaultDigits(v));
  if (/^-0\.?0*$/.test(s)) s = s.slice(1);
  return options.tex ? s : s.replace('-', '−');
}
