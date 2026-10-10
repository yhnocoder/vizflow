export function formatNumber(v: number, options: { digits?: number; tex?: boolean } = {}): string {
  const a = Math.abs(v);
  let s: string;
  if (options.digits !== undefined) s = v.toFixed(options.digits);
  else if (a >= 100) s = v.toFixed(0);
  else if (a >= 10) s = v.toFixed(1);
  else if (a >= 1) s = v.toFixed(2);
  else s = v.toFixed(3);
  if (/^-0\.?0*$/.test(s)) s = s.slice(1);
  return options.tex ? s : s.replace('-', '−');
}
