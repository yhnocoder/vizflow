function range(start: number, end: number, step: number): number[] {
  const out: number[] = [];
  for (let x = start; x < end; x += step) out.push(x);
  return out;
}

export function vllmCaptureSizes(maxSeqs: number, budget: number): number[] {
  const max = Math.min(maxSeqs * 2, 512);
  const sizes = [1, 2, 4, ...range(8, 256, 8), ...range(256, max + 1, 16)].filter((x) => x <= max);
  if (budget <= max && !sizes.includes(budget)) sizes.push(budget);
  return sizes.sort((a, b) => a - b);
}

export function sglangDecodeSizes(maxBs: number): number[] {
  const sizes = [1, 2, 4, 8, 12, ...range(16, 257, 8), ...range(272, 512, 16), ...range(512, maxBs + 1, 32)].filter((x) => x <= maxBs);
  if (!sizes.includes(maxBs)) sizes.push(maxBs);
  return sizes;
}

export function sglangPrefillSizes(maxTokens: number): number[] {
  return [...range(4, 33, 4), ...range(48, 257, 16), ...range(288, 513, 32), ...range(576, 1025, 64), ...range(1280, 4097, 256), ...range(4608, 65537, 512)].filter((x) => x <= maxTokens);
}

export function padTo(sizes: number[], n: number): number | null {
  return sizes.find((x) => x >= n) ?? null;
}
