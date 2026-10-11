export type Token = number | string

const FNV_OFFSET = 0x811c9dc5
const FNV_PRIME = 0x01000193

function fnv1a(text: string): string {
  let h = FNV_OFFSET
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, FNV_PRIME) >>> 0
  }
  return h.toString(16).padStart(8, '0')
}

export const NONE_HASH = fnv1a('vllm-none-hash')

export function hashBlockTokens(
  parentHash: string | null,
  tokens: readonly Token[],
  extraKeys: readonly string[] | null,
): string {
  return fnv1a(JSON.stringify([parentHash ?? NONE_HASH, tokens, extraKeys]))
}

export function blockExtraKeys(startTokenIndex: number, cacheSalt: string | undefined): string[] | null {
  return startTokenIndex === 0 && cacheSalt ? [`cache_salt:${cacheSalt}`] : null
}

export function chainBlockHashes(
  tokens: readonly Token[],
  hashBlockSize: number,
  cacheSalt: string | undefined,
  known: readonly string[] = [],
): string[] {
  const hashes = [...known]
  let start = hashes.length * hashBlockSize
  while (start + hashBlockSize <= tokens.length) {
    const parent = hashes.length > 0 ? hashes[hashes.length - 1]! : null
    hashes.push(hashBlockTokens(parent, tokens.slice(start, start + hashBlockSize), blockExtraKeys(start, cacheSalt)))
    start += hashBlockSize
  }
  return hashes
}
