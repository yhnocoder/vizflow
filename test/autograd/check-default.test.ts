import { afterEach, describe, expect, test, vi } from 'vitest';
import { graph } from '../../src/autograd';
import { checkByDefault as moduleCheckByDefault } from '../../src/autograd/check-default.module';
import { checkByDefault as standaloneCheckByDefault } from '../../src/autograd/check-default';

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe('finite-difference check default', () => {
  test('the module build turns the check on outside production', () => {
    vi.stubEnv('NODE_ENV', 'development');
    expect(moduleCheckByDefault()).toBe(true);
    vi.stubEnv('NODE_ENV', 'production');
    expect(moduleCheckByDefault()).toBe(false);
  });

  test('the module build turns the check off where process is not defined', () => {
    vi.stubGlobal('process', undefined);
    expect(moduleCheckByDefault()).toBe(false);
  });

  test('the standalone build keeps the check off', () => {
    vi.stubEnv('NODE_ENV', 'development');
    expect(standaloneCheckByDefault()).toBe(false);
  });

  test('an explicit check option wins over the default', () => {
    const read = (g: object) => (g as unknown as { check: boolean }).check;
    expect(read(graph())).toBe(false);
    expect(read(graph({ check: true }))).toBe(true);
    expect(read(graph({ check: false }))).toBe(false);
  });
});
