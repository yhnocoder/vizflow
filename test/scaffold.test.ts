import { describe, expect, it } from 'vitest';
import * as core from '../src/core';

describe('scaffold', () => {
  it('loads the core entry', () => {
    expect(typeof core).toBe('object');
  });
});
