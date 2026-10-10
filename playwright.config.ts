import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'test/e2e',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    launchOptions: { executablePath: process.env.VIZFLOW_CHROMIUM ?? '/opt/pw-browsers/chromium' },
  },
  webServer: {
    command: 'npx vite --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173/examples/index.html',
    reuseExistingServer: false,
  },
});
