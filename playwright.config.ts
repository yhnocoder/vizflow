import { createServer } from 'node:net';
import { defineConfig } from '@playwright/test';

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

process.env.VIZFLOW_E2E_PORT ??= String(await freePort());
const port = process.env.VIZFLOW_E2E_PORT;
const executablePath = process.env.VIZFLOW_CHROMIUM;

export default defineConfig({
  testDir: 'test/e2e',
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    launchOptions: executablePath ? { executablePath } : {},
  },
  webServer: {
    command: `npx vite --host 127.0.0.1 --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}/examples/index.html`,
    reuseExistingServer: false,
  },
});
