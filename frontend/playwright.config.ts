/** Playwright smoke suite. It runs against a real build served by `vite preview`, with the
 * API stubbed through `page.route`, so the tests assert on UI behaviour and never on a
 * particular server's numbers. Set PLAYWRIGHT_BASE_URL to point at a deployed site.
 *
 * Browsers fetch from cdn.playwright.dev, which is not reachable from some sandboxes; in
 * that case the suite is skipped by scripts/check_frontend.sh rather than reported green.
 */
import { defineConfig, devices } from '@playwright/test'

const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:4173'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  expect: { timeout: 7_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: process.env.PLAYWRIGHT_BASE_URL
    ? undefined
    : {
        command: 'npm run preview -- --port 4173 --strictPort',
        url: 'http://127.0.0.1:4173',
        reuseExistingServer: true,
        timeout: 60_000,
      },
})
