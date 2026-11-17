import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * End-to-end tests: the real React build + the real FastAPI backend on the *test* database + a fake Groq API.
 * No real API key is used and the developer's database and .env secrets are never touched.
 *   npx playwright install chromium   (once)
 *   npm run e2e
 */
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const PY = process.env.E2E_PYTHON
  ?? (process.platform === "win32" ? path.join(ROOT, ".venv", "Scripts", "python.exe") : path.join(ROOT, ".venv", "bin", "python"));
export const E2E_DB = process.env.E2E_DATABASE_URL ?? "postgresql://postgres:postgres@localhost:5433/carwale_test";
export const PORTS = { fake: 18081, api: 18000, web: 14173 };

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1, // the fake LLM is shared state
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  outputDir: "test-results",
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }],
             ["json", { outputFile: "test-results/e2e-results.json" }]],
  globalSetup: "./e2e/global-setup.ts",
  use: {
    baseURL: `http://127.0.0.1:${PORTS.web}`,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    { name: "desktop-chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } },
      testIgnore: /mobile\.spec\.ts/ },
    { name: "mobile-chromium", use: { ...devices["Pixel 7"] }, testMatch: /(chat-flow|mobile)\.spec\.ts/ },
  ],
  webServer: [
    {
      command: `"${PY}" -m tests.fakes.fake_groq --port ${PORTS.fake}`,
      cwd: ROOT,
      url: `http://127.0.0.1:${PORTS.fake}/health`,
      timeout: 60_000,
      reuseExistingServer: false,
    },
    {
      command: `"${PY}" run_local.py ${PORTS.api}`,
      cwd: path.join(ROOT, "backend"),
      url: `http://127.0.0.1:${PORTS.api}/api/health`,
      timeout: 180_000,
      reuseExistingServer: false,
      env: {
        DATABASE_URL: E2E_DB,
        GROQ_API_KEY: "e2e-fake-key-not-real",
        GROQ_BASE_URL: `http://127.0.0.1:${PORTS.fake}`,
        CORS_ORIGINS: `http://127.0.0.1:${PORTS.web},http://localhost:${PORTS.web}`,
        RATE_LIMIT_PER_MIN: "1000",
        RATE_LIMIT_PER_DAY: "100000",
        FORCE_FAIL_MODELS: "",
      },
    },
    {
      command: `npx vite build --outDir e2e-dist --emptyOutDir && npx vite preview --outDir e2e-dist --port ${PORTS.web} --strictPort --host 127.0.0.1`,
      cwd: HERE,
      url: `http://127.0.0.1:${PORTS.web}`,
      timeout: 180_000,
      reuseExistingServer: false,
      env: { VITE_API_BASE_URL: `http://127.0.0.1:${PORTS.api}` },
    },
  ],
});
