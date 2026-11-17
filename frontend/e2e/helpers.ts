import { expect, type APIRequestContext, type Page, test as base } from "@playwright/test";

export const FAKE = "http://127.0.0.1:18081";
export const API = "http://127.0.0.1:18000";

/** Unique wording keeps a question out of the server's first-turn answer cache. */
let seq = 0;
export const uniq = (q: string) => `${q} (run ${Date.now() % 100000}-${seq++})`;

export async function fakeLLM(request: APIRequestContext, cfg: Record<string, unknown>) {
  const r = await request.post(`${FAKE}/__control/configure`, { data: cfg });
  expect(r.ok()).toBeTruthy();
}

export const test = base.extend<{ resetFake: void }>({
  resetFake: [async ({ request }, use) => {
    await request.post(`${FAKE}/__control/reset`);
    await use();
    await request.post(`${FAKE}/__control/reset`);
  }, { auto: true }],
});
export { expect };

/** Opens the app and waits until the backend is reachable (starter buttons become clickable). */
export async function openApp(page: Page) {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Smart hybrid cars under 30 lakhs" })).toBeEnabled({ timeout: 30_000 });
}

export async function ask(page: Page, text: string) {
  const box = page.getByRole("textbox");
  await box.fill(text);
  await box.press("Enter");
}

/** Waits until `n` assistant answers have finished (each finished answer gets a 'Helpful' button). */
export async function answers(page: Page, n: number) {
  await expect(page.getByRole("button", { name: "Helpful", exact: true })).toHaveCount(n, { timeout: 30_000 });
}

export const assistantText = (page: Page) => page.locator(".answer");
