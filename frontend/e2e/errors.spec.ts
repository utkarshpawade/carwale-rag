import { answers, ask, expect, fakeLLM, openApp, test, uniq } from "./helpers";

const sse = (events: [string, unknown][]) => events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("");

test.describe("errors, outages and slow networks (U9)", () => {
  test("sleeping server: waking banner, disabled input, then recovery", async ({ page }) => {
    let failures = 2;
    await page.route("**/api/health", (route) => (failures-- > 0 ? route.abort() : route.continue()));
    await page.goto("/");
    await expect(page.getByText(/Waking up the server/)).toBeVisible();
    await expect(page.getByRole("textbox")).toHaveAttribute("placeholder", "Connecting to the server…");
    await expect(page.getByRole("button", { name: "Smart hybrid cars under 30 lakhs" })).toBeDisabled();
    await expect(page.getByRole("button", { name: "Smart hybrid cars under 30 lakhs" })).toBeEnabled({ timeout: 20_000 });
    await expect(page.getByText(/Waking up the server/)).toHaveCount(0);
  });

  test("server that never wakes is reported as unavailable", async ({ page }) => {
    await page.clock.install();
    await page.route("**/api/health", (route) => route.abort());
    await page.goto("/");
    await expect(page.getByText(/Waking up the server/)).toBeVisible();
    await page.clock.fastForward("02:40");
    await expect(page.getByText("Server unavailable")).toBeVisible();
  });

  test("AI provider at its rate limit: retries visibly, then a clear message; app stays usable", async ({ page, request }) => {
    // a short retry-after: the backend's per-model cooldown is process-wide, and a long one would leak into later tests
    await fakeLLM(request, { rules: [{ action: "status", status: 429, retry_after: "2", kind: "planner" }] });
    await openApp(page);
    await ask(page, uniq("Creta vs Seltos"));
    await expect(page.getByText(/free-tier limit reached — retrying in 2s/)).toBeVisible();
    await expect(page.getByText("The AI service is at its free-tier limit right now. Please try again in about 2 seconds."))
      .toBeVisible({ timeout: 20_000 });
    await expect(page.getByRole("textbox")).toBeEnabled();
  });

  test("AI provider key rejected: says it is misconfigured", async ({ page, request }) => {
    await fakeLLM(request, { rules: [{ action: "status", status: 401 }] });
    await openApp(page);
    await ask(page, uniq("Creta vs Seltos"));
    await expect(page.getByText(/not configured correctly/)).toBeVisible();
  });

  test("our own rate limit (HTTP 429) is shown to the user", async ({ page }) => {
    await page.route("**/api/chat", (route) => route.fulfill({ status: 429, contentType: "application/json",
      body: JSON.stringify({ error: "Too many requests. Please retry in 42 seconds." }) }));
    await openApp(page);
    await ask(page, "Creta vs Seltos");
    await expect(page.getByText("Too many requests. Please retry in 42 seconds.")).toBeVisible();
  });

  test("server error event mid-answer is shown", async ({ page }) => {
    await page.route("**/api/chat", (route) => route.fulfill({ status: 200, contentType: "text/event-stream",
      body: sse([["conversation", { conversation_id: crypto.randomUUID() }], ["token", { text: "Partial " }],
                 ["error", { message: "Something went wrong while answering. Please try again." }]]) }));
    await openApp(page);
    await ask(page, "Creta vs Seltos");
    await expect(page.getByText("Something went wrong while answering. Please try again.")).toBeVisible();
    await expect(page.getByText("Partial")).toBeVisible();
  });

  test("a stream that ends without 'done' is flagged as interrupted", async ({ page }) => {
    // B-23: a connection closed by a proxy mid-answer left a truncated answer that looked complete
    await page.route("**/api/chat", (route) => route.fulfill({ status: 200, contentType: "text/event-stream",
      body: sse([["conversation", { conversation_id: crypto.randomUUID() }], ["token", { text: "The Creta is" }]]) }));
    await openApp(page);
    await ask(page, "Creta vs Seltos");
    await expect(page.getByText("The Creta is")).toBeVisible();
    await expect(page.getByRole("button", { name: "Send" })).toBeVisible(); // streaming ended
    await expect(page.getByText(/interrupted/i)).toBeVisible();
  });

  test("Stop keeps the partial answer", async ({ page, request }) => {
    await fakeLLM(request, { token_delay_ms: 300 });
    await openApp(page);
    await ask(page, uniq("Creta vs Seltos"));
    await expect(page.locator(".answer").first()).toContainText("Here is", { timeout: 20_000 });
    await page.getByRole("button", { name: "Stop" }).click();
    await expect(page.getByText("(stopped)")).toBeVisible();
    await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
  });

  test("works on a slow 3G-like network", async ({ page }) => {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Network.enable");
    await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 600, downloadThroughput: 60 * 1024,
                                                         uploadThroughput: 30 * 1024 });
    await openApp(page);
    await ask(page, uniq("Is the Tata Punch safe?"));
    await answers(page, 1);
    await expect(page.locator(".answer").first()).toContainText("Tata Punch");
  });

  test("going offline shows an error; coming back online works", async ({ page, context }) => {
    await openApp(page);
    await context.setOffline(true);
    await ask(page, "Creta vs Seltos");
    await expect(page.getByText(/Failed to fetch|NetworkError|network/i)).toBeVisible();
    await context.setOffline(false);
    await ask(page, uniq("Mileage of Maruti Swift"));
    await answers(page, 1);
  });

  test("hostile or huge input is handled", async ({ page }) => {
    await openApp(page);
    await page.getByRole("textbox").fill("x".repeat(5000));
    expect((await page.getByRole("textbox").inputValue()).length).toBe(1000);
    await page.getByRole("textbox").fill("   ");
    await expect(page.getByRole("button", { name: "Send" })).toBeDisabled();
    await ask(page, "क्रेटा का माइलेज कितना है? 🚗 '; DROP TABLE models; --");
    await answers(page, 1);
  });
});
