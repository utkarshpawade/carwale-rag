import { answers, ask, assistantText, expect, openApp, test, uniq } from "./helpers";

test.describe("core chat flows (U1-U5)", () => {
  test("first visit: the server wakes, stats and starters appear", async ({ page }) => {
    await openApp(page);
    await expect(page.getByText(/I search 288 models and 2,202 variants/)).toBeVisible();
    await expect(page.getByRole("heading", { name: "Ask anything about new cars in India" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Smart hybrid cars|Creta vs Seltos|Mileage of Maruti Swift/ })).toHaveCount(3);
    expect(await page.evaluate(() => localStorage.getItem("cw_user_id"))).toMatch(/^[0-9a-f-]{36}$/);
  });

  test("a starter question streams a full, grounded answer", async ({ page }) => {
    await openApp(page);
    await page.getByRole("button", { name: "Smart hybrid cars under 30 lakhs" }).click();
    await expect(page.getByText("Smart hybrid cars under 30 lakhs", { exact: true }).first()).toBeVisible(); // user bubble
    await answers(page, 1);
    await expect(assistantText(page).first()).toContainText("Here is what CarWale's data says about");
    expect(await page.getByRole("link", { name: /View on CarWale/ }).count()).toBeGreaterThan(0); // car cards
    await expect(page.getByText("Sources (CarWale)")).toBeVisible();
    await expect(page.getByRole("button", { name: "Which one has the best mileage?" })).toBeVisible();
    await expect(page.getByRole("button", { name: "How I answered this" })).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem("cw_conv_id"))).toMatch(/^[0-9a-f-]{36}$/);
  });

  test("a follow-up chip continues the conversation using memory", async ({ page }) => {
    await openApp(page);
    await ask(page, uniq("smart hybrid cars under 30 lakhs"));
    await answers(page, 1);
    await page.getByRole("button", { name: "Which one has the best mileage?" }).click();
    await answers(page, 2);
    await page.getByRole("button", { name: "How I answered this" }).nth(1).click();
    const panel = page.locator("aside").filter({ hasText: "How this answer was built" });
    await expect(panel.getByText("2 past messages")).toBeVisible();
    await expect(panel.getByText("Active cars:")).toBeVisible();
    await expect(panel.getByText("refers to: all")).toBeVisible();
  });

  test("typing: Shift+Enter makes a new line, Enter sends, input is cleared", async ({ page }) => {
    await openApp(page);
    const box = page.getByRole("textbox");
    await box.fill("Creta vs Seltos");
    await box.press("Shift+Enter");
    await box.pressSequentially("diesel automatic");
    await expect(box).toHaveValue("Creta vs Seltos\ndiesel automatic");
    await box.press("Enter");
    await expect(box).toHaveValue("");
    await answers(page, 1);
    await expect(page.getByText(/Creta vs Seltos\s+diesel automatic/).first()).toBeVisible();
  });

  test("trace panel: open, raw JSON, close", async ({ page }) => {
    await openApp(page);
    await ask(page, uniq("Is the Tata Punch safe?"));
    await answers(page, 1);
    await page.getByRole("button", { name: "How I answered this" }).click();
    const panel = page.locator("aside").filter({ hasText: "How this answer was built" });
    await expect(panel.getByText("car_details")).toBeVisible();
    await panel.getByText("Show raw trace JSON").click();
    await expect(panel.locator("pre")).toContainText('"route": "car_details"');
    await panel.getByRole("button", { name: "Close trace" }).click();
    await expect(panel).toHaveCount(0);
  });

  test("feedback and copy", async ({ page, context, browserName }) => {
    test.skip(browserName !== "chromium", "clipboard permissions are chromium-only here");
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await openApp(page);
    await ask(page, uniq("Mileage of Maruti Swift"));
    await answers(page, 1);
    const fb = page.waitForResponse((r) => r.url().endsWith("/api/feedback") && r.request().method() === "POST");
    await page.getByRole("button", { name: "Helpful", exact: true }).click();
    expect((await fb).status()).toBe(200);
    await expect(page.getByRole("button", { name: "Helpful", exact: true })).toHaveClass(/text-emerald-600/);
    await page.getByRole("button", { name: /^Copy$/ }).click();
    await expect(page.getByRole("button", { name: /Copied/ })).toBeVisible();
    const clip = await page.evaluate(() => navigator.clipboard.readText());
    expect(clip).toContain("Here is what CarWale's data says about");
  });

  test("a repeated first question is served from the answer cache", async ({ browser }) => {
    const q = uniq("Creta vs Seltos");
    for (const expectCached of [false, true]) {
      const ctx = await browser.newContext(); // a different (new) user each time
      const page = await ctx.newPage();
      await openApp(page);
      await ask(page, q);
      await answers(page, 1);
      await expect(page.getByText("⚡ cached")).toHaveCount(expectCached ? 1 : 0);
      await ctx.close();
    }
  });
});
