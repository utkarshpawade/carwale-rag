import { answers, ask, expect, openApp, test, uniq } from "./helpers";

test.describe("mobile layout (U10)", () => {
  test("the sidebar is a drawer: menu opens it, the backdrop closes it", async ({ page }) => {
    await openApp(page);
    const drawerNew = page.getByRole("button", { name: "New chat" }).filter({ hasText: "New chat" });
    await expect(drawerNew).not.toBeInViewport();
    await page.getByRole("button", { name: "Open menu" }).click();
    await expect(drawerNew).toBeInViewport();
    await page.mouse.click(390, 400); // the dimmed backdrop to the right of the 288 px drawer
    await expect(drawerNew).not.toBeInViewport();
  });

  test("opening a past chat from the drawer closes the drawer", async ({ page }) => {
    await openApp(page);
    const q = uniq("Mileage of Maruti Swift");
    await ask(page, q);
    await answers(page, 1);
    await page.locator("header").getByRole("button", { name: "New chat" }).click(); // the header (+) button on mobile
    await page.getByRole("button", { name: "Open menu" }).click();
    await page.getByText(q.slice(0, 25)).first().click();
    await expect(page.getByRole("button", { name: "Close menu" })).not.toBeInViewport();
    await expect(page.getByRole("main").getByText(q, { exact: true })).toBeVisible();
  });

  test("no horizontal page scroll, even with a wide comparison table", async ({ page }) => {
    await openApp(page);
    await ask(page, uniq("smart hybrid cars under 30 lakhs"));
    await answers(page, 1);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });

  test("the trace opens full screen and closes", async ({ page }) => {
    await openApp(page);
    await ask(page, uniq("Is the Tata Punch safe?"));
    await answers(page, 1);
    await page.getByRole("button", { name: "How I answered this" }).click();
    const panel = page.locator("aside").filter({ hasText: "How this answer was built" });
    const box = await panel.boundingBox();
    expect(box?.width).toBeGreaterThanOrEqual((page.viewportSize()?.width ?? 0) - 1);
    await panel.getByRole("button", { name: "Close trace" }).click();
    await expect(panel).toHaveCount(0);
  });
});
