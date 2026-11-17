import { answers, ask, expect, fakeLLM, openApp, test, uniq } from "./helpers";

test.describe("security in the browser", () => {
  test("HTML typed by the user is shown as text and never executed", async ({ page }) => {
    await openApp(page);
    const payload = `<img src=x onerror="window.__xss=1"><script>window.__xss=2</script>`;
    await ask(page, payload);
    await answers(page, 1);
    await expect(page.getByRole("main").getByText(payload, { exact: true })).toBeVisible();
    expect(await page.locator('img[src="x"]').count()).toBe(0);
    expect(await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)).toBeUndefined();
  });

  test("HTML and javascript: links in the model's answer are neutralised", async ({ page, request }) => {
    await fakeLLM(request, { answer_override:
      'Answer <img src=x onerror="window.__xss=3"> <script>window.__xss=4</script> [click me](javascript:window.__xss=5) ' +
      "[ok](https://www.carwale.com/) [1]" });
    await openApp(page);
    await ask(page, uniq("Creta vs Seltos"));
    await answers(page, 1);
    const answer = page.locator(".answer").first();
    await expect(answer).toContainText("Answer");
    expect(await answer.locator("img, script").count()).toBe(0);
    for (const href of await answer.locator("a").evaluateAll((as) => as.map((a) => a.getAttribute("href") || ""))) {
      expect(href.toLowerCase()).not.toContain("javascript:");
    }
    await answer.getByText("click me").click();
    expect(await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)).toBeUndefined();
  });

  test("external links open in a new tab without a referrer", async ({ page }) => {
    await openApp(page);
    await ask(page, uniq("Creta vs Seltos"));
    await answers(page, 1);
    const links = page.locator('a[href^="https://www.carwale.com"]');
    expect(await links.count()).toBeGreaterThan(0);
    for (const rel of await links.evaluateAll((as) => as.map((a) => `${a.getAttribute("target")}|${a.getAttribute("rel")}`))) {
      expect(rel).toBe("_blank|noreferrer");
    }
  });

  test("another user's conversation id in localStorage is refused, not shown", async ({ browser }) => {
    const owner = await browser.newContext();
    const p1 = await owner.newPage();
    await openApp(p1);
    const q = uniq("Is the Tata Punch safe?");
    await ask(p1, q);
    await answers(p1, 1);
    const stolen = await p1.evaluate(() => localStorage.getItem("cw_conv_id"));
    const intruder = await browser.newContext();
    await intruder.addInitScript((id) => localStorage.setItem("cw_conv_id", id as string), stolen);
    const p2 = await intruder.newPage();
    await openApp(p2);
    await expect(p2.getByText(q)).toHaveCount(0);
    await owner.close();
    await intruder.close();
  });
});
