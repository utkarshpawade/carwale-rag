import AxeBuilder from "@axe-core/playwright";
import { answers, ask, expect, openApp, test, uniq } from "./helpers";

test.describe("accessibility", () => {
  test("axe-core: no violations except the known colour-contrast issue (A-01)", async ({ page }) => {
    await openApp(page);
    await ask(page, uniq("Creta vs Seltos"));
    await answers(page, 1);
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .disableRules(["color-contrast"]).analyze();
    test.info().annotations.push({ type: "axe violations", description: JSON.stringify(results.violations.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.length }))) });
    expect(results.violations.map((v) => v.id)).toEqual([]);
  });

  test("axe-core: text colour contrast meets WCAG AA (A-01, reported - design decision)", async ({ page }) => {
    test.fail(true, "A-01: several 11 px grey labels are below 4.5:1; changing the palette is a design decision");
    await openApp(page);
    await ask(page, uniq("Creta vs Seltos"));
    await answers(page, 1);
    const results = await new AxeBuilder({ page }).withRules(["color-contrast"]).analyze();
    test.info().annotations.push({ type: "contrast nodes", description: String(results.violations[0]?.nodes.length ?? 0) });
    for (const n of results.violations[0]?.nodes ?? []) {
      const d = (n.any[0]?.data ?? {}) as { contrastRatio?: number; fgColor?: string; bgColor?: string; fontSize?: string };
      test.info().annotations.push({ type: "contrast", description:
        `${n.target.join(" ")} | ${d.contrastRatio}:1 ${d.fgColor} on ${d.bgColor} ${d.fontSize} | ${n.html.slice(0, 90)}` });
    }
    expect(results.violations).toEqual([]);
  });

  test("keyboard only: compose, send and reopen a past chat", async ({ page }) => {
    await openApp(page);
    await page.getByRole("textbox").focus();
    await page.keyboard.type(uniq("Is the Tata Punch safe?"));
    await page.keyboard.press("Enter");
    await answers(page, 1);
    await page.getByRole("button", { name: "New chat" }).first().focus();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { name: "Ask anything about new cars in India" })).toBeVisible();
    // B-16: the recent-chat rows must be reachable with Tab and open with Enter
    const row = page.getByRole("button", { name: /Is the Tata Punch safe/ });
    await row.focus();
    await expect(row).toBeFocused();
    await page.keyboard.press("Enter");
    await answers(page, 1);
  });

  test("the delete button becomes visible when focused with the keyboard", async ({ page }) => {
    await openApp(page);
    await ask(page, uniq("Mileage of Maruti Swift"));
    await answers(page, 1);
    const del = page.getByRole("button", { name: "Delete chat" }).first();
    await del.focus();
    await expect(del).toBeFocused();
    expect(Number(await del.evaluate((el) => getComputedStyle(el).opacity))).toBeGreaterThan(0.5);
  });
});
