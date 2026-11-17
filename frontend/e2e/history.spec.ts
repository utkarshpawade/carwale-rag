import { API, answers, ask, expect, fakeLLM, openApp, test, uniq } from "./helpers";

test.describe("history, refresh, navigation and tabs (U6-U8)", () => {
  test("refresh restores the open conversation", async ({ page }) => {
    await openApp(page);
    const q = uniq("Is the Tata Punch safe?");
    await ask(page, q);
    await answers(page, 1);
    await page.reload();
    await expect(page.getByRole("main").getByText(q, { exact: true })).toBeVisible();
    await expect(page.locator(".answer").first()).toContainText("Tata Punch");
    await answers(page, 1); // restored answer has its id, so feedback is still possible
  });

  test("refresh in the middle of an answer recovers cleanly", async ({ page, request }) => {
    await fakeLLM(request, { token_delay_ms: 250 });
    await openApp(page);
    const q = uniq("Creta vs Seltos");
    await ask(page, q);
    await expect(page.locator(".answer").first()).toContainText("Here is", { timeout: 20_000 }); // streaming started
    await page.reload();
    await expect(page.getByRole("main").getByText(q, { exact: true })).toBeVisible(); // the question survived the reload
    await fakeLLM(request, { token_delay_ms: 0 });
    await ask(page, uniq("Is the Tata Punch safe?"));
    await expect(page.locator(".answer").last()).toContainText("Tata Punch");
    // record what the server kept for the interrupted turn (see report B-22)
    const convId = await page.evaluate(() => localStorage.getItem("cw_conv_id"));
    const userId = await page.evaluate(() => localStorage.getItem("cw_user_id"));
    const conv = await (await request.get(`${API}/api/conversations/${convId}?user_id=${userId}`)).json();
    test.info().annotations.push({ type: "persisted roles after mid-stream reload",
                                   description: conv.messages.map((m: { role: string }) => m.role).join(",") });
  });

  test("new chat, then reopen the old one from the sidebar", async ({ page }) => {
    await openApp(page);
    const q = uniq("Mileage of Maruti Swift");
    await ask(page, q);
    await answers(page, 1);
    await page.getByRole("button", { name: "New chat" }).first().click();
    await expect(page.getByRole("heading", { name: "Ask anything about new cars in India" })).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem("cw_conv_id"))).toBeNull();
    const row = page.getByText(q.slice(0, 40), { exact: false }).first();
    await expect(row).toBeVisible();
    await row.click();
    await expect(page.getByRole("main").getByText(q, { exact: true })).toBeVisible();
    await answers(page, 1);
  });

  test("delete a chat from the sidebar", async ({ page }) => {
    await openApp(page);
    const q = uniq("Creta vs Seltos");
    await ask(page, q);
    await answers(page, 1);
    const sidebarRow = page.locator("div.group").filter({ hasText: q.slice(0, 30) });
    await expect(sidebarRow).toHaveCount(1);
    await sidebarRow.hover();
    await sidebarRow.getByRole("button", { name: "Delete chat" }).click();
    await expect(sidebarRow).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Ask anything about new cars in India" })).toBeVisible(); // it was open
  });

  test("leaving and coming back with the back button restores the chat", async ({ page }) => {
    await openApp(page);
    const q = uniq("Is the Tata Punch safe?");
    await ask(page, q);
    await answers(page, 1);
    await page.goto("about:blank");
    await page.goBack();
    await expect(page.getByRole("main").getByText(q, { exact: true })).toBeVisible({ timeout: 30_000 });
  });

  test("two tabs of the same user share history and can both send", async ({ context }) => {
    const a = await context.newPage();
    await openApp(a);
    const q1 = uniq("Creta vs Seltos");
    await ask(a, q1);
    await answers(a, 1);
    const b = await context.newPage();
    await openApp(b); // same localStorage: same user and the same open conversation
    await expect(b.getByRole("main").getByText(q1, { exact: true })).toBeVisible();
    await Promise.all([ask(a, uniq("Is the Tata Punch safe?")), ask(b, uniq("Mileage of Maruti Swift"))]);
    await answers(a, 2);
    await answers(b, 2);
    await a.reload();
    await answers(a, 3); // after a reload, tab A sees every turn, including tab B's
  });

  test("two users chatting at the same time see only their own chats", async ({ browser }) => {
    const [ca, cb] = [await browser.newContext(), await browser.newContext()];
    const [a, b] = [await ca.newPage(), await cb.newPage()];
    await Promise.all([openApp(a), openApp(b)]);
    const qa = uniq("Creta vs Seltos"), qb = uniq("Mileage of Maruti Swift");
    await Promise.all([ask(a, qa), ask(b, qb)]);
    await Promise.all([answers(a, 1), answers(b, 1)]);
    await a.reload();
    await b.reload();
    await expect(a.getByText(qa.slice(0, 30)).first()).toBeVisible();
    await expect(a.getByText(qb.slice(0, 30))).toHaveCount(0);
    await expect(b.getByText(qa.slice(0, 30))).toHaveCount(0);
    await ca.close();
    await cb.close();
  });
});
