# Testing Report — CarWale AI Assistant

> **Status: complete** (17 Nov 2026). Every number below comes from a run I watched finish. Nothing was committed;
> all changes are in the working tree for you to review.

## Progress

- [x] Phase 1 — Understand the project
- [x] Phase 2 — Make sure it runs (install, lint, type check, build, start)
- [x] Phase 3 — Test plan (existing suite run first: 48/48 passed)
- [x] Phase 4 — Unit tests (Python + React)
- [x] Phase 5 — Integration and API tests (real Postgres, real `groq` SDK, fake Groq server)
- [x] Phase 6 — End-to-end tests (Playwright, desktop + mobile)
- [x] Phase 7 — Bug hunting beyond the tests (security, errors, performance, code smells, accessibility)
- [x] Phase 8 — Fix bugs (each with a failing test first, then the fix, then a full re-run)
- [x] Phase 9 — Final report (sections 4–11 below)

## Results at a glance

| Suite | Tests | Passed | Failed | Notes |
|---|---|---|---|---|
| Python unit (pytest) | 518 | **518** | 0 | 48 pre-existing + 470 new |
| Python integration (pytest) | 117 | **116** | 0 | + 1 *expected failure* (`xfail`, strict) that documents unfixed S-01 |
| Frontend unit (Vitest) | 66 | **66** | 0 | |
| End-to-end (Playwright) | 44 | **43** | 0 | + 1 *expected failure* that documents unfixed A-01 (colour contrast); 33 desktop + 11 mobile |
| **Total** | **745** | **743** | **0** | 2 expected failures for known, reported, unfixed issues |

- **Coverage:** backend app **94.8 %** (1,884 statements), frontend **94.9 %** statements / **97.4 %** lines. Python overall
  71.7 %, pulled down by the crawler and loader scripts in `ingestion/`, which do network and file I/O (41.6 %).
- **Bugs:** 25 functional bugs (2 High, 12 Medium, 11 Low), 9 security findings (1 High), 3 accessibility, 2 performance.
  **22 bugs fixed** (one of them, B-10, partly), each with a test that failed before the fix and passes after. No Critical
  issues.
- **Verdict:** a solid, well-designed demo that is now much more robust. It is **not ready for a public production launch**
  until S-01 (rate-limit bypass), S-02 (no real auth) and S-03 (CORS) are decided, and a real-model evaluation is run.
  See section 11.

---

## 1. Project summary (Phase 1)

### What the app does

A retrieval-augmented (RAG) chatbot for people buying a new car in India. It answers questions about prices, mileage, specs,
features, safety, warranty and owner opinions, and handles comparisons and open searches such as "smart hybrid cars under
30 lakhs". All answers are grounded in a snapshot of CarWale's public catalog. The bot remembers the conversation, so
follow-ups like "which one has the best mileage?" work.

Each chat turn works like this. One LLM call turns the message into a validated JSON *plan*. Deterministic code then runs
that plan: parameterized SQL over the structured catalog, plus hybrid vector and full-text search over text chunks. A second
LLM call writes a streamed, cited answer from the retrieved context. The LLM never writes SQL.

### Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI 0.163, uvicorn, Server-Sent Events (SSE) streaming |
| LLM | Groq API (`openai/gpt-oss-120b`, `gpt-oss-20b`, `qwen/qwen3.8-27b`) through the `groq` SDK, with per-role fallback chains |
| Database | PostgreSQL 17 + pgvector 0.8 (Docker locally, Neon in production), psycopg 3 async pool |
| Embeddings | fastembed `BAAI/bge-small-en-v1.5` (ONNX, 384 dimensions), run locally on CPU |
| Fuzzy matching | rapidfuzz |
| Frontend | React 19, TypeScript 5.9, Vite 8, Tailwind 4, react-markdown + remark-gfm |
| Ingestion | httpx crawler with a gzip disk cache, a parser for the JSON state embedded in CarWale pages, chunker, loader |
| Deploy | Render (Docker, `render.yaml`) for the API, Vercel for the UI |

### How to install, run and build (from the README, verified)

- Database: `docker run … pgvector/pgvector:pg17` on port 5433 (the container `carwale-pg` already exists on this machine).
- Python: `uv venv --python 3.12 .venv`, then `uv pip install -r backend/requirements.txt …`.
- Backend: `cd backend && python run_local.py`. On Windows this works around psycopg's event-loop requirement. Port 8000.
- Frontend: `cd frontend && npm install && npm run dev` (port 5173). Build: `npm run build`. Type check: `npm run typecheck`.
- Ingestion (offline, about 30 minutes): `python -m ingestion.discover`, then `fetch`, `parse`, `build_chunks`, `load_db`.

### Folder structure

```
backend/app/   main.py (FastAPI app, CORS) · config.py (settings) · bootstrap.py (singletons) · embeddings.py
               api/chat.py (POST /api/chat, rate limiter) · api/routes.py (health, meta, conversations, feedback)
               db/ (async pool, schema.sql) · memory/store.py (history + working state + answer cache)
               llm/client.py (Groq wrapper, fallbacks, token budgeting) · llm/prompts.py
               rag/ schemas (QueryPlan) · entities (car-name resolver) · car_search (SQL search) · specs (fact sheets)
                    hybrid_search (vector + FTS + RRF) · context (token budget) · followups · orchestrator (the pipeline)
backend/cli.py · run_local.py · Dockerfile         frontend/src/ App · api · types · components/*
ingestion/     discover · discover_more · fetch · parse · normalize · build_chunks · load_db · common
eval/          golden.json (32 cases) · run_eval.py      tests/  test_core.py (48 unit tests)
```

### Database schema (`backend/app/db/schema.sql`)

| Table | Purpose |
|---|---|
| `models` | One row per car model: catalog facts plus a `data` JSONB of extra structured details |
| `variants` | One row per version, linked to `models` (on delete cascade): price, fuel, transmission, mileage and other specs, feature tags |
| `chunks` | Retrieval text with a `vector(384)` embedding and a generated `tsvector` for full-text search |
| `conversations` | Chat metadata: `id` (UUID), `user_id`, `title`, timestamps |
| `messages` | Messages of a conversation (on delete cascade). `meta` JSONB holds the plan, cars, sources, trace and digest |
| `conversation_state` | Working memory per conversation: summary, active cars, focus car, filters, preferences |
| `answer_cache` | Cached first-turn answers, valid for 7 days |
| `feedback` | Thumbs up/down per message (no foreign key) |
| `meta` | Key/value pairs, e.g. `data_snapshot_date` |

Local data: 400 models (288 launched, 112 upcoming), 2,283 variants and 30,602 embedded chunks. The README says 25,200
chunks, so the README is out of date.

### Every API endpoint

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/` | none | Service name and links |
| GET | `/api/health` | none | `{status, models, variants, chunks, data_snapshot}` (cached at startup; does not check the database) |
| GET | `/api/meta` | none | Stats, starter questions, fallback starters (not used by the frontend) |
| POST | `/api/chat` | `{user_id, conversation_id?, message}` | SSE stream of events: `conversation`, `status`, `trace`, `cars`, `sources`, `token`, `followups`, `done`, `error`. Returns 429 when rate-limited, 400 for a bad conversation ID, 403 for another user's conversation |
| GET | `/api/conversations?user_id=` | `user_id` | The user's 30 most recent chats |
| GET | `/api/conversations/{id}?user_id=` | UUID, `user_id` | Full history, or 400 / 403 / 404 |
| DELETE | `/api/conversations/{id}?user_id=` | UUID, `user_id` | `{deleted: true}`, or 404 (also for another user's chat) |
| POST | `/api/feedback` | `{message_id, rating -1..1, comment?}` | `{ok: true}` |
| GET | `/docs`, `/redoc`, `/openapi.json` | none | FastAPI's auto-generated docs (public) |

### Every page and screen

The app is a single page with no router. It has these areas:

1. **Empty state**: title, catalog stats and 6 starter questions.
2. **Chat thread**: user bubbles; assistant bubbles with a streaming status, Markdown answer, citation links, car cards, a
   source list, "How I answered this", Copy, thumbs up/down, a "⚡ cached" badge and follow-up chips.
3. **Composer**: auto-growing textarea (1,000 character limit). Enter sends, Shift+Enter adds a new line, and a Stop button
   shows while an answer streams.
4. **Sidebar**: New chat, recent chats (open or delete) and data credits. It is fixed on desktop and a drawer with a
   backdrop on mobile.
5. **Trace drawer**: the 5 pipeline steps (memory, plan, resolve, retrieve, generate) and the raw JSON.
6. **Banners**: "Waking up the server…" while the health check retries, then a "Server unavailable" badge after 150 seconds.

### Every user flow

There is **no sign-up or login**. The browser generates a random `user_id` (UUID) on its first visit and keeps it in
`localStorage`. That ID is the only credential.

| # | Flow |
|---|---|
| U1 | Open the app → health ping → "Waking server…" → "Online" → starters become clickable |
| U2 | Ask a question (starter or typed) → status text → streamed answer → car cards → sources → follow-ups |
| U3 | Follow-up in the same chat (chip or typed) → memory resolves "it", "these" or "the second one" |
| U4 | Use an answer: open the trace drawer, toggle raw JSON, copy the answer, thumbs up/down, click a citation or car card (opens CarWale) |
| U5 | Stop an answer while it streams |
| U6 | Refresh the page → the current chat is restored from `localStorage` and the server |
| U7 | New chat → empty state; the earlier chat appears in the sidebar → open it → its history loads |
| U8 | Delete a chat from the sidebar (the active chat resets to the empty state) |
| U9 | Error paths: server asleep or down, rate limited (429), AI service busy or misconfigured, network drop |
| U10 | Mobile: open the menu → drawer → open a chat → the drawer closes; the trace opens full-screen |

### Assumptions (where something was unclear)

- **No authentication, by design.** Wherever the brief says "auth", I tested the equivalent: a missing, malformed or
  someone else's `user_id`. "Wrong or expired token" doesn't apply because there are no tokens.
- **No real-time collaboration.** SSE streaming is per request, with no websockets or shared sessions. "Multi-user sync"
  was tested as two users chatting at once (isolation) and one user in two tabs (shared history).
- **The real Groq key is never used**, as the rules require. Integration and E2E tests run against a **fake Groq server**
  (`tests/fakes/fake_groq.py`) that speaks the same HTTP API, including streaming, `429`, `401` and `5xx`. The real
  `groq` SDK and the whole pipeline are exercised. The quality of real LLM answers is not tested here (see `eval/`).
- **Live crawling is not re-run.** It would hit carwale.com. The ingestion code is unit-tested on synthetic fixtures and on
  the existing processed data.
- **The dev database is never modified.** Tests use `carwale_test`, a clone of the local `carwale` database with every
  conversation, feedback and cache row removed, so no real chat history is touched.

---

## 2. Make sure it runs (Phase 2)

| Step | Command | Result |
|---|---|---|
| Postgres | `docker start carwale-pg` | ✅ Up (pg 17.12, pgvector 0.8.7). Test DB `carwale_test` created as a clone |
| Python deps | existing `.venv` (Python 3.12.13) + `pytest-cov` | ✅ All imports resolve |
| Frontend deps | `npm ci` | ✅ 143 packages. `npm audit`: 2 **moderate** advisories in `postcss-selector-parser`, pulled in by the build-time-only `@tailwindcss/typography` (not shipped to browsers) |
| Existing tests | `python -m pytest tests -q` | ✅ **48 passed** in 0.8 s |
| TS type check | `npm run typecheck` | ✅ 0 errors |
| Production build | `npm run build` | ✅ JS 411.8 kB (126.7 kB gzipped), CSS 50.0 kB (8.8 kB gzipped), built in 0.2 s. No secrets in the bundle |
| Python lint | `uvx ruff check backend ingestion eval tests` | ⚠️ 31 style findings (29 × E702 semicolons, 2 × E741 variable named `l`). **No pyflakes errors** (no unused imports, undefined names or redefinitions) |
| Python lint (bug rules) | `ruff --select B,ASYNC,S,PL` | ⚠️ 22 findings, reviewed by hand: 6 × S608 "SQL built from strings" (all whitelisted fragments plus bound parameters, so not injectable — see Phase 7), S110 `try/except/pass` (silent summarizer failures), 4 × B904, 3 × S324 (SHA-1 used for cache keys, not security; fine) |
| Python types | `uvx mypy backend/app` | ⚠️ 100 errors. 67 are one false-positive pattern (`plan = None` before the retry loop; the loop always returns or sets it). The other 33 are loose typing (untyped dicts, `Literal` narrowing). None is a real runtime defect |
| Frontend lint | ESLint 9 + typescript-eslint + react-hooks + jsx-a11y (temporary config; the project has none) | ⚠️ 0 TypeScript or React-hooks issues. **4 accessibility errors**: the recent-chat rows (`Sidebar.tsx:42`) and the mobile backdrop (`App.tsx:137`) are clickable `div`s with no keyboard support |
| Start backend | `run_local.py` on the test DB with a fake key | ✅ Starts in ~1 s. `/`, `/api/health`, `/api/meta` and `/docs` return 200 |
| Start frontend | `vite` | ✅ Ready in 0.3 s; serves `index.html` and modules |
| Chat with the LLM unreachable | `POST /api/chat` with Groq unreachable | ❌ **Bug B-03**: the user waits ~39 s and is then told *"free-tier limit reached"*, which is the wrong diagnosis |

---

## 3. Test plan (Phase 3)

**Tools:** pytest with anyio, httpx `ASGITransport` and pytest-cov for Python; Vitest with Testing Library and jsdom for
frontend units; Playwright (Chromium, desktop and mobile profiles) with axe-core for end-to-end and accessibility tests.
A fake Groq server stands in for the LLM provider everywhere.

Legend: **U** = unit, **I** = integration (real Postgres, fake LLM, real `groq` SDK), **E** = end-to-end in a real browser.

| Feature | Happy path | Wrong / empty / huge input | Missing or foreign "auth" (`user_id`) | Network / server failure |
|---|---|---|---|---|
| Chat pipeline `POST /api/chat` | I: event order and shape, DB rows written, memory saved; E: full flows U1–U5 | I: missing fields, wrong types, empty, whitespace-only, 4,001 chars, over 1,000 (truncated), NUL bytes, unicode/emoji/Hinglish, SQL- and XSS-looking text | I: missing / short / bad-character `user_id`, posting into another user's conversation → 403, malformed `conversation_id` → 400 | I: LLM 429 (fallback, retry, give up), 401, 5xx, connection error, stream cut mid-way; DB down |
| Plan schema (`QueryPlan`) | U: valid plan | U: nulls, missing keys, odd casing, lakh/crore text, "Rs." prefixes, NaN/∞, wrong types | none | U: invalid JSON triggers repair, then fallback |
| Car-name resolver | U: aliases, renames, typos | U: empty, generic words, brand only, unicode, very long, punctuation | none | none |
| Structured search (+ upcoming) | I: every filter respected, sorting, relaxation | I: impossible filters, min > max, unknown tags | none | none |
| Fact sheets, comparisons, trims, city prices | U: formatting functions; I: real models | U: empty variant lists, zero, negative, missing fields | none | none |
| Hybrid retrieval | I: filters applied before ranking, per-model cap, `k` limit | U/I: tsquery sanitising, non-ASCII-only query, empty sections | none | none |
| Context builder | U: budget respected, numbering | U: no blocks, no chunks, huge chunks | none | none |
| Memory store and summary | I: CRUD, cascade delete, state round trip, rolling summary | I: unknown IDs | I: isolation between users | I: summarizer failure is swallowed |
| Answer cache | I: second identical first question is a cache hit with 0 LLM calls | I: city questions are not cached | none | none |
| LLM client | U: header parsing, token budget, cooldown, candidate selection; I: fallback chain | U: odd header formats | I: bad API key → clear error | I: 429 / 5xx / timeout / mid-stream failure |
| Rate limiter | U/I: under the limit OK, over the limit 429 with a wait | U: many distinct clients (memory) | none | I: spoofed `X-Forwarded-For` |
| Conversations list / get / delete | I: CRUD and ordering, limit of 30; E: U6–U8 | I: bad UUID, empty `user_id` | I: another user's chat → 403/404, nothing leaked or deleted | I: DB down → 500 |
| Feedback | I: stored row; E: thumbs | I: rating out of range, wrong types, 1,001-character comment, huge ID, NUL bytes | none (no ownership check — reported) | E: network error shows no unhandled rejection |
| Health / meta | I: shape | none | none | E: health failing → waking banner, recovery, "Server unavailable" |
| Frontend rendering | U: Markdown citations, cards, trace, sidebar, composer; E: all flows | U/E: raw HTML, `javascript:` links, tables, long text | none | U/E: SSE split across chunks, error JSON, abort |
| Responsive and accessibility | E: desktop 1280×800 and mobile (Pixel 7) | E: no horizontal scroll at phone width (412 px) | none | E: axe-core scan, keyboard-only use |
| Ingestion (offline) | U: normalisers, parsers, chunk splitter, upsert SQL | U: `None`, empty, malformed numbers and dates, unicode | none | U: truncated gzip cache file |

Also covered: refreshing in the middle of a flow, the back button, two tabs, two users at once, slow network (throttled
CDP), and screenshots, traces and videos kept for every E2E failure.

---

## 4. What I built to test it

| Piece | Purpose |
|---|---|
| `tests/conftest.py` (extended) | Test isolation, set before any app import. It never reads `.env` (which holds live keys), forces a fake API key, points `GROQ_BASE_URL` at a closed local port so an accidental real call fails fast, and uses only the `*_test` database |
| `tests/fakes/fake_groq.py` | A **fake Groq API**. In pytest it plugs into the real `groq` SDK as a custom httpx transport; for E2E it runs as a real HTTP server. It provides a rule-based planner, answers built only from the context it was sent, streaming, and failure injection: 429 with `retry-after`, 401, 5xx, timeout, connection error, a stream cut after N chunks, an in-stream error event, invalid JSON, an empty answer, slow tokens |
| `tests/fakes/reset_test_db.py` | Empties the per-user tables before E2E runs, and **refuses any database whose name doesn't end in `_test`** |
| `tests/integration/conftest.py` | Runs the real FastAPI app in-process (httpx `ASGITransport`) on the real test DB with the real embedder. It also refuses non-`_test` databases and skips cleanly if the DB is down |
| `frontend/vitest.config.ts`, `src/test/*` | Vitest + Testing Library + jsdom, plus a helper that catches unhandled promise rejections |
| `frontend/playwright.config.ts`, `frontend/e2e/*` | Starts the fake Groq server, the backend (test DB, fake key) and a **production build** of the UI, then runs every flow on desktop Chrome and a Pixel 7 profile. Screenshots, traces and videos are kept for failures |
| `requirements-dev.txt`, npm scripts | `npm test`, `npm run test:coverage`, `npm run e2e`, `npm run e2e:report` |

The test database `carwale_test` is a clone of your local `carwale` database (same 400 models, 2,283 variants and 30,602
embedded chunks) with **all conversations, messages, feedback and cache rows removed**. Your dev database was not
modified: it still has its 196 conversations, checked after the last run.

## 5. Results by phase

### Phase 4: unit tests (518 Python + 66 React, all passing)

| File | Tests | What it covers |
|---|---|---|
| `test_core.py` (existing) | 48 | Normalisers, resolver on real data, SQL whitelisting, plan schema, feature/trim helpers |
| `test_unit_schemas.py` | 105 | The tolerant `QueryPlan`: prices in INR/lakh/crore/"Rs.", enum synonyms, nulls, wrong types, NaN/∞, unicode, strict schema |
| `test_unit_entities.py` | 52 | Resolver on a synthetic catalog: aliases, renames, typos, ambiguity, brand-only, hostile and very long input, free-text scans |
| `test_unit_specs.py` | 58 | Fact sheets, prices, powertrains, crash rating, city on-road prices, feature/trim tables, feature detection |
| `test_unit_context.py` | 27 | Token budget, source labels, follow-up chips, tsquery sanitising |
| `test_unit_llm_client.py` | 24 | Rate-limit header parsing, token budget, cooldowns, candidate selection |
| `test_unit_api.py` | 47 | Rate limiter (minute/day windows, memory bounds), SSE framing, client IP, request validation |
| `test_unit_orchestrator.py` | 45 | Filter refinement, cities, cue regexes, memory references ("it", "these", "the second one"), backstops |
| `test_unit_ingestion.py` | 112 | Normalisers, HTML/JSON-state parsing, mileage tables, review merging, chunk splitter, loader SQL |
| Vitest: `api`, `App`, `Markdown`, `Composer`, `Sidebar`, `MessageView`, `CarCards`, `TracePanel` | 66 | SSE parser (split chunks, CRLF, bad frames, every HTTP error), App lifecycle (wake-up, retries, give-up after 150 s, streaming, stop, restore, delete, races), Markdown safety, keyboard behaviour |

### Phase 5: integration and API tests (116 passing + 1 expected failure)

| File | Tests | What it covers |
|---|---|---|
| `test_api_chat.py` | 53 | SSE event order and shape; DB rows written; memory over 3 turns; compare / details / feature / city / upcoming / unknown-car / glossary / chit-chat / out-of-scope routes; answer cache (0 LLM calls on a hit); validation; hostile input (SQL, XSS, RTL override, prompt injection); another user's conversation (403); rate limit; every LLM failure (429 fallback and cooldown, 500, timeout, 401, connection refused, invalid JSON repair, cut stream, empty answer, forced fail); rolling summary; DB outage; concurrent users; two tabs writing at once |
| `test_api_endpoints.py` | 28 | Health/meta/docs, conversation list/get/delete with ownership checks (foreign read is 403, foreign delete is 404 with no existence leak), cascade deletes, 30-chat cap, hostile `user_id`s, feedback validation, CORS preflight, DB outage (500 without leaking internals) |
| `test_db_layer.py` | 21 | Memory store CRUD and cache expiry; every structured-search filter checked against SQL; sorting; budget relaxation; brand/feature filters; upcoming search; hybrid retrieval filters, per-model cap, non-ASCII queries; pinned/titled/trim/brand chunks |
| `test_llm_contract.py` | 15 | `LLMClient` through the real SDK: fallback chains, cooldowns, token-budget skips, auth errors, JSON repair, streaming usage, failures mid-stream |

### Phase 6: end-to-end tests (44 tests, all with the expected outcome)

All user flows from section 1 (U1–U10), in a real Chromium browser against the real backend:

- first visit and wake-up, a starter question, follow-up chips with memory, Enter / Shift+Enter, the trace panel,
  feedback, copy, the answer cache
- refresh, refresh mid-answer, new chat and reopening it, delete, the back button
- two tabs of one user, two users at once
- sleeping server, server that never wakes, AI rate limit with visible retries, bad API key, our own HTTP 429, error
  event, stream cut without `done`, Stop, slow-3G throttling, offline then online, huge and hostile input
- XSS from the user and from the model, `javascript:` links, external links, a stolen conversation id
- mobile drawer, opening a chat from the drawer, no horizontal scroll, full-screen trace
- axe-core scan and keyboard-only use

The desktop flows also run on the mobile profile (11 tests). The **pre-fix** run had 12 failures. 3 were real bugs (B-16 ×2,
B-23) and 9 were mistakes in my own tests, listed in the appendix. The **post-fix** run: 44/44 with the expected outcome,
in 56 s.

### Phase 8: fix verification

Every fix followed the same cycle: a test that failed because of the bug, then the fix, then the test passing. After the
last fix I re-ran the **whole** suite: Python 634 passed + 1 xfailed, Vitest 66/66, Playwright 44/44, `tsc` clean, the
production build clean, and ruff showing no new findings.

Resolver changes (B-10/B-11) carry extra risk, so I also compared the old and new resolver on **2,400 generated mentions**
(name, full name, "Name Brand", slug, "new … 2026", typo) for all 400 models. Exact top-1 matches went from 2,148 to 2,219:
71 fixed and **0 regressions**.

## 6. Coverage

| Area | Statements | Covered |
|---|---|---|
| `backend/app` (the API and RAG pipeline) | 1,884 | **94.8 %** |
| `ingestion/` (offline crawler, parser, loader) | 1,444 | 41.6 % |
| Python total | 3,328 | 71.7 % |
| Frontend `src/` | n/a | **94.9 %** statements · 90.5 % branches · 89.8 % functions · **97.4 %** lines |

What's not covered: in the backend, `bootstrap.build_pipeline` and `main.lifespan` (tests build the pipeline directly; the E2E
run exercises both). In ingestion, `fetch.py`, `discover*.py`, `load_db.main` and `build_chunks.model_chunks`, which need the
network or the full data set. Their helper functions are covered.

## 7. Bugs found

Severity scale: **Critical** = data loss or a security breach for all users · **High** = common failure or crash ·
**Medium** = wrong results or a real user/ops impact · **Low** = cosmetic or edge case.

### Functional bugs

| ID | Severity | What was wrong | Where | Status |
|---|---|---|---|---|
| B-01 | Medium | Upcoming-car search: any fuel filter also matched every non-EV. "Upcoming diesel cars" returned 97 models; only 4 list diesel | `rag/car_search.py` `search_upcoming` | ✅ Fixed |
| B-02 | Medium | Blank or NUL-byte chat messages were accepted. Whitespace-only ran the whole pipeline on an empty question (planner, retrieval, 9 answer calls, then a misleading error); a NUL byte crashed the turn because Postgres TEXT can't store `\x00` | `api/chat.py` `ChatIn` | ✅ Fixed (422 for blank; NUL stripped) |
| B-03 | Medium | When the LLM was down, unreachable or returning 5xx, users waited 20–39 s and were told *"free-tier limit reached"* | `llm/client.py`, `rag/orchestrator.py` | ✅ Fixed (accurate message, short retries, under 10 s) |
| B-04 | Medium | A planner price of `"Rs. 15 lakh"` invalidated the whole plan, because the "." of "Rs." was read as the number. `"Rs.15 Lakh"` silently became ₹15,000. The turn failed after 3 retries | `rag/schemas.py` `_price` | ✅ Fixed |
| B-05 | **High** | Planner output with a list or dict where a number belongs, or JSON `Infinity`, raised `TypeError`/`OverflowError` past validation and the client, **crashing the turn** instead of repairing it or falling back | `rag/schemas.py`, `llm/client.py` | ✅ Fixed |
| B-06 | Low | `[null]` inside planner lists became the string `"None"` (leaking into keywords, aspects and brands) | `rag/schemas.py` `_str_list` | ✅ Fixed |
| B-07 | Medium | Feature detection used plain substrings: "Toy**ota**" read as OTA updates, "Urban **Cruise**r" as cruise control, "hill d**esc**ent" as ESC. A phantom *"OTA updates: not available on any trim"* block went into the context of **every Toyota price/spec question** | `rag/specs.py` `detect_features` | ✅ Fixed (word boundaries) |
| B-08 | Low | Source chips were garbled: glossary chips read "GlossaryADAS · ADAS", and list and brand chips repeated their title | `rag/context.py` `chunk_label` | ✅ Fixed |
| B-09 | **High** | A dropped connection or in-stream error from the LLM **after the answer started** escaped the client, because the SDK raises raw `httpx.ReadError`/`groq.APIError`. The user saw "Something went wrong", the partial answer was thrown away, and nothing was saved | `llm/client.py` `stream` | ✅ Fixed (partial answer kept, marked "cut short", saved) |
| B-10 | Low | The resolver's "prefer the upcoming model" logic for *upcoming/facelift/next* never ran (dead branch) | `rag/entities.py` `resolve` | ✅ Fixed for shared aliases (note in section 8) |
| B-11 | Medium | Reordered words picked a sibling model: "Creta Hyundai" became the Creta **Electric**, "Fortuner Toyota" the Fortuner **Legender**, "Hector MG" the Hector **Tomahawk EV**. 88 of 400 models were wrong in "Name Brand" order | `rag/entities.py` fuzzy scoring | ✅ Fixed (71 corrected, 0 regressions) |
| B-12 | Medium | Variant hints were matched as substrings: base trim **"E"** matched every name containing an "e" (P**e**trol, Di**e**sel, **E**X), and "S" matched "SX". "Creta E price" answered with the whole model's range | `rag/specs.py` `filter_variants` | ✅ Fixed |
| B-13 | Medium | The rate limiter kept one entry per client forever. Memory was unbounded, and with spoofable IPs (S-01) that is a cheap way to exhaust a 512 MB instance | `api/chat.py` `RateLimiter` | ✅ Fixed (idle sweep + 100k cap) |
| B-14 | Low | `POST /api/feedback` returned **HTTP 500** for `message_id ≥ 2^63` and for NUL bytes, and accepted ids ≤ 0 | `api/routes.py` `FeedbackIn` | ✅ Fixed (422 / stripped) |
| B-15 | Low | `/api/health` answers "ok" from stats cached at startup, even when the database is down, so Render's health check can't notice | `api/routes.py` | ❌ Not fixed (section 8) |
| B-16 | Medium | Recent-chat rows were click-only `div`s, so **keyboard and screen-reader users could not open past chats**. The delete button stayed invisible when focused | `components/Sidebar.tsx` | ✅ Fixed |
| B-17 | Low | The Copy button showed "Copied" even when the clipboard refused, and the rejection went unhandled | `components/MessageView.tsx` | ✅ Fixed |
| B-18 | Low | The trace panel printed a stray "0" when the context was empty | `components/TracePanel.tsx` | ✅ Fixed |
| B-19 | Low | Failed delete/feedback requests caused unhandled promise rejections, and delete reset the open chat even when the server refused | `App.tsx` | ✅ Fixed |
| B-20 | Low | Auto-scroll on every token pulls the view to the bottom, so you can't scroll up to read while an answer streams | `App.tsx` scroll effect | ❌ Not fixed (UX change) |
| B-21 | Low | A model returning zero tokens left a blank answer bubble with no explanation | `llm/client.py` `stream` | ✅ Fixed (falls back to the next model) |
| B-22 | Low | Refreshing or closing the tab mid-answer cancels generation on the server: the answer is lost and the question stays unanswered in the history (E2E recorded `user, user, assistant`) | `api/chat.py` / orchestrator | ❌ Not fixed (section 8) |
| B-23 | Medium | A stream closed without `done` (proxy timeout, redeploy) left a truncated answer that **looked complete** | `App.tsx` `send` | ✅ Fixed (marked "interrupted") |
| B-24 | Medium | Opening two past chats quickly could show chat A's messages while chat B was selected, so follow-ups went to the wrong conversation | `App.tsx` `openConversation` | ✅ Fixed |
| B-25 | Medium | A first-turn answer that was cut short was **cached for 7 days** and replayed, truncated, to everyone who asked the same question | `rag/orchestrator.py` cache write | ✅ Fixed |

### Security, accessibility, performance and code-quality findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| S-01 | **High** | **Rate-limit bypass.** The client IP is taken from the first `X-Forwarded-For` entry, which the client controls, and uvicorn trusts every proxy (`--forwarded-allow-ips='*'`). Anyone can get unlimited use of the shared Groq key by changing that header | ❌ Not fixed; strict `xfail` test in place (section 8) |
| S-02 | Medium | No real authentication: the anonymous `user_id` is the only credential, and it travels in URL query strings (proxy/server logs, browser history). Whoever has it can read and delete those chats | ❌ Design decision |
| S-03 | Medium | CORS allows **every** `*.vercel.app` origin (regex), including sites anyone can deploy, so other sites can use the API through visitors' browsers | ❌ Config decision |
| S-04 | Low | No rate limit on feedback or conversation endpoints; feedback has no ownership or foreign-key check, so it can be spammed without limit | ❌ |
| S-05 | Low | `/docs`, `/redoc` and `/openapi.json` are public in production | ❌ |
| S-06 | Low | Markdown images in model output are rendered, so a remote image URL echoed from scraped text would load in users' browsers | ❌ |
| S-07 | Info | The local `.env` holds live credentials (Groq key, Neon URLs with passwords, Render and Vercel tokens). They were **never committed** (I searched all git history). Its own comment says to revoke them | Rotate |
| S-08 | Low | The Docker image runs as root | ❌ |
| S-09 | Low | `npm audit`: 2 moderate advisories in a build-only dependency (`postcss-selector-parser` via `@tailwindcss/typography`) | ❌ (no browser impact) |
| A-01 | Medium | 11 small grey labels fall below WCAG AA contrast (2.5–3.7:1 vs 4.5:1): sidebar meta and footer, "View on CarWale", "Sources", the Copy and How-I-answered buttons, the disclaimer | ❌ Design decision; documented by an expected-failure test |
| A-02 | Low | Streaming answers are not announced to screen readers (no `aria-live`) | ❌ |
| A-03 | Low | The drawers can't be closed with Escape, the backdrop is mouse-only, and focus isn't managed (2 remaining jsx-a11y lint errors) | ❌ |
| P-01 | Medium | Every streamed token re-renders and re-parses the Markdown of **every** message in the thread, so cost grows with conversation length | ❌ Needs a small refactor |
| P-02 | Low | Unfiltered vector search (85 ms) and full-text search (72 ms) are sequential scans of all 30,602 chunks. They only run on the general-question route. The README's "milliseconds at 14K rows" is out of date | ❌ |
| C-01 | — | Duplicated logic: starter questions ×3, `inr()`/`price_range` ×2, `robots()` ×2, `FEATURE_TAGS` ×2 (identical today, but they can drift) | Reported |
| C-02 | — | Dead code: `api.getMeta()` and `/api/meta` are unused by the UI; the strict-JSON-schema planner branch is unreachable (the planner always passes `example=`) | Reported |
| C-03 | — | `_maybe_summarize` swallows every exception without logging | Reported |
| C-04 | — | ruff: 29 style findings (semicolons, a variable named `l`); mypy: 100 typing errors, none of them runtime bugs; no ESLint config in the repo; no CI | Reported |
| C-05 | — | README drift: says 25,200 chunks (actually 30,602) and names a summary model the config no longer uses | Reported |

Checked and clean:
- **SQL injection:** every query is parameterized. The f-string SQL only joins whitelisted fragments, and the LLM never
  writes SQL. I reviewed all 6 ruff S608 warnings by hand and tested hostile `user_id`s and messages.
- **XSS:** React escapes all text, and react-markdown renders no raw HTML and strips `javascript:` URLs (unit and E2E tested).
- **CSRF:** not applicable (no cookies; JSON bodies require a CORS preflight).
- **Secrets:** none in git history or in the built bundle.
- **No TODO/FIXME/HACK** markers anywhere.

## 8. Found but not fixed, and why

| ID | Why not fixed | What I recommend |
|---|---|---|
| S-01 | The right fix depends on Render's proxy chain (which hop appends the real IP, and whether Cloudflare sits in front). A wrong guess would put **all** users in one rate-limit bucket. | Confirm the header layout on Render, then key the limiter on the hop your proxy appends (rightmost trusted entry) or on Render's client-IP header, and set `--forwarded-allow-ips` to the proxy range. The `xfail(strict=True)` test will start failing once it's fixed, as a reminder to remove the marker. |
| S-02 | Changes how users are identified, which is a product decision. | Send the id in a header instead of the URL; better, issue a signed token from the server. |
| S-03, S-05 | Deployment configuration. | Limit CORS to the exact Vercel domain(s); set `docs_url=None` in production. |
| A-01 | Changes the visual palette. | On light backgrounds use `text-slate-500`/`600` instead of `400`; on the dark sidebar use `text-slate-400` instead of `500`. |
| P-01, B-20 | A small refactor of how `App` passes callbacks (stable callbacks via refs, then `React.memo` on `MessageView`). A naive memo would keep stale `send` closures, a worse bug. | Do it as a separate change; only auto-scroll if the user is already near the bottom. |
| B-22 | Needs a design choice: persist partial answers when the client disconnects, or run generation detached from the request. | Catch the cancellation, save what was generated, then re-raise. |
| B-15 | A database check in `/api/health` could make Render restart the service during a short Neon hiccup. | Add a separate `/api/ready` with `SELECT 1` for monitoring. |
| B-10 (partly) | "Upcoming Fortuner" still resolves to the launched Fortuner: the "Fortuner Facelift" model shares no alias with it. | Link related upcoming models explicitly in the catalog. |
| S-04, S-06, S-08, A-02, A-03, C-01…C-05 | Low risk; outside "small, focused bug fixes". | Pick them up in a cleanup pass. |

## 9. Security summary and actions

1. **Fix S-01 before any public launch.** It is the main protection for the shared Groq quota, and today it is bypassed by one
   header.
2. **Rotate the credentials in `.env`** (S-07). They were never committed, but its own comment says to revoke them.
3. **Tighten CORS** to the real UI domain (S-03), and **hide `/docs`** in production (S-05).
4. **Decide the identity model** (S-02). At least move `user_id` out of URLs.
5. Add rate limits to feedback and delete (S-04); run the container as a non-root user (S-08).
6. Already solid: parameterized SQL everywhere, no LLM-written SQL, XSS-safe rendering, ownership checks on every
   conversation read, write and delete (403/404 without leaking that the id exists), no secrets in git or the bundle.

## 10. How to run all the tests

Prerequisites: the local Docker Postgres from the README (`carwale-pg` on port 5433) with the catalog loaded, Python 3.12
`.venv`, Node 20+. Commands below are for Windows; on macOS/Linux use `.venv/bin/python`.

```bash
# ---- one-time setup ------------------------------------------------------------------------------
docker start carwale-pg
# isolated test DB = clone of the local catalog without any chat data (stop the dev backend first:
# a template database must have no open connections)
docker exec carwale-pg psql -U postgres -c "CREATE DATABASE carwale_test TEMPLATE carwale;"
docker exec carwale-pg psql -U postgres -d carwale_test -c "TRUNCATE conversations, messages, conversation_state, answer_cache, feedback RESTART IDENTITY CASCADE;"
uv pip install --python .venv -r requirements-dev.txt
cd frontend && npm ci && npx playwright install chromium && cd ..

# ---- Python: unit + integration (integration tests skip themselves if the test DB is unreachable) -
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -m pytest tests -q --ignore=tests/integration      # unit only, no database needed
.venv/Scripts/python -m pytest tests/integration -q                      # integration only
.venv/Scripts/python -m pytest tests -q --cov=backend/app --cov=ingestion --cov-report=term

# ---- Frontend unit tests -------------------------------------------------------------------------
cd frontend
npm test
npm run test:coverage

# ---- End-to-end (starts the fake Groq server, the backend on carwale_test and a production build) --
npm run e2e
npm run e2e:report          # HTML report with screenshots, traces and videos of any failure

# ---- Static checks -------------------------------------------------------------------------------
npm run typecheck && npm run build
cd .. && uvx ruff check backend ingestion eval tests
```

Overrides: `TEST_DATABASE_URL` (pytest), `E2E_DATABASE_URL` and `E2E_PYTHON` (Playwright). All three refuse a database
whose name doesn't end in `_test`. No test uses the real Groq key.

## 11. Is it ready for production?

**For a demo, portfolio piece or small invite-only beta: yes, after these fixes.** The architecture is good: the LLM only
plans, deterministic code runs the plan with parameterized SQL, the plan schema tolerates messy output, retrieval is
hybrid with fallbacks, and the trace makes it explainable. The happy path is fast and works end to end, on desktop and
mobile, in every flow I tested. The bugs fixed here were real (crashed turns, wrong cars, wrong prices, misleading
errors, lost answers), but none needed a redesign.

**For a public launch: not yet.** What's still missing:

1. **Abuse protection that works:** S-01 (spoofable rate limit) and S-03 (open CORS). Without them, anyone can spend the
   shared free-tier key.
2. **An identity model** (S-02): today anyone who learns a `user_id` owns those chats.
3. **A real-model evaluation:** every test here uses a fake LLM (on purpose; the rules forbid real secrets), so answer
   *quality* is untested in this pass. Run `python eval/run_eval.py --answers` with the real key, especially because the
   prompts have uncommitted changes.
4. **Operability:** a readiness check that touches the DB (B-15), structured logs and metrics, error alerting, and CI
   running these 745 tests on every push.
5. **Scale limits:** one worker on Render's free tier with roughly 1-minute cold starts, an in-memory limiter that resets
   on restart and doesn't work across instances, and full-table scans on the general-question route (P-02).
6. **Polish:** colour contrast (A-01), streaming re-render cost (P-01), and the scroll and refresh behaviour (B-20, B-22).

---

## Appendix A: mistakes in my own tests (found and fixed during the work)

For transparency. None of these were app bugs, and none were "fixed" by weakening an assertion:

1. The E2E locator `name: "Helpful"` also matched "Not helpful" (Playwright matches substrings by default). Now `exact: true`.
2. Five E2E assertions matched the question text both in the chat bubble and in the sidebar title. They are now scoped to
   `main`.
3. The E2E rate-limit test used `retry-after: 120`. The backend correctly cooled every planner model for 120 s, which
   broke the next 4 tests. It now uses 2 s and also checks the visible "retrying" status.
4. The fake server crashed on an empty question, which surfaced as a 20 s delay. It's fixed in the fake; the app-side cause
   was B-02.
5. Two integration tests assumed the wrong data: the cheapest car costs ₹3.499 L, and the Creta–Seltos verdict is stored
   under the Seltos. Both now follow the real data, and the second uses the pipeline's own two-way lookup.
6. Three assertions that ended in `or True` (always true) were replaced by real ones before the first run. One electric
   assertion that could never fail was strengthened to check the database.
7. `userEvent.setup()` replaces `navigator.clipboard`, so the clipboard mock is now installed after it.

## Appendix B: environment notes

- On Windows, psycopg's async mode needs a `SelectorEventLoop`. The tests set it through the anyio backend, as
  `run_local.py` does for the app.
- Docker Desktop had to be started, and the stopped `carwale-pg` container started. Another project's stopped container
  (`quote-desk-db-1`) uses the same port 5433, so don't start both.
- Exact commands, versions and timings come from the runs on 17 Nov 2026: pytest 9.2, Vitest 5.1, Playwright 1.65 with
  Chromium, axe-core 4.14, Python 3.12.13, Node 24.16, PostgreSQL 17.12 with pgvector 0.8.7.
