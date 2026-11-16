# CarWale AI Assistant — a RAG chatbot over CarWale's new-car catalog

Ask anything about new cars in India: prices, mileage, specs, features, safety, service/warranty, owner opinions,
"car A vs car B", and open-ended searches like **"smart hybrid cars under 30 lakhs"**. The bot remembers the
conversation, so follow-ups such as *"which one has the best mileage?"*, *"what about its safety?"* and
*"only 7 seaters"* just work.

- **Data:** 400 models (288 on sale, 112 upcoming), 2,283 variants and 13,889 retrieval chunks, crawled politely from carwale.com
- **Stack:** FastAPI · Groq (gpt-oss-120b / gpt-oss-20b / Llama 3.x) · Postgres + pgvector (Neon) · fastembed (bge-small, ONNX) · React + Vite + Tailwind
- **Live:** frontend `https://<your-app>.vercel.app` · API `https://<your-api>.onrender.com/docs`

---

## Architecture

```mermaid
flowchart LR
  subgraph Offline["Offline ingestion (run once, re-runnable)"]
    S[carwale.com sitemaps] --> F[polite fetcher<br/>gzip cache]
    F --> P[parse embedded JSON state<br/>model + top-trim pages]
    P --> N[normalize<br/>prices, fuel, hybrid type,<br/>transmission, feature tags]
    N --> C[chunk + context headers]
    C --> E[embed bge-small-en-v1.5]
  end
  subgraph DB["Neon Postgres"]
    M[(models / variants<br/>structured catalog)]
    K[(chunks<br/>pgvector + tsvector)]
    H[(conversations / messages<br/>conversation_state)]
  end
  N --> M
  E --> K
  U[React UI<br/>Vercel] -- "POST /api/chat (SSE)" --> A[FastAPI<br/>Render]
  A -- "1 plan (strict JSON)" --> G1[Groq gpt-oss-20b]
  A -- "2 SQL search / specs" --> M
  A -- "3 hybrid search (RRF)" --> K
  A -- "4 answer (stream)" --> G2[Groq gpt-oss-120b]
  A <--> H
```

### What happens for one message

| Step | What | Where |
|---|---|---|
| 1. Memory | Load the rolling summary, the last 3 exchanges (as digests), the active cars from the last answer, the focus car and the current filters | `memory/store.py` |
| 2. Plan (LLM #1) | `gpt-oss-20b` with a **strict JSON schema** turns the message into a `QueryPlan`: standalone rewrite, intent, car mentions, references ("it", "the second one"), filters (INR), soft preferences, sort and aspects | `rag/planner prompt`, `rag/schemas.py` |
| 3. Resolve | Map car mentions to catalog IDs: aliases, renames (XUV700 → XUV 7XO), rapidfuzz for typos ("cretta"), and ambiguity handling ("innova" covers Crysta and Hycross) | `rag/entities.py` |
| 4. Route + retrieve | **Structured search:** whitelisted, parameterized SQL over variants, with automatic relaxation when nothing matches. **Car details / compare:** deterministic fact sheets and comparison tables. **Hybrid search:** pgvector KNN plus Postgres full-text search, fused with Reciprocal Rank Fusion and pre-filtered by car and section | `rag/car_search.py`, `rag/specs.py`, `rag/hybrid_search.py` |
| 5. Context | A token-budgeted context (~2.2K tokens): CAR DATA first, then numbered SOURCES for citations | `rag/context.py` |
| 6. Answer (LLM #2) | `gpt-oss-120b`, streamed over SSE, grounded and cited. Prices are labelled ex-showroom, and missing data is stated as missing | `llm/prompts.py` |
| 7. Persist | Save both messages (plus the trace) and update active cars, focus, filters and preferences. Long chats are summarized in the background | `rag/orchestrator.py` |

**Why not plain vector RAG?** "Under 30 lakhs" is a numeric constraint and "hybrid" is a categorical one. Embeddings can't
apply either reliably. Here the LLM only *plans*; deterministic code executes the plan with SQL, and semantic search
supplies the qualitative evidence (reviews, expert opinion, FAQs). The LLM never writes SQL, so there is no injection
surface, and every plan is validated and testable.

### Worked example: "smart hybrid cars under 30 lakhs"
1. **Plan:** `intent=recommend`, `fuel_types=[Hybrid]`, `hybrid_types=[strong, mild]`, `price_max=3000000`,
   `prefer_tags=[connected_car, adas, ota_updates, wireless_carplay]`. Here "smart" becomes soft preferences, not a hard filter.
2. **SQL:** launched models that have at least one hybrid variant at or below ₹30 L ex-showroom, ranked by matching smart features plus popularity
   and rating. Results include Victoris, Duster, Innova Hycross, Invicto, Brezza (mild hybrid), City e:HEV, Grand Vitara and Hyryder.
3. **Hybrid search:** pulls overview, expert-opinion and pros/cons chunks for those models (at most 2 per model).
4. **Answer:** a table with hybrid type, the price of the matching variants, claimed mileage, smart features and NCAP rating. It explains that
   Maruti's "Smart Hybrid" is a *mild* hybrid, and ends with who should pick which car.
5. **Memory:** `active_cars` holds the 8 models, so "which one has the best mileage?", "what about its safety?" and
   "only 7 seaters" all resolve correctly.

The UI's **"How I answered this"** panel shows this exact trace live for every answer.

## Memory design (5 layers)
| Layer | Holds | Enables |
|---|---|---|
| Persistent history | `messages` in Postgres. The browser keeps a `user_id` and the conversation ID | Chats survive a refresh or a server restart; past-chats sidebar |
| Short-term buffer | The last 3 exchanges. Assistant turns are stored as compact digests | Reference resolution in the planner |
| Rolling summary | `llama-3.1-8b-instant` folds older turns into `conversation_state.summary` (background task) | Long chats stay within the token budget |
| Entity memory | `active_cars` (ordered list) and `focus_car` | "it", "the second one", "both", "compare them" |
| Slot memory | Filters and preferences (budget, fuel, seats, city) with `filter_mode: new or refine` | "only automatics", "make it diesel", "what about in Pune?" |

## Data pipeline
- **Discovery:** `sitemap/cw-model.xml` gives 400 English model URLs (the `/hi/` duplicates are skipped). Only paths that robots.txt allows are fetched.
- **Pages:** every **model page** plus **one top-trim page per model**. Each page embeds its full app state as JSON
  (`<script id="initial-state">`), so parsing reads clean JSON rather than fragile CSS selectors. A single trim page holds the
  86-item spec sheet for *every* version of the model, plus the full feature tree (ADAS, connected-car and so on) of that
  trim. That cut the crawl from ~3,500 pages to 720. The crawl took about 7 minutes at ~1.7 requests per second, with 2 parallel requests, backoff and a disk cache.
- **Captured per model:**
  - prices (ex-showroom per version, plus on-road prices in 8 cities)
  - full specs and features
  - ARAI, CarWale-tested and owner-reported mileage
  - EV range, battery and charging
  - NCAP rating, airbags and ADAS
  - warranty
  - pros and cons, the expert verdict, and expert opinion in 4 areas
  - the full expert review
  - FAQs
  - owner reviews with category ratings, including "After Sales Cost and Experience"
  - rival models
  - latest news
  - upcoming launches
- **Quality report (launched variants):**

  | Field | Filled |
  |---|---|
  | Price, fuel, transmission, seats | 100% |
  | Power | 99% |
  | Airbags | 98% |
  | Boot space | 73% |
  | Mileage, petrol/diesel/CNG (ARAI 47%, plus expected/owner figures = 74%) | 74% |
  | EV range | 94% |
  | NCAP | 55% (many cars were never crash-tested) |

- **Known gap:** CarWale doesn't publish scheduled service costs. Service answers therefore use warranty, owner after-sales
  feedback and FAQs, and say plainly what isn't available.

## Groq free-tier strategy
- **Exactly 2 LLM calls per turn.** The planner call is ~1.5K tokens and the answer call ~4K. Follow-up chips are template-based, so they cost no tokens.
- **Fallback chain per role.** A model that returns 429 goes on cooldown for its `retry-after` window. Limits are per model, so the chain multiplies capacity:
  - planner: `gpt-oss-20b` → `qwen3.8-27b` → `llama-3.1-8b-instant`
  - answer: `gpt-oss-120b` → `llama-3.3-70b-versatile` → `gpt-oss-20b`
- **Prompt-cache-friendly layout.** Static prompts come first. Groq doesn't count cached tokens against rate limits.
- **Answer cache** for first-turn questions, which makes the starter chips instant and free. There's also a per-IP rate limit to protect the shared key.
- **Demo the fallback chain:** set `FORCE_FAIL_MODELS=openai/gpt-oss-120b`.

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat` | `{user_id, conversation_id?, message}`. Returns an SSE stream with these events: `conversation`, `status`, `trace`, `cars`, `sources`, `token`, `followups`, `done`, `error` |
| GET | `/api/conversations?user_id=` | List of past chats |
| GET | `/api/conversations/{id}?user_id=` | Full history of one chat, including traces |
| DELETE | `/api/conversations/{id}?user_id=` | Delete a chat |
| POST | `/api/feedback` | Thumbs up/down |
| GET | `/api/health` | Health check, catalog stats and the data snapshot date |

## Run locally
**Prerequisites:** Python 3.12, Node 20+, Docker (for local Postgres), a Groq API key.

1. Start the database and create a Python environment:
   ```bash
   docker run -d --name carwale-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=carwale -p 5433:5432 pgvector/pgvector:pg17
   uv venv --python 3.12 .venv
   uv pip install -r backend/requirements.txt httpx selectolax beautifulsoup4 lxml pytest
   ```
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
3. Run ingestion. It takes about 7 minutes to crawl, plus ~25 minutes to embed on CPU:
   ```bash
   python -m ingestion.discover
   python -m ingestion.fetch models && python -m ingestion.fetch trims
   python -m ingestion.parse && python -m ingestion.build_chunks && python -m ingestion.load_db
   ```
4. Start the backend (`python run_local.py` works around the Windows event-loop issue). Use the CLI harness to try a full conversation:
   ```bash
   cd backend && python run_local.py
   python cli.py "smart hybrid cars under 30 lakhs" "which one has the best mileage?"
   ```
5. Start the frontend:
   ```bash
   cd frontend && npm install && npm run dev      # http://127.0.0.1:5173
   ```
6. Run the tests and the eval:
   ```bash
   python -m pytest tests -q
   python eval/run_eval.py            # dry run: planner + retrieval accuracy
   python eval/run_eval.py --answers  # full answers with fact checks
   ```

## Deploy (all free tiers)
1. **Neon:** create a project in AWS Singapore. Then load the catalog, reusing the local vectors instead of recomputing them:
   ```bash
   DATABASE_URL=<direct url> EMBED_SOURCE_URL=postgresql://postgres:postgres@localhost:5433/carwale python -m ingestion.load_db
   ```
2. **Render:** New → Blueprint → this repo (`render.yaml`). This creates a Docker web service, plan free, region Singapore. Set:
   - `GROQ_API_KEY`
   - `DATABASE_URL` (Neon's *pooled* URL)
   - `CORS_ORIGINS` (your Vercel URL)
3. **Vercel:** import the repo with root directory `frontend` and set `VITE_API_BASE_URL=https://<your-api>.onrender.com`.
4. Render's free instance sleeps after 15 minutes idle. The UI pings `/api/health` on load and shows a "waking up" banner, so open the site a minute before a demo.

## Project structure
```
ingestion/   discover · fetch · parse · normalize · build_chunks · load_db · glossary.md
backend/app/ main · config · bootstrap · embeddings · db/schema.sql
             llm/{client,prompts} · rag/{schemas,entities,car_search,specs,hybrid_search,context,followups,orchestrator}
             memory/store · api/{chat,routes}
frontend/src App · api · components/{Sidebar,MessageView,Markdown,CarCards,TracePanel,Composer}
eval/        golden.json · run_eval.py          tests/  unit tests (parsers, resolver, SQL safety, schema)
```

## Limitations and next steps
- **Snapshot data.** Next step: a scheduled re-crawl that uses sitemap `lastmod` and `content_hash` upserts, so only changed pages are re-fetched and re-embedded.
- **On-road prices cover 8 major cities only.** There's no service-cost data, and used cars are out of scope.
- **Scaling path:**
  - HNSW with `hnsw.iterative_scan`, or a dedicated vector DB, at millions of chunks (exact KNN is used now; it's correct under filters and takes milliseconds at 14K rows)
  - Redis for sessions and caching
  - horizontally scaled stateless API pods
  - an LLM gateway
  - Langfuse/OpenTelemetry tracing
  - feedback-driven evals
