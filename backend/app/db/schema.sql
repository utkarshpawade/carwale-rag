-- CarWale RAG schema (idempotent). Works on Neon Postgres and local pgvector/pgvector images.
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- structured catalog
CREATE TABLE IF NOT EXISTS models (
  id                  TEXT PRIMARY KEY,          -- 'toyota/innova-hycross'
  cw_model_id         INT,
  brand               TEXT NOT NULL,
  brand_slug          TEXT,
  name                TEXT NOT NULL,
  slug                TEXT,
  full_name           TEXT NOT NULL,
  url                 TEXT,
  status              TEXT,                      -- launched | upcoming | discontinued
  launched_on         DATE,
  body_type           TEXT,                      -- canonical: SUV, MUV, Hatchback, Sedan, ...
  body_type_raw       TEXT,                      -- CarWale label: 'Sub-Compact SUV', 'MUV/MPV'
  is_ev               BOOLEAN,
  price_min           BIGINT,                    -- INR ex-showroom
  price_max           BIGINT,
  price_is_estimate   BOOLEAN,
  fuel_types          TEXT[],
  hybrid_types        TEXT[],
  transmissions       TEXT[],
  has_automatic       BOOLEAN,
  seats               INT[],
  ncap_rating         INT,
  ncap_agency         TEXT,
  airbags_max         INT,
  mileage_best        REAL,
  range_best_km       REAL,
  power_max_bhp       REAL,
  boot_l              INT,
  ground_clearance_mm INT,
  user_rating         REAL,
  user_rating_count   INT,
  expert_rating_5     REAL,
  feature_tags        TEXT[],
  image_url           TEXT,
  data                JSONB                      -- remaining structured detail (city prices, warranty, pros/cons, ...)
);

CREATE TABLE IF NOT EXISTS variants (
  version_id             BIGINT PRIMARY KEY,
  model_id               TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  name                   TEXT,
  display_name           TEXT,
  trim                   TEXT,
  url                    TEXT,
  price                  BIGINT,
  fuel_type              TEXT,
  hybrid_type            TEXT,
  transmission           TEXT,
  is_automatic           BOOLEAN,
  engine_cc              INT,
  power_bhp              REAL,
  torque_nm              REAL,
  mileage_arai           REAL,
  mileage_unit           TEXT,
  mileage_owner          REAL,
  mileage_expected       REAL,
  mileage_effective      REAL,                   -- best available figure, see mileage_basis
  mileage_basis          TEXT,                   -- 'ARAI claimed' | 'expected' | 'owner-reported' | 'range_km'
  mileage_city_tested    REAL,
  mileage_highway_tested REAL,
  range_km               REAL,
  battery_kwh            REAL,
  dc_charge_minutes      REAL,
  seats                  INT,
  airbags                INT,
  ncap_rating            INT,
  ncap_agency            TEXT,
  boot_l                 INT,
  ground_clearance_mm    INT,
  length_mm              INT,
  width_mm               INT,
  height_mm              INT,
  wheelbase_mm           INT,
  fuel_tank_l            INT,
  kerb_weight_kg         INT,
  drivetrain             TEXT,
  feature_tags           TEXT[],
  specs                  JSONB
);
CREATE INDEX IF NOT EXISTS variants_model_idx ON variants (model_id);
CREATE INDEX IF NOT EXISTS variants_price_idx ON variants (price);
CREATE INDEX IF NOT EXISTS variants_fuel_idx  ON variants (fuel_type);
CREATE INDEX IF NOT EXISTS variants_tags_idx  ON variants USING GIN (feature_tags);

-- ---------------------------------------------------------------- unstructured chunks (hybrid search)
CREATE TABLE IF NOT EXISTS chunks (
  id           TEXT PRIMARY KEY,
  model_id     TEXT,
  variant_id   BIGINT,
  section      TEXT,
  title        TEXT,
  content      TEXT NOT NULL,
  url          TEXT,
  brand        TEXT,
  body_type    TEXT,
  fuel_types   TEXT[],
  price_min    BIGINT,
  price_max    BIGINT,
  status       TEXT,
  content_hash TEXT,
  embedding    vector(384),
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_model_idx     ON chunks (model_id);
CREATE INDEX IF NOT EXISTS chunks_section_idx   ON chunks (section);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING GIN (tsv);
-- Exact KNN on purpose: at ~15K chunks a scan is a few ms and stays correct under metadata filters
-- (HNSW + selective WHERE can drop results). Scale path: HNSW with hnsw.iterative_scan, or a vector DB.
DROP INDEX IF EXISTS chunks_embedding_idx;

-- ---------------------------------------------------------------- conversation memory
CREATE TABLE IF NOT EXISTS conversations (
  id         UUID PRIMARY KEY,
  user_id    TEXT NOT NULL,
  title      TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
  id              BIGSERIAL PRIMARY KEY,
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role            TEXT NOT NULL,                 -- user | assistant
  content         TEXT NOT NULL,
  meta            JSONB NOT NULL DEFAULT '{}',   -- plan, cars, sources, trace, digest
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conv_idx ON messages (conversation_id, id);

CREATE TABLE IF NOT EXISTS conversation_state (
  conversation_id UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
  summary         TEXT NOT NULL DEFAULT '',
  summarized_upto BIGINT NOT NULL DEFAULT 0,     -- last message id folded into the summary
  active_cars     JSONB NOT NULL DEFAULT '[]',   -- ordered model ids from the latest answer
  focus_car       TEXT,
  filters         JSONB NOT NULL DEFAULT '{}',
  prefs           JSONB NOT NULL DEFAULT '{}',
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS answer_cache (
  key        TEXT PRIMARY KEY,
  payload    JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
  id         BIGSERIAL PRIMARY KEY,
  message_id BIGINT,
  rating     INT,
  comment    TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
