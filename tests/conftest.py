import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# ---------------------------------------------------------------- test isolation (set before any app import)
# Tests must never read the developer's real .env (it holds live API keys) or reach the real Groq API.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/carwale_test")
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["GROQ_API_KEY"] = "test-key-not-real"
os.environ["GROQ_BASE_URL"] = "http://127.0.0.1:9"  # closed port: an accidental real call fails fast, locally
os.environ["RATE_LIMIT_PER_MIN"] = "100000"
os.environ["RATE_LIMIT_PER_DAY"] = "1000000"
os.environ.pop("FORCE_FAIL_MODELS", None)

from app import config as _config  # noqa: E402

_config.Settings.model_config["env_file"] = None  # ignore .env files entirely
_config.get_settings.cache_clear()

import asyncio  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def anyio_backend():
    # psycopg's async mode cannot run on Windows' default Proactor loop
    return ("asyncio", {"loop_factory": asyncio.SelectorEventLoop}) if sys.platform == "win32" else "asyncio"
