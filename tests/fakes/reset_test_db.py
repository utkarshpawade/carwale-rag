"""Empty the per-user tables of the *test* database before an E2E run (catalog tables are left alone).

Usage: python -m tests.fakes.reset_test_db [database_url]
Refuses to touch any database whose name does not end in "_test".
"""
from __future__ import annotations

import os
import sys

import psycopg


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("E2E_DATABASE_URL",
                                                               "postgresql://postgres:postgres@localhost:5433/carwale_test")
    name = url.rsplit("/", 1)[-1].split("?")[0]
    if not name.endswith("_test"):
        raise SystemExit(f"refusing to reset '{name}': not a *_test database")
    with psycopg.connect(url, autocommit=True, connect_timeout=10) as c:
        c.execute("TRUNCATE conversations, messages, conversation_state, answer_cache, feedback RESTART IDENTITY CASCADE")
    print(f"reset {name}")


if __name__ == "__main__":
    main()
