from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Ensure the src/ directory is on sys.path so `irbg` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from irbg.db.schema import create_tables


def test_create_tables_idempotent() -> None:
    conn = sqlite3.connect(":memory:")
    create_tables(conn)
    # Running twice should not fail.
    create_tables(conn)

    # Validate a couple of expected tables exist.
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='models';"
    )
    assert cur.fetchone() is not None

    cur = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='benchmark_runs';"
    )
    assert cur.fetchone() is not None

    conn.close()
