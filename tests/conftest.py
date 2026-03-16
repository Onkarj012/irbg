from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """
    Ensure `src/` is importable when running tests without installing it.

    This repo uses a src-layout (package lives in `src/irbg`). Some environments
    run `pytest` from the repo root without `pip install -e .`, so we add `src/`
    to `sys.path` to make `import irbg` work during collection.
    """

    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    sys.path.insert(0, str(src))
