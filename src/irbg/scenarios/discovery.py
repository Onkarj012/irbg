from __future__ import annotations

from pathlib import Path


class ScenarioDiscoveryError(Exception):
    """Raised when scenario files cannot be discovered."""


def load_template_files(folder_path: Path) -> list[Path]:
    if not folder_path.exists():
        raise ScenarioDiscoveryError(
            f"Scenario folder not found: {folder_path}"
        )

    if not folder_path.is_dir():
        raise ScenarioDiscoveryError(
            f"Scenario folder is not a directory: {folder_path}"
        )

    files = sorted(
        path for path in folder_path.glob("*.json") if path.is_file()
    )

    if not files:
        raise ScenarioDiscoveryError(
            f"No json scenario files found in folder: {folder_path}"
        )

    return files
