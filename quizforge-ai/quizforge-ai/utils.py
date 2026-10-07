from __future__ import annotations

from pathlib import Path
from typing import Iterable


def read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def list_syllabus_files(syllabus_dir: Path) -> list[str]:
    if not syllabus_dir.exists():
        return []
    return sorted(
        path.name for path in syllabus_dir.glob("*.txt") if path.is_file()
    )


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def safe_download_data(data: str, filename: str) -> bytes:
    return data.encode("utf-8")
