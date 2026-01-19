from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def scan_hf_models(*, catalog_path: Path | None, task: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return candidate models for the given task using a lightweight catalog file."""
    if catalog_path is None or not catalog_path.exists():
        return []


    candidates = []
    task_lower = task.lower()
    for entry in data:
        if task_lower and task_lower not in str(entry.get("task", "")).lower():
            continue
        candidates.append(entry)
        if len(candidates) >= limit:
            break
    return candidates



    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    work_dir.mkdir(parents=True, exist_ok=True)

