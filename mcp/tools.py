from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def load_user_request(*, request_path: Path) -> dict[str, Any]:
    """Load a structured user request JSON file."""
    if not request_path.exists():
        raise FileNotFoundError(f"Request file not found: {request_path}")
    return json.loads(request_path.read_text(encoding="utf-8"))


def scan_hf_models(*, catalog_path: Path | None, task: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return candidate models for the given task using a lightweight catalog file."""
    if catalog_path is None or not catalog_path.exists():
        return []

    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    candidates = []
    task_lower = task.lower()
    for entry in data:
        if task_lower and task_lower not in str(entry.get("task", "")).lower():
            continue
        candidates.append(entry)
        if len(candidates) >= limit:
            break
    return candidates


def run_finetune_script(*, script_path: Path, work_dir: Path) -> dict[str, Any]:
    """Execute a finetuning script and return captured output."""
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    work_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["python3", str(script_path)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
