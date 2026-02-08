from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def scan_hf_models(*, catalog_path: Path | None, task: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return candidate models for the given task using a lightweight catalog file."""
    if catalog_path is None or not catalog_path.exists():
        return []

    data = json.loads(catalog_path.read_text(encoding="utf-8"))

    # Try keyword match first; fall back to returning all entries
    candidates = []
    task_lower = task.lower()
    for entry in data:
        entry_text = f"{entry.get('task', '')} {entry.get('notes', '')} {entry.get('model_id', '')}".lower()
        if any(word in entry_text for word in task_lower.split()):
            candidates.append(entry)
            if len(candidates) >= limit:
                break
    # If nothing matched, return all entries up to limit
    if not candidates:
        candidates = data[:limit]
    return candidates


def run_finetune_script(
    *, script_path: Path, work_dir: Path, conda_env: str | None = None,
) -> dict[str, Any]:
    """Execute a finetuning script and capture stdout/stderr."""
    script_path = Path(script_path).resolve()
    work_dir = Path(work_dir).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Pre-execution syntax check (no .pyc written)
    try:
        source = script_path.read_text(encoding="utf-8")
        compile(source, str(script_path), "exec")
    except SyntaxError as exc:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"SyntaxError (pre-validation): {exc}",
        }

    # Build command with optional conda env activation
    if conda_env:
        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python3", str(script_path),
        ]
    else:
        cmd = ["python3", str(script_path)]

    result = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout for training scripts
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
