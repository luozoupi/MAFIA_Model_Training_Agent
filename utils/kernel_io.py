# utils/kernel_io.py
"""Utility helpers for Mind‑Evolution CUDA‑kernel workflow.

This tiny module centralizes two common I/O helpers that were previously
inlined in the end‑to‑end test script:

1. ``extract_code_block`` – extract the first ```python ... ``` (or generic) code
   block from LLM output. Raises if none found.
2. ``save_kernel_code`` – writes extracted code to *kernels/* with a unique
   timestamped filename and returns the *Path* object.

Keeping them here avoids duplication across evolution loops / diagnostics.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Final
import json
from typing import Any, Dict, List
__all__: Final = [
    "extract_code_block",
    "save_kernel_code",
]

# ---------------------------------------------------------------------------
# 1. Code‑block extraction
# ---------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


# Match code‑fence opening; language tag is optional
_CODE_FENCE_OPEN_RE = re.compile(r"```(?:[A-Za-z0-9_+\-]+)?\s*\n?")

def extract_code_block(text: str) -> str:
    """Return the **first** triple‑back‑ticked block in *text*.

    - After finding an opening fence, search for the closing fence. If none is
      found, consume until end of string.
    - If the text contains no ``` fences at all, raise and dump the raw output
      to a timestamped file for debugging.
    """
    if text is None:
        text = ""

    m_open = _CODE_FENCE_OPEN_RE.search(text)
    if not m_open:
        # No ``` found → raise and persist raw output to disk
        dump_path = f"llm_output_error_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(dump_path, "w") as f:
            f.write(text)
        raise RuntimeError(f"No ``` code block found in LLM output – raw output saved to {dump_path}")

    start = m_open.end()
    m_close = re.search(r"```", text[start:])
    if m_close:
        end = start + m_close.start()
        block = text[start:end]
    else:
        # No closing fence: take everything to the end
        block = text[start:]

    return block.strip() + "\n"



# ---------------------------------------------------------------------------
# 2. Persist kernel to file
# ---------------------------------------------------------------------------

def save_kernel_code(code: str, out_dir: Path | str = "kernels") -> Path:
    """Save *code* to *out_dir/kernel_YYYYmmdd_HHMMSS.py* and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"kernel_{stamp}.py"
    path.write_text(code, encoding="utf-8")

    return path


# utils/kernel_io.py



def extract_json(raw: str) -> Any:
    """
    Extract the first JSON object/array from a string and parse it into a Python object.
    Supports fenced code blocks like ```json ...``` or raw JSON embedded in text.

    Args:
        raw: Raw LLM output text.
    Returns:
        A Python object (``dict`` or ``list``).
    Raises:
        ValueError: If no valid JSON can be found/parsed.
    """
    if not isinstance(raw, str):
        raw = str(raw)

    # Try the ```json ...``` fenced format first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try matching the first { ... } or [ ... ]
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback: attempt to parse the whole string
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        raise ValueError(f"Failed to extract valid JSON from reply:\n{raw}")

def save_prompt_text(text: str, out_dir: Path, *, tag: str = "repair") -> Path:
    """
    Save *text* to ``out_dir/{tag}_YYYYMMDD-HHMMSS.txt`` and return the Path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{tag}_{ts}.txt"
    path.write_text(text, encoding="utf-8")
    return path

def extract_cuda_kernel_names(py_path: Path) -> List[str]:
    try:
        src = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    p1 = re.compile(r"""__global__\s+void\s+([A-Za-z_]\w*)\s*\(""", re.MULTILINE)
    p2 = re.compile(
        r"""__global__\s+__launch_bounds__\s*\([^)]*\)\s*void\s+([A-Za-z_]\w*)\s*\(""",
        re.MULTILINE,
    )

    names = p1.findall(src) + p2.findall(src)
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered
