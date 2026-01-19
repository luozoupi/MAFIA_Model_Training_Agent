# prompts/error.py
"""
Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
Adds GPU hardware context and architecture source for better fixes.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template

# Project roots (adjust if your tree differs)
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Reuse your existing GPU spec loader
from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402


COMPILE_ERROR = Template(
    """You are a senior CUDA-extension developer.
Your job is to **FIX** the compilation or runtime errors in the Python script
shown below.

OUTPUT RULES (STRICT) ────────────────────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple-quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.

────────────────────────────────────────────────────────────────
ERROR LOG
────────────────────────────────────────────────────────────────
$ERROR_LOG

────────────────────────────────────────────────────────────────
OLD CODE (read-only)
────────────────────────────────────────────────────────────────
$OLD_CODE

────────────────────────────────────────────────────────────────
Main Critical Problem
────────────────────────────────────────────────────────────────
$Problem

```python
# <your corrected code>
```
# ==========================================================
"""
)

def _escape_template(s: str) -> str:
    return s.replace("$", "$$")

def _sanitize_text(s: str) -> str:
    return s.replace("```", "`")

def _format_problem(problem: Optional[Any]) -> str:
    if problem is None or problem == "":
        return "No prior critical problem provided."
    if isinstance(problem, Mapping):
        # Prefer to concatenate the three key fields into a concise description; otherwise fall back to JSON
        ci  = str(problem.get("critical_issue", "")).strip()
        wim = str(problem.get("why_it_matters", "")).strip()
        mfh = str(problem.get("minimal_fix_hint", "")).strip()
        if ci or wim or mfh:
            return f"critical_issue: {ci}\nwhy_it_matters: {wim}\nminimal_fix_hint: {mfh}"
        return json.dumps(problem, ensure_ascii=False, indent=2)
    # For other types, simply convert to string
    return str(problem)

def build_error_prompt(
    *,
    old_code: str,
    error_log: str,
    problem: Optional[Any] = None,
    gpu_name: Optional[str] = None,
) -> str:
    """
    Build the error-repair prompt with GPU context + architecture source.

    Parameters
    ----------
    old_code : str
        The broken Python script content to show under OLD CODE.
    error_log : str
        The compiler/runtime error text to show under ERROR LOG.
    arch_path : Path
        Path to the reference architecture Python file to display.
    gpu_name : Optional[str]
        Human-readable GPU name key to lookup in gpu_specs.
        If None, attempts torch.cuda.get_device_name(0).

    Returns
    -------
    str
        The final prompt string to send to the LLM.
    """
    # Load the GPU spec dictionary
    gpu_info = _load_gpu_spec()

    # Resolve GPU name
    if gpu_name is None:
        try:
            import torch  # local import to avoid hard dependency if CPU-only
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO (file: {HW_FILE})")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")

    # Bullet list of key specs except the arch line (already printed separately)
    gpu_items = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )
    problem_text = _format_problem(problem)
    # Substitute all fields
    return COMPILE_ERROR.substitute(
        ERROR_LOG=error_log.strip(),
        OLD_CODE=old_code.strip(),
        Problem=_escape_template(_sanitize_text(problem_text.strip())),
    )