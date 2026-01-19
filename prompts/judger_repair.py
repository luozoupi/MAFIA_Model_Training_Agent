# prompts/error.py
"""
Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
Adds GPU hardware context and architecture source for better fixes.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from string import Template

# Project roots (adjust if your tree differs)
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Reuse your existing GPU spec loader
from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402


# -----------------------------
# system_prompt as Template
# -----------------------------
system_prompt_tmpl = Template(
    """You are a senior CUDA + PyTorch correctness auditor. Your job is to read a PyTorch reference and a CUDA candidate and report exactly one most critical correctness issue in the CUDA code that would cause a behavioral mismatch vs. the PyTorch reference. Be terse and precise.

Rules:

Return one and only one issue — the single highest-impact problem.

Prefer semantic/correctness issues over micro-optimizations or style.

If multiple issues exist, pick the one that most changes outputs or gradients.

If nothing clearly wrong is found, say it explicitly.

Keep each field brief; avoid extra commentary, lists, or alternatives.

Output format (JSON):
```json
{
  "critical_issue": "<max 20 words>",
  "why_it_matters": "<max 35 words>",
  "minimal_fix_hint": "<max 20 words>"
}
```
"""
)

# -----------------------------
# instruction as Template
# -----------------------------
instruction_tmpl = Template(
    """You are given:

ERROR_LOG:
$ERROR_LOG

PyTorch reference (ground truth):

$PYTORCH_CODE

CUDA candidate (to audit):

$CUDA_CODE


Follow the Rules and produce the JSON exactly in the specified format."""
)

# -----------------------------
# Build both at once (returns tuple)
# -----------------------------
def build_correctness_prompts(*, error_log: str, arch_path: Path, cuda_code: str):
    """
    Return (system_prompt_str, instruction_str).
    """
    pytorch_code = Path(arch_path).read_text().strip()
    system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        ERROR_LOG=error_log.strip(),
        PYTORCH_CODE=pytorch_code.strip(),
        CUDA_CODE=cuda_code.strip(),
    )
    return system_prompt, instruction
