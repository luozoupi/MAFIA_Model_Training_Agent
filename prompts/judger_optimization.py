#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt builder for **single most impactful optimisation** suggestion.

Use this when you do **not** provide an error log. Instead, supply:
  - NCU metrics block (text/markdown)
  - GPU name (looked up in prompts/hardware/gpu_specs.py)
  - PyTorch reference architecture file (contains `class Model`)
  - (Optional) current CUDA candidate code to inspect

The Judge LLM must return **exactly one** optimisation target with a minimal plan.
"""

from __future__ import annotations
from pathlib import Path
from string import Template
from textwrap import dedent
import importlib.util
import sys
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

__all__ = ["build_single_opt_prompts"]

# -----------------------------------------------------------------------------
# GPU spec loader (shared pattern)
# -----------------------------------------------------------------------------

def _load_gpu_spec() -> dict:
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# System prompt: exactly one optimisation target
# -----------------------------------------------------------------------------

from textwrap import dedent
from string import Template

system_prompt_tmpl = Template(
    dedent(
        """
You are a senior CUDA performance engineer. Read the target GPU spec, the PyTorch
reference code, the current CUDA candidate, and the Nsight Compute
metrics. Then identify **exactly one** highest-impact speed bottleneck, propose **exactly one** optimisation method and propose a
modification plan. Be surgical and metrics-driven.

Rules:
- Return **one and only one** optimisation method — the largest expected speedup.
- Prefer changes that directly address measured bottlenecks (occupancy limits,
  memory coalescing, smem bank conflicts, register pressure, long/short scoreboard
  stalls, tensor-core underutilisation, etc.).
- Keep fields brief; avoid lists of alternatives, disclaimers, or generic advice.

Output format (JSON):
```json
{
  "bottleneck": "<max 30 words>",
  "optimisation method": "<max 35 words>",
  "modification plan": "<max 35 words>"
}
"""
)
)

# -----------------------------------------------------------------------------
# Instruction prompt injects code, metrics, GPU spec
# -----------------------------------------------------------------------------

instruction_tmpl = Template(
    dedent(
        """
# Target GPU
GPU Name: $gpu_name
Architecture: $gpu_arch
Details:
$gpu_items


# Pytorch Reference
$python_code


# CUDA candidate
```python
$CUDA_CODE
```

# Nsight Compute metrics (verbatim)
$NCU_METRICS

Read everything and follow the Rules exactly. Return the JSON in the specified format.
"""
    )
)

# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

def build_judger_optimization_prompts(
    *,
    arch_path: Path,
    gpu_name: str,
    ncu_metrics_block: str,
    cuda_code: str = "",
) -> Tuple[str, str]:
    """Return (system_prompt_str, instruction_str) for single-issue optimisation.

    Args:
        arch_path:   Path to .py that contains the PyTorch reference `class Model`
        gpu_name:    Key in GPU_SPEC_INFO (e.g., "Quadro RTX 6000")
        ncu_metrics_block: Text/Markdown block of Nsight Compute metrics
        cuda_code:   Optional current CUDA candidate source (string)
    """
    gpu_info = _load_gpu_spec()
    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )

    arch_src = Path(arch_path).read_text().strip()
    system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        python_code=arch_src,
        CUDA_CODE=cuda_code.strip(),
        NCU_METRICS=ncu_metrics_block.strip(),
    )
    return system_prompt, instruction