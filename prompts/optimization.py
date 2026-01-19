from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # Adjust import path as needed

_OPTIMIZATION_PROMPT_TEMPLATE = Template("""\
# Target GPU
GPU Name: $gpu_name
Architecture: $gpu_arch
Details:
$gpu_items

# ----------  Previously generated kernels ----------
$history_block

You are a CUDA-kernel optimization specialist.

Analyze the provided architecture and kernel history and **strictly apply the following STRATEGY** to produce an improved CUDA kernel.

[ARCHITECTURE FILE]
```python
$arch_src
```

[optimization instructions]
$optimization_suggestion

GOAL
────
- Improve latency and throughput on the target GPU.
- Maintain correctness within atol=1e-4 or rtol=1e-4.
- Preserve the public Python API (same inputs/outputs, shapes, dtypes).

OUTPUT RULES (STRICT) ────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple-quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.

```python
# <your corrected code>
```
# ==========================================================
""")

def _escape_template(s: str) -> str:
    return s.replace("$", "$$")

def _sanitize_text(s: str) -> str:
    return s.replace("```", "`")

def _format_problem(problem: Optional[Any]) -> str:
    if problem is None or problem == "":
        return "No prior critical problem provided."
    if isinstance(problem, Mapping):
        # Prefer extracting bottleneck / optimisation method / modification plan
        bottleneck = str(problem.get("bottleneck", "")).strip()
        opt_method = str(problem.get("optimisation method", "")).strip()
        mod_plan   = str(problem.get("modification plan", "")).strip()
        if bottleneck or opt_method or mod_plan:
            return (
                "{\n"
                f'  "bottleneck": "{bottleneck}",\n'
                f'  "optimisation method": "{opt_method}",\n'
                f'  "modification plan": "{mod_plan}"\n'
                "}"
            )
        # fallback to JSON dump
        return json.dumps(problem, ensure_ascii=False, indent=2)
    # For other types, convert to string directly
    return str(problem)

def build_optimization_prompt(
    arch_path: Path,
    gpu_name: Optional[str] = None,
    *,
    history_block: str = "",
    optimization_suggestion: Optional[Any] = None,
) -> str:
    """Build LLM prompt for CUDA-kernel optimisation (optimization phase)."""
    gpu_info = _load_gpu_spec()

    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture")

    arch_src = Path(arch_path).read_text().strip()
    hist = history_block or "(None)\n"
    optimization_suggestion_text = _format_problem(optimization_suggestion)
    return _OPTIMIZATION_PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        arch_src=arch_src,
        history_block=hist,
        optimization_suggestion=optimization_suggestion_text,
    )