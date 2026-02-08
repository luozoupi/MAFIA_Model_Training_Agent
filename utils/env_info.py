"""Gather runtime environment info for LLM context injection."""
from __future__ import annotations

import sys
from typing import Any


def gather_env_info(
    *,
    conda_env: str | None = None,
    gpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Collect Python version, key package versions, and GPU info."""
    info: dict[str, Any] = {}

    info["python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # Key package versions
    packages = [
        "transformers", "peft", "trl", "torch", "datasets",
        "accelerate", "bitsandbytes", "evaluate",
    ]
    pkg_versions: dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkg_versions[pkg] = "not installed"
    info["packages"] = pkg_versions

    info["conda_env"] = conda_env

    # GPU info
    try:
        import torch
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                gpu_info.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(total_mem / (1024**3), 1),
                    "free_memory_gb": round(free_mem / (1024**3), 1),
                })
        info["gpus"] = gpu_info
    except Exception:
        info["gpus"] = []

    if gpu_ids is not None:
        info["assigned_gpus"] = gpu_ids
    else:
        info["assigned_gpus"] = [g["index"] for g in info.get("gpus", [])]

    return info


def format_env_info(info: dict[str, Any]) -> str:
    """Format env info dict into a human-readable string for prompt injection."""
    lines = []
    lines.append(f"Python: {info.get('python_version', 'unknown')}")
    if info.get("conda_env"):
        lines.append(f"Conda environment: {info['conda_env']}")
    lines.append("Installed packages:")
    for pkg, ver in info.get("packages", {}).items():
        lines.append(f"  - {pkg}=={ver}")
    lines.append("GPUs:")
    for g in info.get("gpus", []):
        marker = " [ASSIGNED]" if g["index"] in info.get("assigned_gpus", []) else ""
        lines.append(
            f"  - GPU {g['index']}: {g['name']}, "
            f"{g['total_memory_gb']}GB total, "
            f"{g['free_memory_gb']}GB free{marker}"
        )
    if info.get("assigned_gpus"):
        lines.append(f"Use ONLY these GPUs: {info['assigned_gpus']}")
    return "\n".join(lines)
