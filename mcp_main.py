from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.query_server import query_server
from mcp.tooling import MCPToolRegistry
from mcp.tools import run_finetune_script, scan_hf_models
from prompts.mcp_coder import build_coder_fix_prompts, build_coder_prompts
from prompts.mcp_feedback import build_feedback_prompts
from prompts.mcp_planner import build_planner_prompts
from utils.env_info import format_env_info, gather_env_info
from utils.kernel_io import extract_code_block, extract_json


# ---- helpers ----

def _read_task_input(task_arg: str) -> tuple[str, dict[str, Any] | None]:
    """Read task input. Returns (formatted_text, raw_dict_or_None).

    Accepts plain text, a text file path, or a JSON file/string.
    """
    path = Path(task_arg)

    # Try file first
    if path.exists() and path.is_file():
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            raw_dict = json.loads(content)
            return _format_request_dict(raw_dict), raw_dict
        return content, None

    # Try inline JSON
    try:
        raw_dict = json.loads(task_arg)
        return _format_request_dict(raw_dict), raw_dict
    except (json.JSONDecodeError, TypeError):
        pass

    return task_arg, None


def _format_request_dict(d: dict[str, Any]) -> str:
    """Convert a structured request dict into a formatted task description."""
    sections = []

    if "project_goal" in d:
        sections.append(f"Goal: {d['project_goal']}")
    if "model_preference" in d:
        sections.append(f"Model preference: {d['model_preference']}")
    if "gpu_resources" in d:
        sections.append(f"GPU resources: {d['gpu_resources']}")
    if "selected_model" in d:
        m = d["selected_model"]
        sections.append(f"Selected model: {m.get('id', 'unknown')} ({m.get('why', '')})")
    if "dataset_plan" in d:
        datasets = d["dataset_plan"].get("selected_open_datasets", [])
        if datasets:
            ds_lines = [f"  - {ds.get('id', '?')}: {ds.get('why', '')}" for ds in datasets]
            sections.append("Datasets:\n" + "\n".join(ds_lines))
    if "constraints_and_compliance" in d:
        sections.append(f"Constraints: {d['constraints_and_compliance']}")
    if "notes" in d and isinstance(d["notes"], list):
        sections.append("Notes:\n" + "\n".join(f"  - {n}" for n in d["notes"]))
    if "selection_metadata" in d:
        mm = d["selection_metadata"].get("selected_model_metadata", {})
        if mm:
            sections.append(
                f"Model details: {mm.get('id', '?')}, {mm.get('parameter_scale_hint', '?')} params, "
                f"context={mm.get('context_window', '?')}, dtype={mm.get('torch_dtype', '?')}"
            )

    return "\n\n".join(sections) if sections else json.dumps(d, indent=2)


def _safe_extract_code(reply: str) -> str:
    try:
        return extract_code_block(reply)
    except RuntimeError:
        return reply


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_registry() -> MCPToolRegistry:
    registry = MCPToolRegistry()
    registry.register(
        "scan_hf_models",
        scan_hf_models,
        "Scan a lightweight model catalog for candidate models.",
    )
    registry.register(
        "run_finetune_script",
        run_finetune_script,
        "Execute a finetuning script and capture stdout/stderr.",
    )
    return registry


def _call_llm(
    prompt: str,
    system_prompt: str,
    args: argparse.Namespace,
) -> str:
    response = query_server(
        prompt=prompt,
        system_prompt=system_prompt,
        server_type=args.server_type,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        server_address=args.server_address,
        server_port=args.server_port,
    )
    if isinstance(response, list):
        return response[0] if response else ""
    return str(response)


# ---- main ----

def main() -> None:
    parser = argparse.ArgumentParser("MCP multi-agent finetuning workflow")
    parser.add_argument("task", help="Task description text, text file path, or JSON request file")
    parser.add_argument("--server_type", default="openai")
    parser.add_argument("--server_address", default="localhost")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--model_name", default="gpt-4o-mini")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--model_catalog", default="mcp/model_catalog.json")
    parser.add_argument("--run_dir", default="mcp_runs")
    parser.add_argument("--model_limit", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max coder-execute-feedback retry iterations")
    parser.add_argument("--conda_env", default=None,
                        help="Conda environment name for script execution")
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None,
                        help="GPU indices to use (e.g., --gpu_ids 0 1)")
    args = parser.parse_args()

    # Read task input (plain text or structured JSON)
    task_text, _request_dict = _read_task_input(args.task)

    # Gather environment info for the coder
    env_data = gather_env_info(conda_env=args.conda_env, gpu_ids=args.gpu_ids)
    env_info_str = format_env_info(env_data)

    registry = _build_registry()
    catalog_path = Path(args.model_catalog) if args.model_catalog else None

    run_root = Path(args.run_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    io_dir = run_root / "llm_io"

    # Save task and env info for reference
    _write_text(io_dir / "task_input.txt", task_text)
    _write_text(io_dir / "env_info.txt", env_info_str)

    # ---- Model scan ----
    scan_result = registry.run(
        "scan_hf_models",
        catalog_path=catalog_path,
        task=task_text,
        limit=args.model_limit,
    )
    model_scan = scan_result.payload if scan_result.ok else []

    # ---- Planner phase ----
    planner_system, planner_prompt = build_planner_prompts(
        task=task_text,
        model_scan=model_scan,
    )
    _write_text(io_dir / "planner_prompt.txt", planner_prompt)
    planner_reply = _call_llm(planner_prompt, planner_system, args)
    _write_text(io_dir / "planner_reply.txt", planner_reply)
    plan_json = extract_json(planner_reply)
    _write_text(io_dir / "planner_plan.json", json.dumps(plan_json, indent=2))

    # ---- Coder → Execute → Feedback retry loop ----
    last_feedback: dict[str, Any] | None = None
    last_script: str | None = None
    last_run_output: dict[str, Any] | None = None
    script_text = ""

    for attempt in range(args.max_retries):
        iter_dir = io_dir / f"iter_{attempt}"

        # Coder step
        if attempt == 0:
            coder_system, coder_prompt = build_coder_prompts(
                task=task_text, plan=plan_json, env_info=env_info_str,
            )
        else:
            assert last_script is not None
            assert last_feedback is not None
            assert last_run_output is not None
            coder_system, coder_prompt = build_coder_fix_prompts(
                task=task_text,
                plan=plan_json,
                previous_script=last_script,
                feedback=last_feedback,
                run_output=last_run_output,
                env_info=env_info_str,
            )

        _write_text(iter_dir / "coder_prompt.txt", coder_prompt)
        coder_reply = _call_llm(coder_prompt, coder_system, args)
        _write_text(iter_dir / "coder_reply.txt", coder_reply)

        script_text = _safe_extract_code(coder_reply)
        script_path = run_root / f"finetune_script_iter{attempt}.py"
        _write_text(script_path, script_text)

        # Execute step
        run_result = registry.run(
            "run_finetune_script",
            script_path=script_path,
            work_dir=run_root,
            conda_env=args.conda_env,
        )
        run_payload: dict[str, Any] = run_result.payload if run_result.ok else {
            "returncode": -1,
            "stdout": "",
            "stderr": run_result.error or "Unknown error",
        }
        _write_text(iter_dir / "run_result.json", json.dumps(run_payload, indent=2))

        # Feedback step
        feedback_system, feedback_prompt = build_feedback_prompts(
            task=task_text,
            plan=plan_json,
            run_output=run_payload,
            script=script_text,
        )
        _write_text(iter_dir / "feedback_prompt.txt", feedback_prompt)
        feedback_reply = _call_llm(feedback_prompt, feedback_system, args)
        _write_text(iter_dir / "feedback_reply.txt", feedback_reply)

        try:
            feedback_json = extract_json(feedback_reply)
        except ValueError:
            feedback_json = {"status": "failure", "issues": ["Could not parse feedback JSON"]}
        _write_text(iter_dir / "feedback.json", json.dumps(feedback_json, indent=2))

        # Check status
        status = feedback_json.get("status", "failure") if isinstance(feedback_json, dict) else "failure"
        print(f"[Iter {attempt}] Status: {status}")

        if status == "success":
            print(f"[Iter {attempt}] Success!")
            break

        if attempt < args.max_retries - 1:
            print(f"[Iter {attempt}] Failure. Retrying ({attempt + 1}/{args.max_retries})...")
        last_feedback = feedback_json
        last_script = script_text
        last_run_output = run_payload
    else:
        print(f"All {args.max_retries} iterations exhausted.")

    # Save the final script as finetune_script.py for easy access
    _write_text(run_root / "finetune_script.py", script_text)

    print(f"Run complete. Outputs saved to {run_root}")


if __name__ == "__main__":
    main()
