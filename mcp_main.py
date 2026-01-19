from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.query_server import query_server
from mcp.tooling import MCPToolRegistry
from mcp.tools import run_finetune_script, scan_hf_models
from prompts.mcp_coder import build_coder_prompts
from prompts.mcp_feedback import build_feedback_prompts
from prompts.mcp_planner import build_planner_prompts
from utils.kernel_io import extract_code_block, extract_json


def _read_task_text(task_arg: str) -> str:
    path = Path(task_arg)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return task_arg


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


def main() -> None:
    parser = argparse.ArgumentParser("MCP multi-agent finetuning workflow")
    parser.add_argument("task", help="Task description text or path to a task file")
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
    args = parser.parse_args()

    task_text = _read_task_text(args.task)
    registry = _build_registry()
    catalog_path = Path(args.model_catalog) if args.model_catalog else None

    run_root = Path(args.run_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    io_dir = run_root / "llm_io"

    scan_result = registry.run(
        "scan_hf_models",
        catalog_path=catalog_path,
        task=task_text,
        limit=args.model_limit,
    )


    planner_system, planner_prompt = build_planner_prompts(
        task=task_text,
        model_scan=model_scan,
    )
    _write_text(io_dir / "planner_prompt.txt", planner_prompt)
    planner_reply = _call_llm(planner_prompt, planner_system, args)
    _write_text(io_dir / "planner_reply.txt", planner_reply)
    plan_json = extract_json(planner_reply)
    _write_text(io_dir / "planner_plan.json", json.dumps(plan_json, indent=2))

    coder_system, coder_prompt = build_coder_prompts(task=task_text, plan=plan_json)
    _write_text(io_dir / "coder_prompt.txt", coder_prompt)
    coder_reply = _call_llm(coder_prompt, coder_system, args)
    _write_text(io_dir / "coder_reply.txt", coder_reply)

    script_text = _safe_extract_code(coder_reply)
    script_path = run_root / "finetune_script.py"
    _write_text(script_path, script_text)

    run_result = registry.run(
        "run_finetune_script",
        script_path=script_path,
        work_dir=run_root,
    )
    run_payload: dict[str, Any] = run_result.payload if run_result.ok else {
        "returncode": -1,
        "stdout": "",
        "stderr": run_result.error or "Unknown error",
    }
    _write_text(io_dir / "run_result.json", json.dumps(run_payload, indent=2))

    feedback_system, feedback_prompt = build_feedback_prompts(
        task=task_text,
        plan=plan_json,
        run_output=run_payload,
        script=script_text,
    )
    _write_text(io_dir / "feedback_prompt.txt", feedback_prompt)
    feedback_reply = _call_llm(feedback_prompt, feedback_system, args)
    _write_text(io_dir / "feedback_reply.txt", feedback_reply)
    feedback_json = extract_json(feedback_reply)
    _write_text(io_dir / "feedback.json", json.dumps(feedback_json, indent=2))

    print(f"Run complete. Outputs saved to {run_root}")


if __name__ == "__main__":
    main()
