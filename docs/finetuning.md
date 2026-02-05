# MCP Multi-Agent Finetuning Guide

## Overview

The MCP workflow orchestrates three agents:

1. **Planner**: selects a candidate model and proposes a finetuning plan.
2. **Coder**: turns the plan into a runnable finetuning script.
3. **Feedback**: executes the script, inspects results, and suggests next steps.

All prompts, replies, and artifacts are stored under `mcp_runs/<timestamp>/llm_io`.

## Inputs

You can pass either a short natural-language task description or a path to a text file:

```bash
python3 mcp_main.py "Finetune a lightweight model for ticket triage"
```

For structured enterprise workflows, pass `--request_json` with fields such as
goal, compute budget, selected/backup models, dataset constraints, and compliance notes.
An example is available at `examples/legal_summarization_request.json`.

## Tooling

The MCP runner registers simple tools:

- `scan_hf_models`: scans a lightweight catalog for candidate models.
- `run_finetune_script`: executes the generated finetuning script and captures stdout/stderr.

Customize or expand tools under `mcp/tools.py`.

## Outputs

Each run produces:

- `planner_prompt.txt` / `planner_reply.txt` / `planner_plan.json`
- `coder_prompt.txt` / `coder_reply.txt` / `finetune_script.py`
- `run_result.json`
- `feedback_prompt.txt` / `feedback_reply.txt` / `feedback.json`
- `completion_eval.json` (per iteration, deterministic completion gates)

These files let you iterate quickly on the plan or script without rerunning the full loop.

## Backends

- **OpenAI-compatible APIs**: set `OPENAI_API_KEY` and run with `--server_type openai`.
- **vLLM**: start a local server and pass `--server_type vllm --server_address <host> --server_port <port>`.

## YOLOv8 Example

An example YOLOv8 workflow is available under `examples/yolo_v8` with a small `coco128`
training run. See [`examples/yolo_v8/README.md`](../examples/yolo_v8/README.md).

## Next Steps

- Replace the sample catalog (`mcp/model_catalog.json`) with your preferred model list.
- Extend `run_finetune_script` to call your training stack (PEFT, Accelerate, etc.).
- Add evaluation hooks for automatic scoring and regression checks.
