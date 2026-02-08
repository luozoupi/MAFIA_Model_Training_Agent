# Multi-Agent Finetuning Guide

## Overview

The MAFIA workflow orchestrates three agents in an iterative loop:

1. **Planner** — scans a model catalog and proposes a finetuning plan (model, method, hyperparameters, dataset, metrics).
2. **Coder** — generates a runnable Python training script from the plan, aware of the runtime environment (package versions, GPU specs).
3. **Feedback** — analyzes the script execution results and returns a status (success/failure) with identified issues.

On failure, the Coder receives the previous script, error output, and feedback, and generates a corrected version. This repeats up to `--max_retries` times.

## Inputs

Three input formats are supported:

```bash
# Plain text task description
python3 mcp_main.py "Finetune a lightweight model for ticket triage"

# Path to a text file
python3 mcp_main.py tasks/my_task.txt

# Structured JSON request with model/dataset metadata
python3 mcp_main.py fake_request_dict_legal.json
```

JSON requests are automatically parsed and formatted into a rich task description. Plain text and file inputs are passed through directly.

## Environment Awareness

The Coder agent receives runtime environment info injected into its prompt:

- Python version
- Installed package versions (transformers, peft, trl, torch, datasets, etc.)
- GPU names, memory, and which devices are assigned

This helps it generate scripts that use correct API names and avoid deprecated parameters. The info is collected by `utils/env_info.py` and saved to `llm_io/env_info.txt` for reference.

Control the environment with CLI flags:

```bash
--conda_env py310    # Run training scripts inside this conda env
--gpu_ids 0 1        # Assign specific GPUs (sets CUDA_VISIBLE_DEVICES in the script)
```

## Tooling

The MCP runner registers two tools:

- **`scan_hf_models`** — searches a lightweight JSON catalog (`mcp/model_catalog.json`) for candidate models matching the task. Uses word-level keyword matching across model ID, task, and notes fields; falls back to returning all entries if nothing matches.

- **`run_finetune_script`** — executes the generated training script:
  1. Validates syntax with `compile()` before running (catches errors instantly)
  2. Runs via `conda run -n <env>` if `--conda_env` is set, otherwise bare `python3`
  3. Captures stdout, stderr, and return code
  4. 2-hour timeout for long training runs

## Retry Loop

When the Feedback agent reports `"status": "failure"`, the pipeline enters a fix cycle:

1. The Coder receives a **fix prompt** containing:
   - The original task and plan
   - The failed script
   - The execution output (stdout/stderr, truncated to last 200 lines)
   - The feedback JSON with identified issues and suggestions
   - The runtime environment info

2. The Coder generates a corrected script.

3. The corrected script is executed and evaluated again.

Each iteration's artifacts are saved in separate directories (`iter_0/`, `iter_1/`, etc.) so you can inspect the full history.

## Output Structure

```
mcp_runs/20260207_235328/
├── finetune_script.py              # Final version (copy of last iteration)
├── finetune_script_iter0.py        # Original attempt
├── finetune_script_iter1.py        # First fix
├── finetune_script_iter2.py        # Second fix
└── llm_io/
    ├── task_input.txt              # Formatted task description
    ├── env_info.txt                # Detected runtime environment
    ├── planner_prompt.txt
    ├── planner_reply.txt
    ├── planner_plan.json
    ├── iter_0/
    │   ├── coder_prompt.txt        # Initial coder prompt
    │   ├── coder_reply.txt
    │   ├── run_result.json         # {returncode, stdout, stderr}
    │   ├── feedback_prompt.txt
    │   ├── feedback_reply.txt
    │   └── feedback.json           # {status, issues, next_round_suggestions, metrics}
    ├── iter_1/
    │   ├── coder_prompt.txt        # Fix prompt (includes previous script + errors)
    │   └── ...
    └── iter_2/
        └── ...
```

## Backends

| Backend | Flag | Notes |
|---------|------|-------|
| vLLM (local) | `--server_type vllm --server_address localhost --server_port 8000` | Recommended for local GPUs |
| OpenAI | `--server_type openai` | Requires `OPENAI_API_KEY` |
| Anthropic | `--server_type anthropic` | Requires `ANTHROPIC_API_KEY` |
| DeepSeek | `--server_type deepseek` | Requires `DEEPSEEK_API_KEY` |
| Google Gemini | `--server_type google` | Requires `GEMINI_API_KEY` |
| Together | `--server_type together` | Requires `TOGETHER_API_KEY` |
| SambaNova | `--server_type sambanova` | Requires `SAMBANOVA_API_KEY` |
| Fireworks | `--server_type fireworks` | Requires `FIREWORKS_API_KEY` |
| SGLang | `--server_type sglang` | Requires local SGLang server |

## Tips

- Start with `--max_retries 3` — most issues (syntax errors, deprecated APIs, wrong column names) are fixed within 1-2 retries.
- Use `--gpu_ids` to isolate training from your LLM inference server (e.g., `--gpu_ids 0` if vLLM is on GPUs 3,4).
- Check `env_info.txt` in the run directory to verify the Coder saw the correct package versions.
- If a run times out, consider adding guidance to the task description (e.g., "use a subset of 2000 samples for fast iteration").
