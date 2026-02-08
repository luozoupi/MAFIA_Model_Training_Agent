# MAFIA: Multi-Agent Finetuning Infrastructure with Automation

MAFIA is a multi-agent framework that automates end-to-end model finetuning. Given a task description (plain text or structured JSON), it orchestrates three LLM agents — Planner, Coder, and Feedback — in a self-correcting loop to produce, execute, and iteratively fix training scripts until the run succeeds.

<img src="./pic/human_agents_v2.png">

## Architecture

```
Task Input ──> Planner Agent ──> Finetuning Plan (JSON)
                                       │
                          ┌─────────────┘
                          v
                    Coder Agent ──> Training Script (.py)
                          │
                          v
                    Execute Script ──> stdout / stderr
                          │
                          v
                   Feedback Agent ──> Status + Issues (JSON)
                          │
                    ┌─────┴─────┐
                    │           │
                 success     failure
                    │           │
                  Done    ──> Coder Agent (with error context)
                              (retry up to --max_retries)
```

**Key features:**
- **Self-correcting retry loop** — failed scripts are fed back to the Coder with error output and feedback analysis
- **Environment-aware code generation** — the Coder receives installed package versions, GPU specs, and assigned devices
- **Pre-execution syntax validation** — catches syntax errors instantly before running the script
- **Conda environment isolation** — training scripts run inside a specified conda env via `conda run`
- **Structured task input** — accepts plain text or rich JSON request dicts with model/dataset metadata

## Quick Start

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate py310

# Or install key packages manually:
pip install torch transformers peft trl datasets accelerate bitsandbytes evaluate openai
```

For Hugging Face gated models or datasets, set `HF_TOKEN` if needed.

### 2. Start an LLM backend

The agents need an LLM to generate plans and code. Options:

```bash
# Local vLLM server (recommended)
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 2

# Or use any OpenAI-compatible API by setting OPENAI_API_KEY
```

### 3. Run

```bash
# Plain text task
python3 mcp_main.py "Finetune a small model for IMDB sentiment classification" \
  --server_type vllm \
  --model_name "Qwen/Qwen3-32B" \
  --conda_env py310 \
  --gpu_ids 0 \
  --max_retries 3

# Structured JSON request
python3 mcp_main.py fake_request_dict_legal.json \
  --server_type vllm \
  --model_name "Qwen/Qwen3-32B" \
  --conda_env py310 \
  --gpu_ids 0 1
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `task` | (required) | Task description, text file path, or JSON request file |
| `--server_type` | `openai` | LLM backend: `vllm`, `openai`, `anthropic`, `sglang`, `deepseek`, `together`, `google`, `sambanova`, `fireworks` |
| `--server_address` | `localhost` | LLM server address |
| `--server_port` | `8000` | LLM server port |
| `--model_name` | `gpt-4o-mini` | Model name for the LLM backend |
| `--max_tokens` | `4096` | Max tokens per LLM call |
| `--temperature` | `0.2` | Sampling temperature |
| `--conda_env` | `None` | Conda env name for training script execution |
| `--gpu_ids` | `None` | GPU indices for training (e.g., `--gpu_ids 0 1`) |
| `--max_retries` | `3` | Max coder-execute-feedback iterations |
| `--model_catalog` | `mcp/model_catalog.json` | Path to model catalog |

## Project Structure

```
MAFIA_Model_Training_Agent/
├── mcp_main.py                 # Entry point — orchestrates the agent loop
├── agents/
│   ├── query_server.py         # Multi-backend LLM client (10+ providers)
│   └── llm_local.py            # vLLM OpenAI-compatible client
├── mcp/
│   ├── tooling.py              # MCPToolRegistry and ToolResult
│   ├── tools.py                # scan_hf_models, run_finetune_script
│   └── model_catalog.json      # Lightweight model catalog
├── prompts/
│   ├── mcp_planner.py          # Planner agent prompts
│   ├── mcp_coder.py            # Coder agent prompts (initial + fix/retry)
│   └── mcp_feedback.py         # Feedback agent prompts
├── utils/
│   ├── env_info.py             # Runtime environment info for prompt injection
│   └── kernel_io.py            # Code block / JSON extraction utilities
├── fake_request_dict_legal.json  # Example structured request
├── environment.yml             # Conda environment spec
└── docs/
    └── finetuning.md           # Detailed finetuning guide
```

## Output Structure

Each run creates a timestamped directory under `mcp_runs/`:

```
mcp_runs/20260207_235328/
├── finetune_script.py              # Final training script
├── finetune_script_iter0.py        # Iteration 0 script
├── finetune_script_iter1.py        # Iteration 1 script (fixed)
└── llm_io/
    ├── task_input.txt              # Formatted task description
    ├── env_info.txt                # Detected runtime environment
    ├── planner_prompt.txt          # Planner input
    ├── planner_reply.txt           # Planner raw output
    ├── planner_plan.json           # Extracted plan
    ├── iter_0/
    │   ├── coder_prompt.txt
    │   ├── coder_reply.txt
    │   ├── run_result.json         # Script execution result
    │   ├── feedback_prompt.txt
    │   ├── feedback_reply.txt
    │   └── feedback.json           # Extracted feedback
    ├── iter_1/                     # Retry iteration (with fix context)
    │   └── ...
    └── iter_2/
        └── ...
```

## Structured JSON Input

Instead of a plain text task, you can provide a JSON request dict with rich metadata. See [fake_request_dict_legal.json](fake_request_dict_legal.json) for the full schema. Key fields:

```json
{
  "project_goal": "Build a tiny LLM for legal document summarization.",
  "model_preference": "small-scale LLM",
  "gpu_resources": "2x A100 80GB",
  "selected_model": { "id": "Qwen/Qwen2.5-7B-Instruct", "why": "..." },
  "dataset_plan": {
    "selected_open_datasets": [
      { "id": "FiscalNote/billsum", "why": "..." }
    ]
  },
  "constraints_and_compliance": "No PII; plain-language output.",
  "selection_metadata": { ... }
}
```

The pipeline automatically extracts and formats this into a structured task description for the agents.

## Extending

- **Add models**: edit `mcp/model_catalog.json` or point `--model_catalog` to your own catalog
- **Add tools**: register new tools in `mcp_main.py:_build_registry()` and implement them in `mcp/tools.py`
- **Customize prompts**: modify templates in `prompts/mcp_planner.py`, `prompts/mcp_coder.py`, `prompts/mcp_feedback.py`
- **Add environment packages**: update the `packages` list in `utils/env_info.py:gather_env_info()` to surface additional version info to the Coder
