from __future__ import annotations

from string import Template
from textwrap import dedent
from typing import Any, Tuple

system_prompt_tmpl = Template(
    dedent(
        """
You are a planning agent for an MCP-enabled finetuning workflow.
Create a concise finetuning plan based on the task description and available models.
Return JSON only.

Output format (JSON):
```json
{
  "task_summary": "<max 40 words>",
  "recommended_model": "<model id>",
  "training_recipe": {
    "method": "<LoRA/QLoRA/full>",
    "epochs": "<number>",
    "batch_size": "<number>",
    "learning_rate": "<number>",
    "dataset": "<name or path>",
    "metrics": ["<metric1>", "<metric2>"]
  },
  "tool_calls": [
    {"tool": "scan_hf_models", "args": {"query": "<string>", "limit": 5}},
    {"tool": "run_finetune_script", "args": {"script_path": "<path>"}}
  ]
}
```
"""
    )
)

instruction_tmpl = Template(
    dedent(
        """
# Task
$task

# Model scan results (if any)
$model_scan

Provide the JSON plan. Keep it executable and aligned with the available models.
"""
    )
)


def build_planner_prompts(*, task: str, model_scan: list[dict[str, Any]]) -> Tuple[str, str]:
    system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        task=task.strip(),
        model_scan=model_scan,
    )
    return system_prompt, instruction
