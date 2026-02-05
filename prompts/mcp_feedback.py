from __future__ import annotations

from string import Template
from textwrap import dedent
from typing import Any, Tuple

system_prompt_tmpl = Template(
    dedent(
        """
You are the feedback agent. Review the finetuning run output and suggest next steps.
Return JSON only.

Output format (JSON):
```json
{
  "status": "<success|failure>",
  "issues": ["<issue1>", "<issue2>"],
  "next_round_suggestions": ["<action1>", "<action2>"],
  "metrics": {"<name>": "<value>"},
  "rubric": {
    "score": "<0-1>",
    "criteria_met": "<true|false>",
    "summary": "<max 30 words>"
  }
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

# Plan (JSON)
$plan_json

# Script output
$run_output

# Raw script (for reference)
$script

Analyze and return the JSON feedback.
"""
    )
)


def build_feedback_prompts(
    *,
    task: str,
    plan: dict[str, Any],
    run_output: dict[str, Any],
    script: str,
) -> Tuple[str, str]:
    system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        task=task.strip(),
        plan_json=plan,
        run_output=run_output,
        script=script.strip(),
    )
    return system_prompt, instruction
