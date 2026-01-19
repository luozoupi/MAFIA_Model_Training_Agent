from __future__ import annotations

from string import Template
from textwrap import dedent
from typing import Any, Tuple

system_prompt_tmpl = Template(
    dedent(
        """
You are the coding agent. Generate a runnable finetuning script based on the plan.
Return only a python code block.

Rules:
- The script must be runnable with python3.
- Keep dependencies minimal and inline configuration.
- Write logs to stdout for the feedback agent.
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

Produce the finetuning script as:
```python
# code here
```
"""
    )
)


def build_coder_prompts(*, task: str, plan: dict[str, Any]) -> Tuple[str, str]:
    system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        task=task.strip(),
        plan_json=plan,
    )
    return system_prompt, instruction
