from __future__ import annotations

import json
from string import Template
from textwrap import dedent
from typing import Any, Tuple


def _escape(s: str) -> str:
    """Escape $ for string.Template substitution."""
    return s.replace("$", "$$")


# --------------- Initial coder prompts ---------------

system_prompt_tmpl = Template(
    dedent(
        """\
You are the coding agent. Generate a runnable finetuning script based on the plan.
Return only a python code block.

Rules:
- The script must be runnable with python3.
- Keep dependencies minimal and inline configuration.
- Write logs to stdout for the feedback agent.
- Use ONLY the package APIs compatible with the versions listed in the environment info.
- Set CUDA_VISIBLE_DEVICES at the top of the script to use only the assigned GPUs.
"""
    )
)

instruction_tmpl = Template(
    dedent(
        """\
# Task
$task

# Plan (JSON)
$plan_json

# Runtime Environment
$env_info

Produce the finetuning script as:
```python
# code here
```
"""
    )
)


def build_coder_prompts(
    *, task: str, plan: dict[str, Any], env_info: str = "",
) -> Tuple[str, str]:
    system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        task=_escape(task.strip()),
        plan_json=_escape(json.dumps(plan, indent=2) if isinstance(plan, dict) else str(plan)),
        env_info=_escape(env_info) if env_info else "Not available.",
    )
    return system_prompt, instruction


# --------------- Fix / retry prompts ---------------

fix_system_prompt_tmpl = Template(
    dedent(
        """\
You are the coding agent. A previous version of the finetuning script failed.
Fix the script based on the error output and feedback. Return only a python code block.

Rules:
- The script must be runnable with python3.
- Keep dependencies minimal and inline configuration.
- Write logs to stdout for the feedback agent.
- Use ONLY the package APIs compatible with the versions listed in the environment info.
- Set CUDA_VISIBLE_DEVICES at the top of the script to use only the assigned GPUs.
- Fix ALL issues identified in the feedback.
"""
    )
)

fix_instruction_tmpl = Template(
    dedent(
        """\
# Task
$task

# Plan (JSON)
$plan_json

# Runtime Environment
$env_info

# Previous Script (FAILED)
```python
$previous_script
```

# Execution Output
$run_output

# Feedback (issues found)
$feedback_json

Fix all issues and produce a corrected finetuning script as:
```python
# code here
```
"""
    )
)


def build_coder_fix_prompts(
    *,
    task: str,
    plan: dict[str, Any],
    previous_script: str,
    feedback: dict[str, Any],
    run_output: dict[str, Any],
    env_info: str = "",
) -> Tuple[str, str]:
    # Truncate long stdout/stderr to last 200 lines
    run_out_copy = dict(run_output)
    for key in ("stdout", "stderr"):
        text = run_out_copy.get(key, "")
        lines = text.splitlines()
        if len(lines) > 200:
            run_out_copy[key] = "\n".join(["... (truncated) ..."] + lines[-200:])

    system_prompt = fix_system_prompt_tmpl.substitute()
    instruction = fix_instruction_tmpl.substitute(
        task=_escape(task.strip()),
        plan_json=_escape(json.dumps(plan, indent=2) if isinstance(plan, dict) else str(plan)),
        env_info=_escape(env_info) if env_info else "Not available.",
        previous_script=_escape(previous_script.strip()),
        run_output=_escape(json.dumps(run_out_copy, indent=2)),
        feedback_json=_escape(json.dumps(feedback, indent=2) if isinstance(feedback, dict) else str(feedback)),
    )
    return system_prompt, instruction
