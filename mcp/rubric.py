from __future__ import annotations

from typing import Any


def evaluate_completion(
    *,
    plan: dict[str, Any],
    feedback: dict[str, Any],
    run_payload: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate deterministic completion gates for early-stop decisions."""
    status = str(feedback.get("status", "")).lower()
    rubric = feedback.get("rubric", {}) if isinstance(feedback, dict) else {}
    criteria_met_flag = str(rubric.get("criteria_met", "")).lower() == "true"

    run_ok = int(run_payload.get("returncode", 1) or 1) == 0

    success_criteria = plan.get("success_criteria", {}) if isinstance(plan, dict) else {}
    primary_metric = str(success_criteria.get("metric", "")).strip()
    target = success_criteria.get("target")

    feedback_metrics = feedback.get("metrics", {}) if isinstance(feedback, dict) else {}
    metric_available = primary_metric in feedback_metrics if primary_metric else False

    gates = {
        "status_success": status == "success",
        "rubric_criteria_met": criteria_met_flag,
        "run_returncode_zero": run_ok,
        "target_metric_present": metric_available,
    }

    completed = (gates["status_success"] or gates["rubric_criteria_met"]) and gates["run_returncode_zero"]
    if primary_metric and target is not None:
        completed = completed and gates["target_metric_present"]

    return {
        "completed": completed,
        "gates": gates,
        "status": status,
        "primary_metric": primary_metric,
        "target": target,
        "feedback_metric_value": feedback_metrics.get(primary_metric) if primary_metric else None,
    }
