from __future__ import annotations

from typing import Any

REQUIRED_TOP_LEVEL_KEYS = [
    "project_goal",
    "model_preference",
    "post_train_time_budget",
    "gpu_resources",
    "dataset_support_requirements",
    "constraints_and_compliance",
    "inference_stack",
]


def validate_request_payload(payload: dict[str, Any]) -> list[str]:
    """Return a list of validation errors. Empty list means valid enough for planning."""
    errors: list[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload or payload[key] in (None, ""):
            errors.append(f"Missing required field: {key}")

    selected_model = payload.get("selected_model")
    if selected_model is not None:
        if not isinstance(selected_model, dict):
            errors.append("selected_model must be an object")
        else:
            if not selected_model.get("id"):
                errors.append("selected_model.id is required when selected_model is provided")

    dataset_plan = payload.get("dataset_plan")
    if dataset_plan is not None and not isinstance(dataset_plan, dict):
        errors.append("dataset_plan must be an object")

    backup_models = payload.get("backup_models")
    if backup_models is not None and not isinstance(backup_models, list):
        errors.append("backup_models must be a list")

    return errors
