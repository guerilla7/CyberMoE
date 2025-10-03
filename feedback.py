#!/usr/bin/env python3
"""
Lightweight RLHF feedback logger

Appends JSONL feedback entries with model outputs and user evaluation.
Later can be used to fine-tune or train a reward model.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Dict


DEFAULT_FEEDBACK_PATH = os.path.join("data", "feedback.jsonl")


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def log_feedback(record: Dict[str, Any], path: str = DEFAULT_FEEDBACK_PATH) -> None:
    """Append a single feedback record as JSONL.

    Expected record keys (suggested):
    - timestamp, user_input, pred_label, pred_confidence
    - domain_pred, domain_confidence, gating_scores (list[float])
    - expert_logits (list[list[float]]), features (dict), domain_scores (list[float])
    - user_feedback (bool), correction (str|None), notes (str|None)
    """
    ensure_dir(path)
    if "timestamp" not in record:
        record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = ["log_feedback", "DEFAULT_FEEDBACK_PATH"]
