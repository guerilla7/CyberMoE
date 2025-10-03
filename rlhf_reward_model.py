#!/usr/bin/env python3
"""
Train a tiny reward model from feedback JSONL.

The reward model predicts correctness (1 for correct, 0 for incorrect)
based on model outputs logged in feedback.jsonl.
"""
from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEFAULT_FEEDBACK_PATH = os.path.join("data", "feedback.jsonl")
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "reward_model.pt")


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _pad_or_truncate(lst: List[float], length: int, pad_value: float = 0.0) -> List[float]:
    if len(lst) >= length:
        return lst[:length]
    return lst + [pad_value] * (length - len(lst))


class FeedbackRewardDataset(Dataset):
    def __init__(self, feedback_path: str = DEFAULT_FEEDBACK_PATH,
                 num_experts: int = 5, num_domains: int = 5) -> None:
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if not os.path.exists(feedback_path):
            return
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                # Build feature vector
                pred_conf = float(rec.get("pred_confidence", 0.0))
                domain_conf = float(rec.get("domain_confidence", 0.0))

                gating = rec.get("gating_scores", []) or []
                gating = [float(x) for x in gating]
                gating = _pad_or_truncate(gating, num_experts)

                expert_logits = rec.get("expert_logits", []) or []
                # expert_logits expected shape ~ [num_experts, 2]
                flat_expert = []
                for i in range(num_experts):
                    row = expert_logits[i] if i < len(expert_logits) else [0.0, 0.0]
                    if isinstance(row, list):
                        flat_expert.extend([float(v) for v in _pad_or_truncate(row, 2)])
                    else:
                        flat_expert.extend([0.0, 0.0])

                domain_scores = rec.get("domain_scores", []) or []
                domain_scores = [float(x) for x in domain_scores]
                domain_scores = _pad_or_truncate(domain_scores, num_domains)

                feats = [pred_conf, domain_conf] + flat_expert + gating + domain_scores
                x = torch.tensor(feats, dtype=torch.float32)

                # Target from feedback: True -> 1.0, False -> 0.0
                y_bool = bool(rec.get("user_feedback", False))
                y = torch.tensor([1.0 if y_bool else 0.0], dtype=torch.float32)
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class TinyRewardModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1)  # Logit for reward
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def infer_input_dim(num_experts: int = 5, num_domains: int = 5) -> int:
    # pred_conf(1) + domain_conf(1) + expert_logits(num_experts*2) + gating(num_experts) + domain_scores(num_domains)
    return 1 + 1 + (num_experts * 2) + num_experts + num_domains


def train_reward_model(
    feedback_path: str = DEFAULT_FEEDBACK_PATH,
    save_path: str = DEFAULT_CHECKPOINT,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_experts: int = 5,
    num_domains: int = 5,
    device: str | torch.device = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = FeedbackRewardDataset(feedback_path, num_experts=num_experts, num_domains=num_domains)
    if len(ds) == 0:
        raise RuntimeError("No feedback data found to train reward model.")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    input_dim = infer_input_dim(num_experts, num_domains)
    model = TinyRewardModel(input_dim=input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

    _ensure_dir(save_path)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_experts": num_experts,
        "num_domains": num_domains,
    }, save_path)
    return save_path


if __name__ == "__main__":
    path = train_reward_model()
    print(f"Saved reward model to: {path}")
