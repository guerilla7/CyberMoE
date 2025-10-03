#!/usr/bin/env python3
"""
Supervised fine-tuning of CyberMoE using feedback.jsonl

Uses user_feedback (correct/incorrect) and optional correction label
to create training targets and continue training CyberMoE.
"""
from __future__ import annotations
import os
import json
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import CyberMoE, NUM_EXPERTS


DEFAULT_FEEDBACK_PATH = os.path.join("data", "feedback.jsonl")
FINETUNED_CHECKPOINT = os.path.join("checkpoints", "cybermoe_finetuned.pt")


class FeedbackFTDataset(Dataset):
    def __init__(self, feedback_path: str = DEFAULT_FEEDBACK_PATH) -> None:
        self.items: List[Dict[str, Any]] = []
        if not os.path.exists(feedback_path):
            return
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # Require user_input and at least a binary signal
                if not rec.get("user_input"):
                    continue
                self.items.append(rec)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        text: str = rec.get("user_input", "")
        user_ok: bool = bool(rec.get("user_feedback", False))
        # If wrong and correction provided, use correction; else fallback to current pred
        correction = rec.get("correction")
        if correction is not None:
            target = 1 if str(correction).lower() == "malicious" else 0
        else:
            pred_label = rec.get("pred_label", "Benign")
            # If user said correct, keep pred; else flip for a weak signal
            if user_ok:
                target = 1 if str(pred_label).lower() == "malicious" else 0
            else:
                target = 0 if str(pred_label).lower() == "malicious" else 1
        return text, torch.tensor(target, dtype=torch.long)


def collate_texts(batch):
    texts = [t for t, _ in batch]
    # Ensure labels are consistent 1D tensor
    labels = torch.tensor([int(y.item()) if isinstance(y, torch.Tensor) else int(y) for _, y in batch], dtype=torch.long)
    return texts, labels


def finetune_from_feedback(
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-5,
    feedback_path: str = DEFAULT_FEEDBACK_PATH,
    device: str | torch.device = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = FeedbackFTDataset(feedback_path)
    if len(ds) == 0:
        raise RuntimeError("No feedback data available for fine-tuning.")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_texts)

    model = CyberMoE(top_k=2).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for texts, labels in dl:
            opt.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs['logits'], labels.to(device))
            loss.backward()
            opt.step()

    # Persist checkpoint
    os.makedirs(os.path.dirname(FINETUNED_CHECKPOINT), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "top_k": model.top_k,
        "num_experts": NUM_EXPERTS,
    }, FINETUNED_CHECKPOINT)

    return model


def load_finetuned_model(device: str | torch.device = None):
    """Load finetuned model checkpoint if present; return None if missing."""
    ckpt_path = FINETUNED_CHECKPOINT
    if not os.path.exists(ckpt_path):
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    top_k = int(ckpt.get("top_k", 2))
    model = CyberMoE(top_k=top_k).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    model = finetune_from_feedback()
    print("Fine-tuning completed from feedback.")
