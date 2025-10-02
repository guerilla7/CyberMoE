#!/usr/bin/env python3
"""
CyberMoE – Minimal Mixture‑of‑Experts for Adaptive Cybersecurity

Author:  Ron F. Del Rosario
Date:    2025‑09‑11
"""

# --------------------------------------------------------------------------- #
# Imports – only PyTorch and HuggingFace Transformers are required.
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import random
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------------------------------------------------- #
# Hyper‑parameters & constants
# --------------------------------------------------------------------------- #
NUM_EXPERTS   = 3          # Network, Malware, Phishing
EXPERT_LABELS = 2          # 0 – benign, 1 – malicious
TOP_K         = 2          # Optional: number of experts to keep (unused in this demo)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# 1️⃣ Expert – a tiny head that turns the CLS token into logits
# --------------------------------------------------------------------------- #
class Expert(nn.Module):
    """
    A single expert head.
    In a real system this would be a full LLM (e.g., GPT‑4) fine‑tuned on
    its domain.  Here we keep it lightweight for demo purposes.
    """
    def __init__(self, hidden_size: int, num_labels: int = EXPERT_LABELS):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_state: (batch, seq_len, hidden_size) from the shared encoder
        :return: logits of shape (batch, num_labels)
        """
        cls_token = hidden_state[:, 0, :]          # CLS token
        return self.classifier(cls_token)           # (batch, num_labels)

# --------------------------------------------------------------------------- #
# 2️⃣ Gating Network – decides which experts to trust
# --------------------------------------------------------------------------- #
class GatingNetwork(nn.Module):
    """
    Takes the CLS token and outputs a probability distribution over experts.
    """
    def __init__(self, hidden_size: int, num_experts: int = NUM_EXPERTS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_state: (batch, seq_len, hidden_size)
        :return: probs over experts of shape (batch, num_experts)
        """
        cls_token = hidden_state[:, 0, :]
        logits    = self.mlp(cls_token)
        return torch.softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
# 3️⃣ CyberMoE – the full model (now with sparse Top-K routing)
# --------------------------------------------------------------------------- #
class CyberMoE(nn.Module):
    """
    A minimal Mixture‑of‑Experts cybersecurity model.
    This version uses sparse Top-K routing for efficiency.
    """
    def __init__(self, num_experts: int = NUM_EXPERTS,
                 expert_labels: int = EXPERT_LABELS,
                 top_k: int = TOP_K):
        super().__init__()
        self.top_k = top_k

        # Shared encoder & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoder   = AutoModel.from_pretrained("distilbert-base-uncased")

        hidden_size = self.encoder.config.hidden_size

        # Sub‑components
        self.gating_net = GatingNetwork(hidden_size, num_experts)
        self.experts    = nn.ModuleList(
            [Expert(hidden_size, expert_labels) for _ in range(num_experts)]
        )
        # The FusionNetwork has been removed for a sparse implementation

    # ----------------------------------------------------------------------- #
    def forward(self, texts: list[str]) -> tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
        """
        :param texts: list of raw strings (batch)
        :return:
            final_logits  – (batch, expert_labels)   -> final decision
            gating_probs  – (batch, num_experts)     -> raw gating probabilities
            expert_logits – (batch, num_experts, expert_labels) -> (sparse) logits from experts
        """
        # Tokenisation
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)

        # Shared encoder
        hidden_state = self.encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state

        # Gating: get probabilities for all experts
        gating_probs = self.gating_net(hidden_state)  # (batch, num_experts)

        # Routing: select Top-K experts and their probabilities
        top_k_probs, top_k_indices = torch.topk(gating_probs, self.top_k, dim=-1)

        # Normalize the probabilities of the top-k experts, so they sum to 1
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)

        # Initialize tensors to store results
        batch_size, _, hidden_size = hidden_state.shape
        final_logits = torch.zeros(batch_size, EXPERT_LABELS).to(DEVICE)
        # expert_logits is kept to show which experts were activated (it's sparse)
        expert_logits = torch.zeros(batch_size, NUM_EXPERTS, EXPERT_LABELS).to(DEVICE)

        # Loop over batch items for sparse computation (clearer for demo)
        for i in range(batch_size):
            item_hidden_state = hidden_state[i:i+1]
            
            # Get the outputs from the top-k experts for this item
            top_k_expert_outputs = []
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j]
                expert = self.experts[expert_idx]
                
                # Compute and store expert output
                logit = expert(item_hidden_state)
                top_k_expert_outputs.append(logit)
                expert_logits[i, expert_idx] = logit.squeeze(0)

            # Stack the expert outputs for this item
            item_top_k_logits = torch.cat(top_k_expert_outputs, dim=0) # (top_k, expert_labels)

            # Weight the outputs by the normalized gating probabilities
            item_final_logit = torch.sum(
                item_top_k_logits * top_k_probs[i].unsqueeze(-1), dim=0
            )
            
            final_logits[i] = item_final_logit

        return final_logits, gating_probs, expert_logits

def train_model(progress_callback=None):
    """
    Very small training demo on synthetic data.
    In practice you would use real labeled logs, malware samples,
    phishing emails, etc.  Each domain expert should be fine‑tuned
    on its own data first.
    """
    
    # More realistic synthetic dataset with themed keywords
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=2000):
            self.texts = []
            self.labels = []

            themes = {
                "Network": ["IP address", "firewall", "login attempt", "traffic", "port scan"],
                "Malware": ["malware", "trojan", "virus", "injection", "system DLL", "exploit"],
                "Phishing": ["phishing", "email", "attachment", "invoice", "password", "account"],
                "Benign": ["user activity", "internal portal", "accessing file", "normal operation", "scheduled task"]
            }

            for _ in range(num_samples):
                theme_name = random.choice(list(themes.keys()))
                keyword1 = random.choice(themes[theme_name])
                keyword2 = random.choice(themes[theme_name])
                
                self.texts.append(f"Log entry: detected {keyword1} and {keyword2}.")
                
                label = 1 if theme_name != "Benign" else 0
                self.labels.append(label)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    # Collate function that returns a list of strings and a tensor of labels
    def collate_fn(batch):
        texts, labels = zip(*batch)
        return list(texts), torch.tensor(labels, dtype=torch.long)

    dataset = DummyDataset()
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )

    # Model & optimizer
    model = CyberMoE().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop (few epochs for demo)
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        all_preds = []
        all_labels = []
        for i, (texts, labels) in enumerate(loader):
            optimizer.zero_grad()
            logits, _, _ = model(texts)
            loss = criterion(logits, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            if progress_callback:
                progress_callback((epoch * len(loader) + i) / (3 * len(loader)))


        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
    
    if progress_callback:
        progress_callback(1.0)

    return model
