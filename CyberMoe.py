#!/usr/bin/env python3
"""
CyberMoE – Minimal Mixture‑of‑Experts for Adaptive Cybersecurity

Author:  Ron F. Del Rosario
Date:    2025‑09‑11
"""
#!/usr/bin/env python3
"""
CyberMoE – Minimal Mixture-of-Experts for Adaptive Cybersecurity

Author:  Ron F. Del Rosario
Date:    2025-09-11
"""

# --------------------------------------------------------------------------- #
# Imports – only PyTorch and HuggingFace Transformers are required.
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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
# 3️⃣ Fusion Network – aggregates weighted expert outputs
# --------------------------------------------------------------------------- #
class FusionNetwork(nn.Module):
    """
    Takes the weighted logits from all experts and produces a final decision.
    """
    def __init__(self, num_experts: int = NUM_EXPERTS,
                 expert_labels: int = EXPERT_LABELS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_experts * expert_labels, 128),
            nn.ReLU(),
            nn.Linear(128, expert_labels)
        )

    def forward(self, weighted_logits: torch.Tensor) -> torch.Tensor:
        """
        :param weighted_logits: (batch, num_experts, expert_labels)
        :return: final logits of shape (batch, expert_labels)
        """
        flat = weighted_logits.view(weighted_logits.size(0), -1)  # flatten
        return self.mlp(flat)

# --------------------------------------------------------------------------- #
# 4️⃣ CyberMoE – the full model
# --------------------------------------------------------------------------- #
class CyberMoE(nn.Module):
    """
    A minimal Mixture‑of‑Experts cybersecurity model.
    """
    def __init__(self, num_experts: int = NUM_EXPERTS,
                 expert_labels: int = EXPERT_LABELS):
        super().__init__()

        # Shared encoder & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoder   = AutoModel.from_pretrained("distilbert-base-uncased")

        hidden_size = self.encoder.config.hidden_size

        # Sub‑components
        self.gating_net = GatingNetwork(hidden_size, num_experts)
        self.experts    = nn.ModuleList(
            [Expert(hidden_size, expert_labels) for _ in range(num_experts)]
        )
        self.fusion_net = FusionNetwork(num_experts, expert_labels)

    # ----------------------------------------------------------------------- #
    def forward(self, texts: list[str]) -> tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
        """
        :param texts: list of raw strings (batch)
        :return:
            final_logits  – (batch, expert_labels)   -> final decision
            gating_probs  – (batch, num_experts)     -> why we routed to experts
            expert_logits – (batch, num_experts, expert_labels)
        """
        # Tokenisation
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        # Shared encoder
        outputs = self.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Gating probabilities
        gating_probs = self.gating_net(hidden_state)  # (batch, num_experts)

        # Experts produce logits
        expert_logits = []
        for expert in self.experts:
            logits = expert(hidden_state)          # (batch, expert_labels)
            expert_logits.append(logits.unsqueeze(1))  # add expert dim
        expert_logits = torch.cat(expert_logits, dim=1)   # (batch, num_experts, expert_labels)

        # Weight by gating probabilities
        weighted_logits = expert_logits * gating_probs.unsqueeze(-1)  # broadcast

        # Fusion to final decision
        final_logits = self.fusion_net(weighted_logits)  # (batch, expert_labels)

        return final_logits, gating_probs, expert_logits

# --------------------------------------------------------------------------- #
# 5️⃣ Demo – inference & explanation
# --------------------------------------------------------------------------- #
def demo():
    # Instantiate model
    model = CyberMoE().to(DEVICE)
    model.eval()  # inference mode

    # Example inputs (feel free to add your own)
    texts = [
        "Suspicious login attempt from unknown IP address",
        "New vulnerability discovered in Apache HTTP Server",
        "Phishing email with malicious attachment detected",
        "Normal user activity: accessing internal portal",
        "Malware sample shows code injection in system DLL"
    ]

    with torch.no_grad():
        final_logits, gating_probs, expert_logits = model(texts)

    # Convert to probabilities
    probs = torch.softmax(final_logits, dim=-1).cpu().numpy()

    # Print results
    for i, text in enumerate(texts):
        print("\n" + "=" * 80)
        print(f"[{i}] Input: {text}")

        # Final prediction
        pred = int(probs[i].argmax())
        print(f"  -> Predicted label: {pred} (prob={probs[i][pred]:.3f})")

        # Gating probabilities
        gate = gating_probs[i].cpu().numpy()
        print(f"  Gating probs: {gate}")

        # Top‑k experts (for explanation)
        topk_vals, topk_idx = torch.topk(gating_probs[i], TOP_K)
        print(f"  Top-{TOP_K} experts: {topk_idx.tolist()} (scores={topk_vals.numpy().tolist()})")

        # Expert logits
        for j, exp in enumerate(expert_logits[i].cpu().numpy()):
            print(f"    Expert {j} logits: {exp}")

# --------------------------------------------------------------------------- #
# 6️⃣ Optional training loop (synthetic data)
# --------------------------------------------------------------------------- #
def train_demo():
    """
    Very small training demo on synthetic data.
    In practice you would use real labeled logs, malware samples,
    phishing emails, etc.  Each domain expert should be fine‑tuned
    on its own data first.
    """
    # Simple synthetic dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=2000):
            self.texts = [f"sample {i}" for i in range(num_samples)]
            # Random labels 0/1
            self.labels = torch.randint(0, EXPERT_LABELS, (num_samples,))

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    # Collate function that returns a list of strings and a tensor of labels
    def collate_fn(batch):
        texts, labels = zip(*batch)
        return list(texts), torch.tensor(labels)

    dataset = DummyDataset()
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )

    # Model & optimizer
    model = CyberMoE().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop (few epochs for demo)
    from sklearn.metrics import accuracy_score, confusion_matrix
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        all_preds = []
        all_labels = []
        for texts, labels in loader:
            optimizer.zero_grad()
            logits, _, _ = model(texts)
            loss = criterion(logits, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    # After training, run the demo inference again
    print("\n--- Post‑training inference ---")
    # Evaluate on demo inputs
    model.eval()
    texts = [
        "Suspicious login attempt from unknown IP address",
        "New vulnerability discovered in Apache HTTP Server",
        "Phishing email with malicious attachment detected",
        "Normal user activity: accessing internal portal",
        "Malware sample shows code injection in system DLL"
    ]
    with torch.no_grad():
        final_logits, gating_probs, expert_logits = model(texts)
    probs = torch.softmax(final_logits, dim=-1).cpu().numpy()
    preds = [int(p.argmax()) for p in probs]
    print("Demo input predictions:")
    for i, text in enumerate(texts):
        print(f"[{i}] {text} -> Predicted label: {preds[i]} (prob={probs[i][preds[i]]:.3f})")
    # Print confusion matrix for demo inputs (assuming true labels: [1,1,1,0,1] for illustration)
    true_demo_labels = [1,1,1,0,1]
    demo_acc = accuracy_score(true_demo_labels, preds)
    demo_cm = confusion_matrix(true_demo_labels, preds)
    print(f"Demo Accuracy: {demo_acc:.4f}")
    print(f"Demo Confusion Matrix:\n{demo_cm}")

# --------------------------------------------------------------------------- #
# 7️⃣ Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Choose one of the two options below:
    # demo()          # fast demo (no training)
    train_demo()  # run the synthetic training loop