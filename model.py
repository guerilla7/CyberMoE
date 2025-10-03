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
from datasets import load_dataset

# --------------------------------------------------------------------------- #
# Hyper‑parameters & constants
# --------------------------------------------------------------------------- #
NUM_EXPERTS   = 5          # Network, Malware, Phishing, Cloud, Web App
EXPERT_LABELS = 2          # 0 – benign, 1 – malicious
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
    Enhanced gating network that learns richer text representations for expert routing.
    Uses self-attention over sequence tokens and deeper feature extraction.
    """
    def __init__(self, hidden_size: int, num_experts: int = NUM_EXPERTS):
        super().__init__()
        
        # Multi-head self-attention for sequence-level understanding
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Domain-aware context network
        self.context_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Expert routing head
        self.routing_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_experts)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_state: (batch, seq_len, hidden_size)
        :return: probs over experts of shape (batch, num_experts)
        """
        # Apply self-attention over the sequence
        attn_output, _ = self.attention(hidden_state, hidden_state, hidden_state)
        
        # Global sequence representation (mean pooling)
        sequence_repr = attn_output.mean(dim=1)  # (batch, hidden_size)
        
        # Extract rich features
        features = self.feature_net(sequence_repr)  # (batch, hidden_size * 2)
        
        # Learn domain-aware context
        context = self.context_net(features)  # (batch, hidden_size)
        
        # Generate expert routing probabilities
        logits = self.routing_head(context)  # (batch, num_experts)
        return torch.softmax(logits, dim=-1)
        logits = self.mlp(x)
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
                 top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder   = AutoModel.from_pretrained("bert-base-uncased")
        hidden_size = self.encoder.config.hidden_size
        self.gating_net = GatingNetwork(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [Expert(hidden_size, expert_labels) for _ in range(num_experts)]
        )
        
        # Domain prediction head for auxiliary task
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 5)  # 5 domains
        )

    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param texts: list of raw strings (batch)
        :return:
            final_logits  – (batch, expert_labels)   -> final decision
            gating_probs  – (batch, num_experts)     -> raw gating probabilities
            expert_logits – (batch, num_experts, expert_labels) -> (sparse) logits from experts
            domain_logits – (batch, num_domains)     -> domain predictions
        """
        # Tokenisation
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)

        # Shared encoder
        hidden_state = self.encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state

        # Domain prediction
        cls_token = hidden_state[:, 0, :]
        domain_logits = self.domain_head(cls_token)  # (batch, num_domains)

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

        return final_logits, gating_probs, expert_logits, domain_logits

    def explain_gating(self, text: str, target_expert_idx: int):
        """
        Calculates the importance of each word in the input text for the gating
        decision for a specific expert.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        
        # Get the word embeddings
        word_embeddings = self.encoder.embeddings.word_embeddings(input_ids)
        word_embeddings.retain_grad()

        # Forward pass through the encoder
        encoder_output = self.encoder(inputs_embeds=word_embeddings)
        hidden_state = encoder_output.last_hidden_state

        # Gating decision
        gating_probs = self.gating_net(hidden_state)
        target_expert_prob = gating_probs[0, target_expert_idx]

        # Calculate gradients
        target_expert_prob.backward()

        # Get the gradients of the word embeddings
        word_gradients = word_embeddings.grad.squeeze(0)
        
        # Calculate the norm of the gradients for each word
        word_importances = torch.norm(word_gradients, dim=1)
        
        # Normalize the importances
        word_importances = (word_importances - word_importances.min()) / (word_importances.max() - word_importances.min())
        
        # Get the tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        return list(zip(tokens, word_importances.cpu().numpy()))

def train_model(progress_callback=None, weighted_loss=True, aux_loss_weight=1e-2, top_k=2):
    """
    Training demo on a real-world dataset.
    """
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("csv", data_files="cybermoe-dataset.csv")["train"]

    # Collate function that returns a list of strings and a tensor of labels
    def collate_fn(batch):
        # The dataset has 'text', 'domain', and 'label' columns.
        texts = [item['text'] for item in batch if item['text']]
        label_map = {'benign': 0, 'malicious': 1}
        domain_list = ['Network', 'Malware', 'Phishing', 'Cloud', 'Web App']
        domain_map = {d: i for i, d in enumerate(domain_list)}
        labels = []
        domains = []
        for item in batch:
            if item['text']:
                label = item['label']
                if isinstance(label, str):
                    label = label_map.get(label, 0)
                labels.append(label)
                domain = item.get('domain', 'Network')
                domain_onehot = [0] * len(domain_list)
                if domain in domain_map:
                    domain_onehot[domain_map[domain]] = 1
                domains.append(domain_onehot)
        # Debug print to catch unexpected label types
        if any(isinstance(l, str) for l in labels):
            print('Label conversion error:', labels)
        return texts, torch.tensor(labels, dtype=torch.long), torch.tensor(domains, dtype=torch.float)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )

    # Model & optimizer
    model = CyberMoE(top_k=top_k).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Weighted loss
    if weighted_loss:
        # Roughly 5 malicious themes to 1 benign theme
        class_weights = torch.tensor([0.6, 3.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training loop (few epochs for demo)
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        all_preds = []
        all_labels = []
        for i, (texts, labels, domain_features) in enumerate(loader):
            optimizer.zero_grad()
            logits, gating_probs, _, domain_pred = model(texts)
            
            # Main classification loss
            loss = criterion(logits, labels.to(DEVICE))
            
            # Domain prediction loss
            domain_loss = nn.CrossEntropyLoss()(domain_pred, domain_features.to(DEVICE).argmax(dim=1))
            loss = loss + 0.1 * domain_loss
            
            # Auxiliary load balancing loss
            if aux_loss_weight > 0:
                mean_gating_probs = torch.mean(gating_probs, dim=0)
                cv_sq = (torch.std(mean_gating_probs) / torch.mean(mean_gating_probs))**2
                aux_loss = aux_loss_weight * cv_sq
                loss += aux_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            if progress_callback:
                progress_callback((epoch * len(loader) + i) / (5 * len(loader)))


        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
    
    if progress_callback:
        progress_callback(1.0)

    return model

# Example: default to 'Network' domain, or let user select domain from UI
# Example usage (uncomment to test):
# domain_list = ['Network', 'Malware', 'Phishing', 'Cloud', 'Web App']
# domain_map = {d: i for i, d in enumerate(domain_list)}
# user_domain = 'Network'
# domain_onehot = [0] * len(domain_list)
# domain_onehot[domain_map[user_domain]] = 1
# domain_features = torch.tensor([domain_onehot], dtype=torch.float)
# model = train_model()  # Train the model first
# final_logits, gating_probs, expert_logits = model(['Example input'], domain_features)
