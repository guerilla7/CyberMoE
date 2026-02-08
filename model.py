#!/usr/bin/env python3
"""
CyberMoE - Minimal Mixture-of-Experts for Adaptive Cybersecurity

Author:  Ron F. Del Rosario
Date:    2025-09-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Constants
NUM_EXPERTS = 5
EXPERT_LABELS = 2  # Binary classification (benign/malicious)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GatingNetwork(nn.Module):
    """
    Hierarchical attention-based gating network for expert routing.
    """
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Token-level attention
        self.token_query = nn.Linear(hidden_size, hidden_size)
        self.token_key = nn.Linear(hidden_size, hidden_size)
        self.token_value = nn.Linear(hidden_size, hidden_size)
        
        # Expert routing layers
        self.expert_attention = nn.Linear(hidden_size, num_experts)
        self.expert_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_experts)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
        :return: Expert routing probabilities of shape (batch_size, num_experts)
        """
        # Token-level self-attention
        Q = self.token_query(hidden_states)
        K = self.token_key(hidden_states)
        V = self.token_value(hidden_states)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32)
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        token_context = torch.matmul(attention_probs, V)
        
        # Sequence pooling
        sequence_repr = token_context.mean(dim=1)
        
        # Expert routing scores
        expert_attention = self.expert_attention(sequence_repr)
        expert_logits = self.expert_gate(sequence_repr)
        
        # Combine attention and gate scores
        routing_logits = expert_attention + expert_logits
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        return routing_probs

class Expert(nn.Module):
    """Individual expert network"""
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure we have a batch dimension and properly average sequence dimension
        if len(hidden_states.shape) == 2:
            # Add batch dimension if missing
            hidden_states = hidden_states.unsqueeze(0)
        
        # Mean pooling over sequence length dimension
        pooled = hidden_states.mean(dim=1)
        
        # Get logits, ensuring shape is [batch_size, num_labels]
        logits = self.classifier(pooled)
        
        return logits

class CyberSecurityEmbeddings(nn.Module):
    """
    Enhanced embeddings layer with cybersecurity-specific features.
    Combines BERT embeddings with technical entity embeddings and domain knowledge.
    """
    def __init__(self, base_model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_encoder = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.base_encoder.config.hidden_size
        
        # Technical entity type embeddings
        self.entity_embeddings = nn.Embedding(7, self.hidden_size)  # [PAD, IP, URL, PORT, HASH, CVE, DOMAIN]
        
        # Domain-specific token adaptation
        self.domain_adaptation = nn.ModuleDict({
            'network': nn.Linear(self.hidden_size, self.hidden_size),
            'malware': nn.Linear(self.hidden_size, self.hidden_size),
            'phishing': nn.Linear(self.hidden_size, self.hidden_size),
            'cloud': nn.Linear(self.hidden_size, self.hidden_size),
            'webapp': nn.Linear(self.hidden_size, self.hidden_size)
        })
        
        # Output fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                entity_types: torch.Tensor, domain_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enhanced embeddings
        :param input_ids: Token IDs from BERT tokenizer
        :param attention_mask: Attention mask for padding
        :param entity_types: Technical entity type IDs for each token
        :param domain_weights: Domain relevance scores (batch_size, num_domains)
        """
        # Get base BERT embeddings
        base_outputs = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        base_embeddings = base_outputs.last_hidden_state
        
        # Add technical entity embeddings
        entity_emb = self.entity_embeddings(entity_types)
        
        # Apply domain-specific adaptations
        domain_adapted = torch.zeros_like(base_embeddings)
        batch_size, seq_len, hidden_size = base_embeddings.shape
        
        for i, (domain, adapter) in enumerate(self.domain_adaptation.items()):
            # Handle domain weights safely - they might have unexpected shape
            if i < domain_weights.size(1):
                # Process each batch item separately to avoid broadcasting issues
                adapted_batch = []
                # Process each item in the batch
                for b in range(batch_size):
                    # Get a single sample's embeddings
                    sample_embeddings = base_embeddings[b:b+1]  # Keep batch dim as [1, seq, hidden]
                    # Get adapted features for this sample
                    sample_adapted = adapter(sample_embeddings)
                    # Get weight for this domain and sample
                    sample_weight = domain_weights[b, i].item()
                    # Apply weight
                    adapted_batch.append(sample_adapted * sample_weight)
                
                # Stack batch back together and add to accumulated adaptations
                if adapted_batch:
                    stacked_adaptations = torch.cat(adapted_batch, dim=0)
                    domain_adapted += stacked_adaptations
        
        # Combine embeddings
        combined = torch.cat([base_embeddings, domain_adapted], dim=-1)
        enhanced_embeddings = self.fusion(combined)
        
        return enhanced_embeddings

class CyberMoE(nn.Module):
    """
    Enhanced Mixture-of-Experts cybersecurity model with domain awareness.
    Uses custom embeddings and hierarchical attention for better expert routing.
    """
    def __init__(self, num_experts: int = NUM_EXPERTS,
                 expert_labels: int = EXPERT_LABELS,
                 top_k: int = 2,
                 pretrain_experts: bool = True,
                 pretrain_epochs: int = 5,
                 pretrain_lr: float = 1e-4,
                 pretrain_batch_size: int = 32):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Pre-training configuration
        self.pretrain_config = {
            'enabled': pretrain_experts,
            'epochs': pretrain_epochs,
            'learning_rate': pretrain_lr,
            'batch_size': pretrain_batch_size
        }
        
        # Initialize preprocessor for technical feature extraction
        from preprocessor import CyberPreprocessor
        self.preprocessor = CyberPreprocessor()
        
        # Enhanced embeddings layer
        self.embeddings = CyberSecurityEmbeddings()
        hidden_size = self.embeddings.hidden_size
        
        # Gating network for expert routing
        self.gate = GatingNetwork(hidden_size, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(hidden_size, expert_labels)
            for _ in range(num_experts)
        ])
        
        # Domain classifier
        self.domain_classifier = nn.Linear(hidden_size, num_experts)
        
        self.device = DEVICE
        self.to(self.device)
    
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with enhanced embeddings
        :param texts: Batch of input texts
        :return: Dict with expert outputs and routing probabilities
        """
        # Process inputs through preprocessor
        batch_features = [self.preprocessor.process(text) for text in texts]
        
        # Prepare inputs
        encoded = self.embeddings.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Stack preprocessed features
        # Align entity_types per-sample to tokenizer sequence length via pad/truncate
        seq_len = input_ids.size(1)
        entity_types_list = []
        for f in batch_features:
            et = f['entity_types']
            if not isinstance(et, torch.Tensor):
                et = torch.tensor(et, dtype=torch.long)
            et = et.long().view(-1)
            L = et.size(0)
            if L >= seq_len:
                et_adj = et[:seq_len]
            else:
                et_adj = torch.zeros(seq_len, dtype=torch.long)
                et_adj[:L] = et
            entity_types_list.append(et_adj)
        entity_types = torch.stack(entity_types_list, dim=0).to(self.device)
        
        # Handle domain scores with more careful shape handling
        # Each item should be [num_domains], stacked to [batch, num_domains]
        domain_scores_list = []
        for f in batch_features:
            ds = f['domain_scores']
            # Ensure each score is 1D
            if ds.dim() > 1:
                ds = ds.view(-1)  # Flatten to 1D
            domain_scores_list.append(ds)
            
        # Stack into batch
        domain_scores = torch.stack(domain_scores_list).to(self.device)
        # Ensure shape is [batch_size, num_domains]
        if domain_scores.dim() > 2:
            domain_scores = domain_scores.view(len(texts), -1)  # Reshape to [batch, domains]
        
        # Get enhanced embeddings
        hidden_states = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_types=entity_types,
            domain_weights=domain_scores
        )
        
        # Get routing probabilities
        routing_probs = self.gate(hidden_states)
        
        # Get top-k experts
        top_k_scores, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Get predictions from top-k experts
        expert_outputs = []
        for i in range(routing_probs.size(0)):  # For each item in batch
            batch_outputs = []
            for j in range(NUM_EXPERTS):
                if j in top_k_indices[i]:
                    expert_idx = j
                    # Expert returns [1, EXPERT_LABELS]; squeeze batch dim -> [EXPERT_LABELS]
                    expert_output = self.experts[expert_idx](hidden_states[i:i+1]).squeeze(0)
                else:
                    # Zero logits for skipped experts -> [EXPERT_LABELS]
                    expert_output = torch.zeros(
                        EXPERT_LABELS,
                        device=self.device
                    )
                batch_outputs.append(expert_output)  # [EXPERT_LABELS]
            # Stack experts for this sample -> [NUM_EXPERTS, EXPERT_LABELS]
            expert_outputs.append(torch.stack(batch_outputs))
        # Stack batch -> [B, NUM_EXPERTS, EXPERT_LABELS]
        expert_outputs = torch.stack(expert_outputs)
        
        # Weight and combine expert outputs
        routing_probs_expanded = routing_probs.unsqueeze(-1)  # [B, NUM_EXPERTS, 1]
        weighted_outputs = expert_outputs * routing_probs_expanded  # [B, NUM_EXPERTS, EXPERT_LABELS]
        # Sum over experts -> [B, EXPERT_LABELS]
        final_output = weighted_outputs.sum(dim=1)
        
        # Domain prediction
        domain_logits = self.domain_classifier(hidden_states.mean(dim=1))
        
        return {
            'logits': final_output,
            'routing_probs': routing_probs,
            'expert_outputs': expert_outputs,
            'domain_logits': domain_logits
        }
    
    def explain_gating(self, text: str, expert_idx: int) -> List[Tuple[str, float]]:
        """
        Explain which parts of the input influenced the gating decision
        :param text: Input text to explain
        :param expert_idx: Index of expert to explain (0-4)
        :return: List of (token, importance_score) tuples
        """
        self.eval()
        with torch.no_grad():
            # Tokenize input
            tokens = self.embeddings.tokenizer.tokenize(text)
            encoded = self.embeddings.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Get preprocessor features
            features = self.preprocessor.process(text)
            # Align entity_types to tokenizer sequence length
            seq_len = input_ids.size(1)
            et = features['entity_types']
            if not isinstance(et, torch.Tensor):
                et = torch.tensor(et, dtype=torch.long)
            et = et.long().view(-1)
            L = et.size(0)
            if L >= seq_len:
                et_adj = et[:seq_len]
            else:
                et_adj = torch.zeros(seq_len, dtype=torch.long)
                et_adj[:L] = et
            entity_types = et_adj.unsqueeze(0).to(self.device)
            domain_scores = features['domain_scores'].unsqueeze(0).to(self.device)
            
            # Get embeddings
            hidden_states = self.embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_types=entity_types,
                domain_weights=domain_scores
            )
            
            # Get attention scores from gating network
            Q = self.gate.token_query(hidden_states)
            K = self.gate.token_key(hidden_states)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1))[0]
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            # Get importance scores for the specified expert
            expert_attention = self.gate.expert_attention(hidden_states)[0]
            token_importances = expert_attention[:, expert_idx].cpu().numpy()
            
            # Combine token texts with their importance scores
            token_scores = list(zip(tokens, token_importances))
            
            return token_scores

def train_model(progress_callback=None, weighted_loss=True, aux_loss_weight=1e-2, top_k=2,
             pretrain_experts=True, pretrain_epochs=5, pretrain_lr=1e-4):
    """Train a new CyberMoE model with optional expert pre-training"""
    
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
                # Convert string or numeric labels to integers
                label = item['label']
                if isinstance(label, str):
                    label = label_map.get(label.lower(), 0)  # Handle case insensitively
                elif isinstance(label, bool):
                    label = 1 if label else 0
                labels.append(int(label))  # Ensure integer type
                
                # Convert domain to one-hot
                domain = item.get('domain', 'Network')
                domain_onehot = [0] * len(domain_list)
                if domain in domain_map:
                    domain_onehot[domain_map[domain]] = 1
                domains.append(domain_onehot)
        
        # Convert to tensors and move to device
        label_tensor = torch.tensor(labels, dtype=torch.long).to(DEVICE)
        domain_tensor = torch.tensor(domains, dtype=torch.float).to(DEVICE)
        return texts, label_tensor, domain_tensor

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )

    # Initialize model with pre-training configuration
    model = CyberMoE(
        top_k=top_k,
        pretrain_experts=pretrain_experts,
        pretrain_epochs=pretrain_epochs,
        pretrain_lr=pretrain_lr
    ).to(DEVICE)
    
    # Pre-train experts if enabled
    if pretrain_experts and progress_callback:
        from pretraining import create_expert_dataloaders, pretrain_expert
        
        # Load domain-specific data
        expert_domains = ['network', 'malware', 'phishing', 'cloud', 'webapp']
        dataloaders = create_expert_dataloaders(
            'data/pretraining',
            expert_domains,
            model.preprocessor,
            batch_size=32
        )
        
        # Pre-train each expert
        pre_progress_start = 0.0
        pre_progress_per_expert = 0.2  # Reserve 20% for pre-training
        
        for i, (domain, loader) in enumerate(dataloaders.items()):
            expert = model.experts[i]
            
            def expert_progress(p):
                overall_progress = pre_progress_start + (p * pre_progress_per_expert)
                progress_callback(overall_progress)
            
            pretrain_expert(
                expert,
                model.embeddings,  # Pass the embeddings layer
                loader,
                DEVICE,
                num_epochs=pretrain_epochs,
                learning_rate=pretrain_lr,
                progress_callback=expert_progress
            )
            pre_progress_start += pre_progress_per_expert
    
    # Model optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Weighted loss
    if weighted_loss:
        # Roughly 5 malicious examples to 1 benign example
        class_weights = torch.tensor([0.6, 3.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for i, (texts, labels, domains) in enumerate(loader):
            # Ensure labels are tensor and on the right device
            if not isinstance(labels, torch.Tensor):
                # Convert to tensor if somehow it's not
                if isinstance(labels, list):
                    labels = torch.tensor(labels, dtype=torch.long)
                elif isinstance(labels, str):
                    # Convert string label
                    label_map = {'benign': 0, 'malicious': 1}
                    label_value = label_map.get(labels.lower(), 0)
                    labels = torch.tensor([label_value], dtype=torch.long)
                # Move to device
                labels = labels.to(DEVICE)
                
            # Tensors are already on device from collate_fn
            optimizer.zero_grad()
            outputs = model(texts)
            
            # Print shapes for debugging
            logits_shape = outputs['logits'].shape
            labels_shape = labels.shape
            
            # Classification loss - ensure shapes are compatible
            # Cross entropy expects class indices, not one-hot encoded labels
            if len(labels.shape) == 1 or (len(labels.shape) == 2 and labels.shape[1] == 1):
                # Labels are already indices, use them directly
                loss = criterion(outputs['logits'], labels)
            else:
                # Handle potential shape mismatch
                if labels.shape != outputs['logits'].shape:
                    # If labels are one-hot encoded or different shape, get class indices
                    if len(labels.shape) > 1 and labels.shape[1] > 1:
                        labels = labels.argmax(dim=1)  # Convert one-hot to indices
                    # Check if we have a batch dimension mismatch
                    if len(labels.shape) == 1 and outputs['logits'].shape[0] > 1:
                        # Expand labels to match batch size
                        labels = labels.expand(outputs['logits'].shape[0])
                    elif len(labels.shape) == 1 and outputs['logits'].shape[0] == 1:
                        # If single item batch, reshape to match
                        labels = labels.view(1)
                loss = criterion(outputs['logits'], labels)
            
            # Domain prediction loss: ensure domain targets are LongTensor class indices on DEVICE
            if isinstance(domains, torch.Tensor):
                if domains.dim() == 2:
                    # One-hot or probabilities -> to indices
                    domain_targets = domains.argmax(dim=1).to(DEVICE)
                elif domains.dim() == 1:
                    # Already class indices
                    domain_targets = domains.to(DEVICE).long()
                else:
                    # Unexpected shape; flatten and take indices safely
                    domain_targets = domains.view(domains.size(0), -1).argmax(dim=1).to(DEVICE)
            elif isinstance(domains, list):
                # Could be list of one-hot lists or strings
                if len(domains) > 0 and isinstance(domains[0], (list, tuple)):
                    domain_targets = torch.tensor([int(max(range(len(d)), key=lambda k: d[k])) for d in domains], dtype=torch.long, device=DEVICE)
                else:
                    domain_map = {'Network': 0, 'Malware': 1, 'Phishing': 2, 'Cloud': 3, 'Web App': 4}
                    domain_targets = torch.tensor([domain_map.get(str(d), 0) for d in domains], dtype=torch.long, device=DEVICE)
            elif isinstance(domains, str):
                domain_map = {'Network': 0, 'Malware': 1, 'Phishing': 2, 'Cloud': 3, 'Web App': 4}
                idx = domain_map.get(domains, 0)
                # Match batch size of logits (usually 1 here)
                batch_sz = outputs['domain_logits'].shape[0]
                domain_targets = torch.tensor([idx] * batch_sz, dtype=torch.long, device=DEVICE)
            else:
                # Fallback to zeros
                batch_sz = outputs['domain_logits'].shape[0]
                domain_targets = torch.zeros(batch_sz, dtype=torch.long, device=DEVICE)

            domain_loss = F.cross_entropy(outputs['domain_logits'], domain_targets)
            loss += domain_loss * 0.1
            
            # Auxiliary load balancing loss
            if aux_loss_weight > 0:
                # Encourage uniform expert utilization
                routing_probs = outputs['routing_probs']
                expert_usage = routing_probs.mean(0)
                target_usage = torch.ones_like(expert_usage) / NUM_EXPERTS
                aux_loss = F.kl_div(
                    expert_usage.log(),
                    target_usage,
                    reduction='batchmean'
                )
                loss += aux_loss_weight * aux_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if progress_callback:
                progress = (epoch + (i + 1) / len(loader)) / 5
                progress = 0.2 + (progress * 0.8)  # Scale to 20%-100% (after pre-training)
                progress_callback(progress)
    
    return model
