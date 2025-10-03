#!/usr/bin/env python3
"""
Expert pre-training utilities for CyberMoE.
Handles domain-specific data loading and pre-training logic.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import json

class DomainDataset(Dataset):
    """Dataset for domain-specific expert pre-training"""
    def __init__(self, data_path: str, domain: str, preprocessor=None):
        """
        :param data_path: Path to the data directory containing domain files
        :param domain: Domain name (network, malware, phishing, cloud, webapp)
        :param preprocessor: Optional CyberPreprocessor instance
        """
        self.domain = domain
        self.preprocessor = preprocessor
        self.data = self._load_domain_data(data_path, domain)
        
    def _load_domain_data(self, data_path: str, domain: str) -> pd.DataFrame:
        """Load domain-specific training data"""
        path = Path(data_path) / f"{domain}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Domain data not found: {path}")
            
        data = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    'text': item['text'],
                    'label': item['label']  # 0=benign, 1=malicious
                })
        return pd.DataFrame(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data.iloc[idx]
            text = item['text']
            label = item['label']
            
            # Apply preprocessing if available
            if self.preprocessor:
                try:
                    features = self.preprocessor.process(text)
                    return {
                        'text': text,
                        'features': features,
                        'label': torch.tensor(label, dtype=torch.long)
                    }
                except Exception as e:
                    print(f"Warning: Preprocessing failed for item {idx}: {str(e)}")
                    # Fallback to basic text
                    return {
                        'text': text,
                        'label': torch.tensor(label, dtype=torch.long)
                    }
            else:
                return {
                    'text': text,
                    'label': torch.tensor(label, dtype=torch.long)
                }
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            # Return a simple example in case of error
            return {
                'text': "Error loading text",
                'label': torch.tensor(0, dtype=torch.long)
            }

def custom_collate_fn(batch):
    """Custom collate function to handle variable-length data"""
    # Separate different items in the batch
    text_list = []
    label_list = []
    features_list = []
    
    for item in batch:
        text_list.append(item['text'])
        label_list.append(item['label'])
        if 'features' in item:
            features_list.append(item['features'])
    
    # Stack labels
    labels = torch.stack(label_list)
    
    # Create the batch dictionary
    batch_dict = {
        'text': text_list,
        'label': labels
    }
    
    # Handle features if they exist
    if features_list:
        # Combine and pad feature tensors if needed
        batch_dict['features'] = {}
        
        # Handle the case where features is a dict
        if isinstance(features_list[0], dict):
            for key in features_list[0].keys():
                if torch.is_tensor(features_list[0][key]):
                    try:
                        batch_dict['features'][key] = torch.stack([f[key] for f in features_list])
                    except:
                        # For variable length tensors, use padding
                        max_len = max(f[key].size(0) for f in features_list)
                        padded = []
                        for f in features_list:
                            if f[key].size(0) < max_len:
                                padding = torch.zeros(max_len - f[key].size(0), *f[key].size()[1:])
                                padded.append(torch.cat([f[key], padding]))
                            else:
                                padded.append(f[key])
                        batch_dict['features'][key] = torch.stack(padded)
                else:
                    batch_dict['features'][key] = [f[key] for f in features_list]
    
    return batch_dict

def create_expert_dataloaders(
    data_path: str,
    domains: List[str],
    preprocessor,
    batch_size: int = 32,
    num_workers: int = 0  # Changed to 0 for stability
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for each expert's domain-specific data
    
    :param data_path: Path to data directory
    :param domains: List of domain names
    :param preprocessor: CyberPreprocessor instance
    :param batch_size: Batch size for training
    :param num_workers: Number of worker processes (default 0 for main process)
    :return: Dict mapping domain names to DataLoaders
    """
    dataloaders = {}
    for domain in domains:
        dataset = DomainDataset(data_path, domain, preprocessor)
        dataloaders[domain] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False,
            collate_fn=custom_collate_fn  # Use our custom collate function
        )
    return dataloaders

def pretrain_expert(
    expert: nn.Module,
    embeddings: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    progress_callback: Optional[callable] = None
):
    """
    Pre-train a single expert on domain-specific data
    
    :param expert: Expert module to train
    :param embeddings: CyberSecurityEmbeddings module
    :param dataloader: DataLoader with domain data
    :param device: Device to train on
    :param num_epochs: Number of pre-training epochs
    :param learning_rate: Learning rate for pre-training
    :param progress_callback: Optional callback for progress updates
    """
    optimizer = torch.optim.AdamW([
        {'params': expert.parameters(), 'lr': learning_rate},
        {'params': embeddings.parameters(), 'lr': learning_rate * 0.1}
    ])
    criterion = torch.nn.CrossEntropyLoss()
    
    expert.train()
    embeddings.train()
    total_steps = len(dataloader) * num_epochs
    current_step = 0
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Update progress
            if progress_callback:
                progress = current_step / total_steps
                progress_callback(progress)
            current_step += 1
            
            # Process batch inputs
            text_batch = batch['text']
            labels = batch['label'].to(device)
            
            try:
                # Try using preprocessed features if available
                if 'features' in batch:
                    features = batch['features']
                    # Check if we have all required tensors
                    if all(k in features for k in ['entity_types', 'domain_scores']):
                        encoded = embeddings.tokenizer(
                            text_batch,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt"
                        )
                        hidden_states = embeddings(
                            input_ids=encoded['input_ids'].to(device),
                            attention_mask=encoded['attention_mask'].to(device),
                            entity_types=features['entity_types'].to(device),
                            domain_weights=features['domain_scores'].to(device)
                        )
                    else:
                        raise KeyError("Missing required feature tensors")
                else:
                    raise KeyError("No features in batch")
            except Exception as e:
                # Fallback to basic text processing
                encoded = embeddings.tokenizer(
                    text_batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                batch_size = encoded['input_ids'].size(0)
                seq_len = encoded['input_ids'].size(1)
                
                # Create properly shaped tensors
                entity_types = torch.zeros_like(encoded['input_ids'])
                domain_weights = torch.ones(batch_size, len(embeddings.domain_adaptation)) / len(embeddings.domain_adaptation)
                
                hidden_states = embeddings(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    entity_types=entity_types,
                    domain_weights=domain_weights
                )
                
            # Forward pass through expert
            logits = expert(hidden_states)
            loss = criterion(logits, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    expert.eval()