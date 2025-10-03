#!/usr/bin/env python3
"""
CyberPreprocessor - Technical feature extraction for CyberMoE

Author:  Ron F. Del Rosario
Date:    2025-09-12
"""

import re
import torch
from typing import List, Dict, Tuple
from collections import Counter

class CyberPreprocessor:
    """Technical feature extraction and preprocessing for cybersecurity text"""
    
    def __init__(self):
        # Domain list
        self.domains = ['network', 'malware', 'phishing', 'cloud', 'webapp']
        
        # Regular expressions for technical entities
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+|\S+\.[a-z]{2,}(?:/\S*)?')
        self.port_pattern = re.compile(r'(?:port\s+)?(?:\d{1,5})/(?:tcp|udp)|:(\d{1,5})')
        self.hash_pattern = re.compile(r'\b[a-fA-F0-9]{32,64}\b')
        self.cve_pattern = re.compile(r'CVE-\d{4}-\d{4,7}', re.IGNORECASE)
        self.domain_pattern = re.compile(r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b')
        
        # Domain-specific keywords and weights
        self.domain_keywords = {
            'network': {
                'network': 1.0, 'tcp': 0.8, 'udp': 0.8, 'port': 0.6, 'traffic': 0.7,
                'packet': 0.7, 'router': 0.8, 'firewall': 0.8, 'dns': 0.7, 'ip': 0.6
            },
            'malware': {
                'malware': 1.0, 'virus': 0.8, 'trojan': 0.9, 'ransomware': 0.9,
                'payload': 0.7, 'execution': 0.6, 'process': 0.6, 'binary': 0.7
            },
            'phishing': {
                'phishing': 1.0, 'email': 0.8, 'credential': 0.8, 'login': 0.7,
                'password': 0.7, 'account': 0.6, 'social': 0.6, 'spoof': 0.8
            },
            'cloud': {
                'cloud': 1.0, 'aws': 0.9, 'azure': 0.9, 'gcp': 0.9, 's3': 0.8,
                'bucket': 0.7, 'instance': 0.7, 'container': 0.7, 'kubernetes': 0.8
            },
            'webapp': {
                'web': 0.8, 'application': 0.6, 'sql': 0.8, 'injection': 0.8,
                'xss': 0.9, 'csrf': 0.9, 'cookie': 0.7, 'session': 0.7, 'api': 0.7
            }
        }

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization"""
        return text.lower().split()
    
    def _extract_ips(self, text: str) -> List[str]:
        """Extract IP addresses"""
        return self.ip_pattern.findall(text)
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs and domains"""
        return self.url_pattern.findall(text)
    
    def _extract_ports(self, text: str) -> List[str]:
        """Extract port numbers"""
        return self.port_pattern.findall(text)
    
    def _extract_hashes(self, text: str) -> List[str]:
        """Extract cryptographic hashes"""
        return self.hash_pattern.findall(text)
    
    def _extract_cves(self, text: str) -> List[str]:
        """Extract CVE IDs"""
        return self.cve_pattern.findall(text)
    
    def _extract_domains(self, text: str) -> List[str]:
        """Extract domain names"""
        return self.domain_pattern.findall(text)
    
    def _score_domain_relevance(self, text: str, tokens: List[str], domain: str) -> float:
        """
        Score text relevance to a specific domain using keywords and technical features
        Returns a float between 0 and 1
        """
        # First pass: collect all scores without normalization
        keyword_score = 0.0
        feature_score = 0.0
        
        # Score based on keywords (max 1.0)
        keywords = self.domain_keywords.get(domain, {})
        keyword_matches = sum(1 for token in tokens if token in keywords)
        if keyword_matches > 0:
            keyword_weights = [keywords[token] for token in tokens if token in keywords]
            keyword_score = sum(keyword_weights) / (len(keyword_weights) * 1.0)
        
        # Score based on technical features (max 1.0)
        if domain == 'network':
            num_features = len(self._extract_ips(text)) + len(self._extract_ports(text))
            if num_features > 0:
                feature_score = min(1.0, num_features * 0.25)  # Cap at 1.0
        elif domain == 'malware':
            num_features = len(self._extract_hashes(text))
            if num_features > 0:
                feature_score = min(1.0, num_features * 0.5)
        elif domain == 'phishing':
            num_features = len(self._extract_urls(text)) + len(self._extract_domains(text))
            if num_features > 0:
                feature_score = min(1.0, num_features * 0.25)
        elif domain == 'cloud':
            num_features = len(self._extract_urls(text))
            if num_features > 0:
                feature_score = min(1.0, num_features * 0.5)
        elif domain == 'webapp':
            num_features = len(self._extract_urls(text)) + len(self._extract_ports(text))
            if num_features > 0:
                feature_score = min(1.0, num_features * 0.25)
        
        # Final score is average of keyword and feature scores, each max 1.0
        return (keyword_score + feature_score) / 2.0
    
    def get_entity_types(self, text: str, token_ids: List[int]) -> torch.Tensor:
        """
        Map each token to its entity type (0=PAD, 1=IP, 2=URL, 3=PORT, 4=HASH, 5=CVE, 6=DOMAIN)
        Returns tensor of same length as token_ids
        """
        # Initialize with PAD (0)
        entity_types = [0] * len(token_ids)
        
        # Create span mapping
        text_ptr = 0
        spans = []
        tokens = self.tokenize(text)
        
        for token in tokens:
            start = text[text_ptr:].find(token)
            if start == -1:
                continue
            start += text_ptr
            end = start + len(token)
            spans.append((start, end))
            text_ptr = end
            
        # Extract entities with positions
        entities = {
            1: [(m.start(), m.end()) for m in self.ip_pattern.finditer(text)],
            2: [(m.start(), m.end()) for m in self.url_pattern.finditer(text)],
            3: [(m.start(), m.end()) for m in self.port_pattern.finditer(text)],
            4: [(m.start(), m.end()) for m in self.hash_pattern.finditer(text)],
            5: [(m.start(), m.end()) for m in self.cve_pattern.finditer(text)],
            6: [(m.start(), m.end()) for m in self.domain_pattern.finditer(text)]
        }
        
        # Map entities to tokens
        for entity_type, positions in entities.items():
            for start, end in positions:
                for i, (tok_start, tok_end) in enumerate(spans):
                    # If token overlaps with entity
                    if not (tok_end <= start or tok_start >= end):
                        entity_types[i] = entity_type
        
        return torch.tensor(entity_types)
    
    def process(self, text: str, token_ids: List[int] = None) -> dict:
        """Process input text and extract features"""
        if token_ids is None:
            token_ids = self.tokenize(text)
            
        # Get entity types for each token
        entity_types = self.get_entity_types(text, token_ids)
        
        # Extract technical features
        features = {
            'ips': self._extract_ips(text),
            'urls': self._extract_urls(text),
            'ports': self._extract_ports(text),
            'hashes': self._extract_hashes(text),
            'cves': self._extract_cves(text),
            'domains': self._extract_domains(text)
        }
        
        # Score domain relevance - shape: [num_domains]
        # Keep this 1D per sample; batching happens later in the model
        domain_scores = torch.tensor([
            self._score_domain_relevance(text, token_ids, domain)
            for domain in self.domains
        ], dtype=torch.float32)
        
        return {
            'tokens': token_ids,
            'entity_types': entity_types,
            'features': features,
            'domain_scores': domain_scores
        }
