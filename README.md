# CyberMoE
A Mixture‑of‑Experts Framework for Adaptive Cybersecurity

> **TL;DR** – CyberMoE is a modular, scalable cybersecurity platform that treats every *expert* as an independently‑trained large language model (LLM) fine‑tuned for a specific threat domain. A lightweight *gating network* routes each incoming data item to the most relevant experts, while a *fusion LLM* aggregates their predictions into a single, explainable decision. The whole system is built for real‑time inference, continual learning, and seamless integration into existing SOC workflows.  This code is for educational purposes only, I created this to learn how MoE works and how it can benefit cybersecurity use-cases.

---

### 1. Executive Summary

- **Goal** – Leverage the expressive power of modern LLMs while keeping inference tractable and results explainable.
- **Core idea** – Use a *Mixture‑of‑Experts* (MoE) architecture where each expert is an LLM that has mastered a particular cybersecurity sub‑domain (e.g., network traffic, malware binaries, phishing emails, vulnerability databases).
- **Key benefits** –  
  * **Scalability:** Add or retire experts without retraining the whole model.  
  * **Efficiency:** Only a handful of experts are activated per request (sparse routing).  
  * **Explainability:** Gating probabilities + expert confidence scores reveal *why* a decision was made.  
  * **Adaptivity:** Experts can be fine‑tuned on fresh data, and a meta‑expert can learn to combine expert outputs over time.

---

### 2. Problem Space

Traditional security analytics pipelines are monolithic, often relying on handcrafted rules or flat machine‑learning models that struggle to keep pace with the volume and variety of cyber data. They also lack:

| Limitation | Example |
|------------|---------|
| **Monolithic training** | A single model must learn all threat vectors, forcing a trade‑off between breadth and depth. |
| **Computational bottleneck** | Full‑scale LLM inference on every log, packet, or email is infeasible. |
| **Opaque reasoning** | Black‑box outputs make SOC analysts hesitant to act on them. |

CyberMoE addresses each of these with a *distributed expert* architecture.

---

### 3. Core Concepts

| Concept | Definition |
|---------|------------|
| **Expert LLM** | A small transformer (≈ 200–500 M parameters) fine‑tuned on a domain‑specific corpus. |
| **Gating Network** | A lightweight (≈ 5–10 M parameters) neural network that takes an input embedding and outputs a probability distribution over experts. |
| **Sparse Activation** | Only the top‑k experts (typically k = 2–4) are invoked per input. |
| **Fusion LLM** | A 10‑M parameter transformer that ingests the concatenated expert outputs and produces a final verdict (e.g., threat score, playbook). |
| **Meta‑Expert** | An optional higher‑level LLM that learns to *re‑weight* expert contributions based on context (e.g., past SOC decisions). |

---

### 4. System Architecture

```
+---------------------------------------------+
|               CyberMoE Platform             |
|                                             |
|  +-----------------+   +---------------+    |
|  | Data Ingestion  |-->| Pre‑processor |    |
|  +-----------------+   +---------------+    |
|         |                        |          |
|         v                        v          |
|  +-----------------+   +---------------+    |
|  | Feature Encoder |<--| Embedding Net |    |
|  +-----------------+   +---------------+    |
|         |                        |          |
|         v                        v          |
|  +-----------------+   +---------------+    |
|  |   Gating Net    |-->| Expert Router |    |
|  +-----------------+   +---------------+    |
|         |                        |          |
|         v                        v          |
|  +-----------------+   +---------------+    |
|  |   Experts (N)   |<--| Sparse Invoc. |    |
|  +-----------------+   +---------------+    |
|         |                        |          |
|         v                        v          |
|  +-----------------+   +---------------+    |
|  | Fusion LLM      |<--| Aggregator    |    |
|  +-----------------+   +---------------+    |
|         |                        |          |
|         v                        v          |
|  +-----------------+   +---------------+    |
|  | Explainability  |<--| Explanations  |    |
|  +-----------------+   +---------------+    |
|         |                        |          |
|         v                        v          |
|  +-----------------+   +---------------+    |
|  | SOC / SIEM API  |<--| Output/Act.   |    |
|  +-----------------+   +---------------+    |
+---------------------------------------------+
```

#### 4.1 Data Ingestion

- **Sources** – SIEM logs, NetFlow / sFlow records, EDR telemetry, email headers, threat‑intel feeds (OTX, MISP), CVE databases, raw binaries.
- **Normalization** – All data is converted into a *canonical representation* (JSON) and passed to the encoder.

#### 4.2 Feature Encoder

- **Embedding** – A transformer‑based encoder (e.g., DistilBERT) that produces a 768‑dim vector per input.  
- **Optional multimodal** – For binaries, use a dual‑encoder (text + byte‑level) to capture both metadata and payload.

#### 4.3 Gating Network

- **Architecture** – Multi‑layer perceptron (MLP) with softmax output over *E* experts.  
- **Training** – Supervised: minimize cross‑entropy between predicted expert distribution and a *teacher* signal (e.g., domain label).  
- **Sparse Routing** – At inference, keep only the top‑k experts; set all others to zero.

#### 4.4 Expert Modules

| Domain | Example Data | Typical Tasks |
|--------|--------------|---------------|
| **Network Traffic** (NTE) | NetFlow, Zeek logs | Anomaly detection, protocol fingerprinting |
| **Endpoint Behavior** (EBE) | Process lists, file system events | Insider threat detection |
| **Malware Analysis** (MAE) | PE headers, malware samples | Static/dynamic feature extraction |
| **Phishing / Email** (TIE) | Message headers, URLs | Spam filtering, credential harvesting detection |
| **Vulnerability & Patch** (VPME) | CVE feeds, OS packages | Risk scoring, patch prioritization |
| **Incident Response** (IRDE) | SOC playbooks, ticketing | Automated response suggestions |

- **Expert LLM** – Each is a lightweight transformer fine‑tuned on millions of labeled examples from its domain.  
- **Knowledge Base** – Each expert has access to a *domain‑specific knowledge graph* (e.g., known CVE–CPE relationships) via a key‑value store.

#### 4.5 Fusion LLM

- **Input** – Concatenated expert outputs (softmax scores, embeddings).  
- **Task** – Multi‑label classification: threat severity, recommended action, confidence.  
- **Explainability** – Outputs the gating probabilities and each expert’s *contribution weight*.

#### 4.6 Explainability Module

- **Explainable Heatmaps** – Visualize which tokens in the input influenced each expert.  
- **Expert Confidence** – Log the top‑k experts and their certainty scores.  
- **Audit Trail** – Store gating decisions, expert outputs, fusion output in a tamper‑proof ledger (e.g., WORM storage).

#### 4.7 SOC / SIEM API

- **REST/GraphQL** – Provide endpoints for real‑time alerts, batch scans, and playbook generation.  
- **Webhook** – Push findings to ticketing systems (Jira, ServiceNow).  

---

### 5. Training & Fine‑Tuning

#### 5.1 Multi‑Task, Multi‑Expert Joint Training

- **Objective** – Optimize *both* the gating network and all experts jointly:
  \[
  L = \sum_{i=1}^{N} w_i \cdot L_{\text{expert}_i} + \lambda \cdot L_{\text{gate}}
  \]
  where \(w_i\) are the gating probabilities for input *i*.

- **Teacher‑Forcing** – Use a small “oracle” network that knows the true domain label to supervise gating initially, then let the gating learn from expert performance.

#### 5.2 Knowledge Distillation

- **Shared Backbone** – All experts share a small *shared encoder* that learns universal representations.  
- **Distillation Loss** – Encourage experts to mimic each other on *non‑domain* samples, improving generalization.

#### 5.3 Continual Learning Pipeline

- **Data Ingestion** – New logs, threat feeds are queued for nightly retraining.  
- **Fine‑Tuning** – Each expert is updated on its domain data only, preserving other experts’ knowledge.  
- **Catastrophic Forgetting Mitigation** – Elastic Weight Consolidation (EWC) on shared layers.

#### 5.4 Adversarial Robustness

- **Adversarial Training** – Generate perturbed inputs (e.g., obfuscated binaries) and train experts to remain stable.  
- **Poisoning Defense** – Monitor gating entropy; high variance may indicate a poisoned input.

---

### 6. Inference Flow (Real‑Time)

```
Input --> Preprocess --> Encoder --> Gating Net
   |
   |--> Top‑k experts selected (sparse routing)
        |
        v
  Expert outputs (scores, embeddings)
        |
        v
  Fusion LLM aggregates --> Final verdict + Playbook
        |
        v
  Explainability Module (gating probs, heatmaps)
        |
        v
  SOC API / SIEM alert
```

- **Latency** – < 200 ms per log entry on a single GPU; only the active experts consume compute.  
- **Throughput** – 10k+ logs per second on a modest GPU cluster, thanks to sparse routing.

---

### 7. Integration & Deployment

| Layer | Deployment Strategy |
|-------|---------------------|
| **Experts** | Docker containers, each running a *micro‑service* that exposes an inference endpoint. |
| **Gating + Fusion** | Kubernetes pod orchestrated by a lightweight controller that routes requests. |
| **Explainability & Logging** | Centralized logging (ELK stack) + immutable audit log. |
| **Security of the Model** | Deploy inside a Trusted Execution Environment (TEE) or secure enclave; encrypt model weights. |
| **Data Privacy** | Use *federated* training for cross‑org experts; only share gradients or embeddings, not raw logs. |

---

### 8. Evaluation Plan

| Metric | What it Measures |
|--------|------------------|
| **Detection Accuracy** (TPR/FPR) | Classic classification metrics per domain. |
| **Latency & Throughput** | Real‑time performance under load. |
| **Explainability Score** | Human‑subjective rating of gating clarity and heatmap usefulness. |
| **Adaptation Speed** | Time to incorporate a new threat vector (e.g., zero‑day). |
| **Robustness** | Accuracy under adversarial perturbations. |

- **Datasets** –  
  * CIC‑IDS2017, UNSW‑NB15 for NTE.  
  * VirusShare / Malpedia for MAE.  
  * PhishTank, SpamAssassin for TIE.  
  * CVE‑2024 feed for VPME.

- **Benchmarks** – Compare against monolithic LLM baseline, rule‑based IDS, and other MoE security solutions.

---

### 9. Novelty & Competitive Edge

| Feature | Why It’s New |
|---------|--------------|
| **Domain‑Specific Expert LLMs** | Unlike generic security models, each expert is a *fully fine‑tuned* LLM on its niche. |
| **Sparse MoE Routing** | Only a few experts are invoked per input, reducing compute while keeping accuracy. |
| **Meta‑Fusion LLM** | Learns to *re‑weight* expert outputs, adapting to changing threat landscapes. |
| **Explainable Gate** | Provides transparent routing decisions that SOC analysts can audit. |
| **Federated MoE** | Enables multi‑org collaboration without exposing raw logs. |

---

### 10. Future Extensions

1. **Graph‑Enhanced Experts** – Integrate graph neural networks (GNNs) to model relationships between indicators of compromise (IOCs).  
2. **Cross‑Modal Fusion** – Combine textual logs with network flow graphs, or binaries with OS metadata.  
3. **Policy‑Driven Routing** – Let SOC managers set *policy constraints* (e.g., prioritize EBE for high‑value endpoints).  
4. **Auto‑Scaling Expert Pools** – Dynamically spin up additional expert replicas during peak threat periods.  
5. **Open‑Source Release** – Provide a reference implementation and pre‑trained experts for community contribution.

---

### 11. High‑Level Implementation Roadmap

| Phase | Deliverables |
|-------|--------------|
| **0 – 3 mo** | Prototype encoder, gating network; pre‑train domain experts on public datasets. |
| **4 – 6 mo** | Integrate fusion LLM; build explainability UI; test on live SIEM logs. |
| **7 – 9 mo** | Deploy in a pilot SOC; collect feedback; refine gating thresholds. |
| **10 – 12 mo** | Release federated training pipeline; open‑source core components. |
| **Beyond 12 mo** | Add graph modules, multi‑modal experts; scale to enterprise deployments. |

---

## Conclusion

CyberMoE brings the power of large language models to cybersecurity in a *structured, modular*, and *efficient* way. By treating each threat domain as an expert LLM and routing inputs via a lightweight gating network, the framework delivers high‑accuracy detections with low latency while keeping explanations transparent. Its architecture is inherently extensible—new experts can be added as threats evolve, and the system can learn to re‑weight expert contributions over time. CyberMoE is thus positioned as a next‑generation, AI‑driven security platform ready for the fast‑moving cyber threat landscape.