# CyberMoE: A Minimal Mixture-of-Experts Demonstration
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/guerilla7/CyberMoE) 

This project provides a minimal, educational implementation of a sparse **Mixture-of-Experts (MoE)** model. It is designed to demonstrate the core concepts of MoE within a cybersecurity context, showing how different "expert" models can be used to classify security-related text.

The script uses a shared `distilbert-base-uncased` encoder and a gating network to route inputs to a selection of specialized experts, demonstrating the principles of specialization, explainability, and efficiency.

## Core Concepts Demonstrated

This script is a hands-on demonstration of a modern MoE architecture:

1.  **Shared Encoder**: A single `distilbert-base-uncased` model from HuggingFace processes the input text into numerical representations.
2.  **Specialized Experts**: Three simple neural networks act as classifiers for three distinct (simulated) domains: "Network", "Malware", and "Phishing".
3.  **Gating Network**: A small network that analyzes the encoded text and assigns a relevance score to each expert.
4.  **Sparse Routing (Top-K)**: To demonstrate efficiency, the model only **activates the Top-K (K=2)** most relevant experts for any given input. The output of the non-selected expert is skipped entirely, saving computation.

## How It Works

The script's main function, `train_demo()`, performs two phases:

1.  **Synthetic Training**: First, it trains the MoE model on a "themed" synthetic dataset. It generates sentences with specific keywords (e.g., "IP address," "malware," "email") to teach the gating network how to route different types of inputs to the correct expert.
2.  **Inference**: After the brief training phase, it runs inference on a sample of five security-themed sentences. The output clearly shows the model's final prediction, the gating network's routing decisions, and which experts were sparsely activated.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/CyberMoE.git
    cd CyberMoE
    ```
2.  **Install dependencies:**
    This project uses a standard `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *(For a more detailed guide on setting up the environment, including specific CUDA/GPU drivers, see `SETUP_GUIDE.md`.)*

## How to Run

Simply run the `CyberMoe.py` script from your terminal:

```bash
python CyberMoe.py
```

By default, this executes the `train_demo()` function, which trains the model and shows the final predictions. This is the recommended way to see the model in action.

## Interpreting the Output

After running the script, you will see the training progress followed by the final inference results. Here is a sample output for one of the inputs and an explanation of each part.

```text
================================================================================
[3] Input: Normal user activity: accessing internal portal
  -> Predicted label: 0 (prob=1.000)
  Gating probs: Network=0.90, Malware=0.05, Phishing=0.05
  Top-2 experts: ['Network', 'Phishing'] (scores=[0.8960..., 0.0532...])
  Expert Logits (shows sparse activation):
    Expert 0 (Network) logits: [ 5.236638  -4.4157934] <-- ACTIVATED
    Expert 1 (Malware) logits: [0. 0.] <-- SKIPPED
    Expert 2 (Phishing) logits: [ 0.45459735 -0.10165542] <-- ACTIVATED
================================================================================
```

- **`Predicted label`**: The final classification. `0` is benign, `1` is malicious. Here, the model correctly identified the activity as benign with 100% probability.
- **`Gating probs`**: The initial scores assigned by the gating network to each expert. Here, it gave a 90% score to the "Network" expert.
- **`Top-2 experts`**: The experts that were selected for activation based on the gating scores.
- **`Expert Logits (shows sparse activation)`**: This is the most important part for understanding the MoE's efficiency.
    - `ACTIVATED`: The model computed the output for the "Network" and "Phishing" experts. You can see their raw output scores (logits).
    - `SKIPPED`: The "Malware" expert was deemed irrelevant by the gating network, so its logits are `[0. 0.]`, meaning it was **not computed**. This demonstrates the computational savings of sparse MoE models.

## Data Files Required for Testing

Some scripts (such as `consumer_morpheus_to_cybermoe.py` and `smoke_test.py`) require input and output data files in the `data/` directory:

- `data/cybermoe_input.jsonl`: Input events for smoke tests. Example content:
    ```jsonl
    {"id": 1, "event": "Suspicious login attempt from unknown IP address"}
    {"id": 2, "event": "New vulnerability discovered in Apache HTTP Server"}
    ```
- `data/morpheus_out.jsonl`: Output results for the consumer script. Example content:
    ```jsonl
    {"id": 1, "result": "benign"}
    {"id": 2, "result": "malicious"}
    ```

If these files are missing, you may encounter file-not-found errors. You can create them manually or use the provided samples above.
