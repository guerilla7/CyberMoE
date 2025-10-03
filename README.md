# CyberMoE: A Minimal Mixture-of-Experts Demonstration
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/guerilla7/CyberMoE)

<!-- Repo status badges -->
![GitHub Repo stars](https://img.shields.io/github/stars/guerilla7/CyberMoE?style=social)
![GitHub forks](https://img.shields.io/github/forks/guerilla7/CyberMoE?style=social)
![GitHub issues](https://img.shields.io/github/issues/guerilla7/CyberMoE)
![GitHub License](https://img.shields.io/github/license/guerilla7/CyberMoE)
![GitHub last commit](https://img.shields.io/github/last-commit/guerilla7/CyberMoE)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

This project provides a minimal, educational implementation of a sparse **Mixture-of-Experts (MoE)** model. It is designed to demonstrate the core concepts of MoE within a cybersecurity context, showcasing how domain-specific routing can improve model efficiency and interpretability.

This repository contains two main components:
1.  A command-line script (`CyberMoe.py`) that trains the model and runs inference on a predefined set of examples.
2.  An interactive web demo (`app.py`) built with Streamlit that allows you to classify your own sentences and visualize the MoE routing in real-time.

<img width="1265" height="1064" alt="Screenshot 2025-10-02 at 11 13 37 PM" src="https://github.com/user-attachments/assets/926491be-7e6b-4610-89f4-fcf9efcc30bf" />
<img width="1267" height="1225" alt="Screenshot 2025-10-02 at 11 16 13 PM" src="https://github.com/user-attachments/assets/47cd5b93-cbf2-4f36-ab7d-2a6b09e44fc4" />
<img width="1267" height="1083" alt="Screenshot 2025-10-02 at 11 17 11 PM" src="https://github.com/user-attachments/assets/65666f4e-f49e-424d-ad35-a83a908c6d19" />
<img width="1267" height="1245" alt="Screenshot 2025-10-02 at 11 18 05 PM" src="https://github.com/user-attachments/assets/99d7c305-4d06-491d-8276-34958bfaf994" />
<img width="1263" height="1317" alt="Screenshot 2025-10-02 at 11 19 46 PM" src="https://github.com/user-attachments/assets/02e10235-bad3-47b9-98c7-c919defaf5bd" />
<img width="257" height="435" alt="Screenshot 2025-10-02 at 11 22 07 PM" src="https://github.com/user-attachments/assets/c0196004-700f-4cef-b3fe-4bc5087b42c6" />
<img width="257" height="382" alt="Screenshot 2025-10-02 at 11 22 59 PM" src="https://github.com/user-attachments/assets/59a74041-e37f-488d-8c65-3f305753f5a1" />
<img width="257" height="580" alt="Screenshot 2025-10-02 at 11 23 47 PM" src="https://github.com/user-attachments/assets/564c2fcd-2684-407c-93b7-5bc8aed33ecf" />

## Project Structure

The project is organized into the following key files:

-   `CyberMoe.py`: The original command-line application for training and demonstration.
-   `app.py`: The new interactive Streamlit web demo.
-   `model.py`: Contains the core PyTorch model definitions (`CyberMoE`, `Expert`, `GatingNetwork`) and the training logic, making the model reusable.
-   `preprocessor.py`: Extracts cybersecurity entities and domain relevance scores used by the model.
-   `feedback.py`: Lightweight JSONL logger for RLHF feedback.
-   `rlhf_reward_model.py`: Trains a tiny reward model from feedback signals.
-   `finetune_from_feedback.py`: Fine-tunes `CyberMoE` on collected feedback and persists a checkpoint.
-   `requirements.txt`: A list of all Python dependencies for the project.
-   `SETUP_GUIDE.md`: A detailed guide for setting up the environment, especially for GPU usage.

## Core Concepts Demonstrated

This script is a hands-on demonstration of a modern MoE architecture:

1.  **Shared Encoder**: A single `distilbert-base-uncased` model from HuggingFace processes the input text into numerical representations.
2.  **Specialized Experts**: Five simple neural networks act as classifiers for five distinct (simulated) domains: "Network", "Malware", "Phishing", "Cloud Security", and "Web App Security".
3.  **Gating Network**: A small network that analyzes the encoded text and assigns a relevance score to each expert.
4.  **Sparse Routing (Top-K)**: To demonstrate efficiency, the model only **activates the Top-K (K=2)** most relevant experts for any given input. The output of the non-selected expert is skipped for computational savings.

## How It Works

The script's main function, `train_demo()`, performs two phases:

1.  **Synthetic Training**: First, it trains the MoE model on a "themed" synthetic dataset. It generates sentences with specific keywords (e.g., "IP address," "malware," "email") to teach the gating network to route inputs to the correct experts.
2.  **Inference**: After the brief training phase, it runs inference on a sample of five security-themed sentences. The output clearly shows the model's final prediction, the gating network's routing, and which experts were activated.

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

This project comes with two ways to run the model:

### 1. Interactive Web Demo (Recommended)

For the best experience, run the interactive Streamlit demo. This will launch a web page that allows you to input your own text and see the model's analysis in real-time. After entering a sentence, the app will show the expert routing, confidence, and predictions.

```bash
streamlit run app.py
```

If the `streamlit` command is not found, you can use the following command:

```bash
python -m streamlit run app.py
```

### 2. Original Console Demo

You can still run the original script from your terminal. This will train the model and print the final predictions to the console.

```bash
python CyberMoe.py
```

## Interpreting the Output

### Web Demo Output

The interactive demo provides a visual breakdown of the model's analysis:

-   **Final Prediction:** Shows the final classification ("Benign" or "Malicious") and the model's confidence.
-   **Gating Network Scores:** A bar chart that visualizes how the gating network rates the relevance of each expert for your input.
-   **Sparse Activation (Top-2 Experts):** Shows which two experts were selected and activated for the prediction.
-   **Expert Outputs (Logits):** Displays the raw output from each expert, making it clear which ones were skipped (their output will be `[0. 0.]`).

### Console Output

After running the `CyberMoe.py` script, you will see the training progress followed by the final inference results. Here is a sample output for one of the inputs and an explanation of each part.

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
    - `SKIPPED`: The "Malware" expert was deemed irrelevant by the gating network, so its logits are `[0. 0.]`, meaning it was **not computed**. This demonstrates the computational savings of sparse routing.

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

## Built-in RLHF Loop and Feedback Analytics

The Streamlit app supports a simple RLHF loop:

- Collect feedback per inference (correct/incorrect, optional correction, notes). Feedback is written to `data/feedback.jsonl`.
- Train a tiny reward model on the collected signals via the sidebar button.
- Fine-tune CyberMoE from the feedback data. The fine-tuned checkpoint is saved to `checkpoints/cybermoe_finetuned.pt` and auto-loaded on app restart.

Feedback analytics are available in the main view:

- KPIs (total, correct/incorrect, observed accuracy)
- Distributions (predicted labels, corrections, predicted domains)
- Confidence analysis by outcome
- Average gating signal and domain relevance scores
- Accuracy over time and recent samples table

You can also export/import `feedback.jsonl` via the sidebar:

- Download the current file for backup or transfer.
- Upload a JSONL file and choose Append or Replace. Replace requires a confirmation checkbox, and uploads are validated with a minimal schema before being written.

## Large Files and Git LFS

This repo uses Git LFS to store large artifacts like model checkpoints and some datasets. The patterns are configured in `.gitattributes` (e.g., `checkpoints/**`, `data/**`, `*.pt`, `*.csv`, `*.jsonl`).

Quick start for collaborators:

1. Install Git LFS
    - macOS: `brew install git-lfs`
    - Windows (Chocolatey): `choco install git-lfs`
    - Debian/Ubuntu: `sudo apt-get install git-lfs`
2. Initialize in your environment: `git lfs install`
3. Clone or update the repo:
    - Fresh clone: `git clone <repo>` (LFS files will download automatically)
    - Existing clone: `git lfs pull` (fetch binary content for LFS pointers)
4. Verify LFS tracking: `git lfs ls-files`

Pushing large files:

- Add files that match tracked patterns as usual (e.g., files under `checkpoints/`), then `git add` / `git commit` / `git push`. LFS will store the content outside normal Git history.
- If you encounter GitHub’s 100MB limit for a file that was previously committed without LFS, you can migrate history (use with care):
   - `git lfs migrate import --include='checkpoints/*.pt'`
   - `git push --force-with-lease`
   Coordinate with your team before rewriting history.

Tips:

- In CI or constrained environments, you can skip automatic LFS downloads by setting `GIT_LFS_SKIP_SMUDGE=1` and later fetch with `git lfs pull` when needed.
- Ensure new large file types are added to `.gitattributes` if they fall outside existing patterns.

## Troubleshooting

- Streamlit shows old results or analytics
    - Click “Restart Demo (retrain model)” in the sidebar to clear cached model and rerun training.
    - Feedback analytics are cached with the file modification time; if you manually edit `data/feedback.jsonl`, click the “Rerun” button (top-right) in Streamlit.
    - If a fine-tuned model is active, use “Clear Fine-Tuned” in the sidebar to revert to the base cached model.

- Fine-tuned model not loading after restart
    - Confirm `checkpoints/cybermoe_finetuned.pt` exists locally and is readable.
    - If you just pulled from Git, ensure LFS pulled the binary instead of a pointer file: `git lfs pull`.

- Feedback upload fails schema check
    - The uploader expects line-delimited JSON (JSONL). Each line should be a JSON object with at least `user_input` (string) and `user_feedback` (boolean). Optional fields like `pred_label`/`correction` are supported.
    - Replace mode is destructive and requires the confirmation checkbox.

- Graphviz diagram not rendering
    - The app uses `st.graphviz_chart`. The Python `graphviz` package is included, but some systems also need the Graphviz system binary.
        - macOS: `brew install graphviz`
        - Ubuntu/Debian: `sudo apt-get install graphviz`

- Push to GitHub fails with >100MB file
    - See “Large Files and Git LFS” above. Track the file with LFS or migrate history, then push.

- GPU/CUDA not detected
    - Make sure your PyTorch wheel matches your CUDA version and drivers. See SETUP_GUIDE for CUDA install and verification (`torch.cuda.is_available()`).

- Streamlit port already in use
    - Run on a different port: `streamlit run app.py --server.port=8502`

## Further Reading & Research

Here are some pioneering and relevant research papers on Mixture-of-Experts (MoE) neural network architectures and their applications, including in cybersecurity:

- **Shazeer et al. (2017) – “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”**  
  [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) – Introduces the scalable sparsely-gated MoE layer for neural networks.

- **Fedus et al. (2022) – “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity”**  
  [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) – Shows how simple MoE routing enables massive scale and efficiency in NLP models.

- **Gururangan et al. (2022) – “Mixture-of-Experts for Knowledge-Intensive NLP Tasks”**  
  [arXiv:2202.08906](https://arxiv.org/abs/2202.08906) – Explores MoE’s effectiveness in tasks requiring specialized domain knowledge.
