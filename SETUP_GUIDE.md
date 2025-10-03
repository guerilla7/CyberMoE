Below is a **step‑by‑step recipe** that will let you run the CyberMoE demo on a *local NVIDIA workstation* without touching any cloud‑based LLM service.  
It covers the **minimum hardware** you need, the **OS/driver stack**, and the **Python / library install** that makes the script work out of the box.

---

## 1. Minimum Hardware Requirements

| Component | Minimum / Recommended | Why it matters |
|-----------|-----------------------|----------------|
| **GPU**   | NVIDIA RTX 3060 (or any GPU ≥ 4 GB VRAM, compute capability ≥ 7.0) | DistilBERT (≈400 MB weights + runtime tensors) comfortably fits in 4 GB; larger experts will need more |
| **CPU**   | Quad‑core (Intel i5 / AMD Ryzen 5) or better | Multi‑threaded tokenisation & inference; the script is CPU‑friendly if no GPU |
| **RAM**   | 16 GB DDR4 (≥ 8 GB is OK for small demos) | Keeps the OS + PyTorch cache in memory |
| **Storage** | 512 GB SSD (≥ 10 GB free) | Models (~400 MB) + cache + OS |
| **OS**    | Ubuntu 20.04/22.04 *or* Windows 10/11 (Professional or Enterprise) | The installation scripts below target Ubuntu; Windows is also supported. For Windows, using WSL2 or Anaconda is recommended for easier CUDA setup. |
| **Internet** | For the first run (downloads ~400 MB) | After that you can switch to offline mode |

> **Tip:** If your GPU has 8 GB or more you can experiment with larger expert models (e.g., `bert-base-uncased` or even GPT‑Neo). For the demo as written, 4 GB is plenty.

---

## 2. Software Stack

| Layer | Tool / Version | Why |
|-------|----------------|-----|
| **NVIDIA Driver** | ≥ 535.x (latest) | Enables CUDA and cuDNN |
| **CUDA Toolkit** | 11.8 (or 12.x if you install the matching PyTorch wheel) | Provides GPU runtime |
| **cuDNN** | 8.6+ (matches CUDA) | Accelerates convolution & transformer ops |
| **Python** | 3.10, 3.11, or 3.13 (recommended) | Latest stable Python releases |
| **PyTorch** | 2.1.x + CUDA (or CPU‑only) | Core deep‑learning lib |
| **Transformers** | 4.36.x (latest) | HuggingFace LLM & tokenizer |
| **tqdm** | 4.65+ | Progress bars (optional) |

> All versions are *current* at the time of writing (2025‑09). The exact wheel you install will match your CUDA version.

---

## 3. Install NVIDIA Driver, CUDA & cuDNN (Ubuntu)

> If you already have a working driver + CUDA combo that satisfies the above, skip this section.

```bash
# 1️⃣ Update & install prerequisites
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms

# 2️⃣ Install NVIDIA driver (replace <driver_version> with the latest e.g. 535)
sudo ubuntu-drivers autoinstall
# Verify:
nvidia-smi

# 3️⃣ Install CUDA Toolkit (11.8) – the .deb package is easiest
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2204-11-8-local/7fa2af80.pub
sudo apt update
sudo apt install -y cuda

# 4️⃣ Install cuDNN (8.6) – download from NVIDIA dev portal, then:
#    wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.6.0/cudnn-11.8-linux-x64-v8.6.0.163.tgz
tar -xzvf cudnn-11.8-linux-x64-v8.6.0.163.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

> **PATH & LD_LIBRARY_PATH**  
> Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

> Reload: `source ~/.bashrc`

---

## 4. Install Python & Virtual Environment

```bash
# Option A: Using conda (recommended)
conda create -n cybermoe python=3.10
conda activate cybermoe

# Option B: Using venv (pure pip)
python3 -m venv cybermoe
source cybermoe/bin/activate
```

---

## 5. Install PyTorch (CUDA)

```bash
# For CUDA 11.8 – use the official wheel
pip install --upgrade pip
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

> **Verify**:

```python
>>> import torch, sys
>>> print("CUDA available:", torch.cuda.is_available())
>>> print("CUDA version:", torch.version.cuda)
>>> print("cuDNN enabled:", torch.backends.cudnn.enabled)
```

If `torch.cuda.is_available()` prints `True`, you’re good.

---

## 6. Install Dependencies

```bash
pip install -r requirements.txt
```

> This pulls in all the necessary libraries, including `torch`, `transformers`, and `streamlit`.

--- 

## 7. (Optional) CPU‑Only Install

If you don’t have a compatible GPU, the dependencies are the same. The `torch` package will automatically use the CPU if a GPU is not available.

```bash
pip install -r requirements.txt
```

The script will automatically fall back to CPU (`DEVICE = "cpu"`).

---

## 8. Clone / Copy the Demo Script

```bash
git clone https://github.com/yourname/cybermoe-demo.git   # or copy the script file into a folder
cd cybermoe-demo
# (If you didn’t clone, just create CyberMoe.py and paste the code)
```

> The demo script already contains `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` so it will pick GPU automatically.

---

## 9. Run the Demo

There are two ways to run the demo:

### Option A: Interactive Web Demo (Recommended)

```bash
streamlit run app.py
```

This will start a web server and provide a local URL (usually `http://localhost:8501`) that you can open in your browser.

When the app is running, you can:

- Use the main panel to analyze a sentence and view Mixture-of-Experts routing and domain relevance.
- Provide RLHF feedback (correct/incorrect, optional correction and notes). Feedback is stored in `data/feedback.jsonl`.
- In the sidebar under “🧪 RLHF Utilities,” train a reward model from feedback and fine-tune CyberMoE from feedback. The fine-tuned checkpoint is saved to `checkpoints/cybermoe_finetuned.pt` and will be auto-loaded on subsequent app starts.
- Under “🗂️ Feedback Data,” export `feedback.jsonl` or import an existing one (Append or Replace). Replace requires an explicit confirmation checkbox. Uploads are validated as line-delimited JSON with minimal schema checks.

### Option B: Console Demo

```bash
python CyberMoe.py
```

You should see output similar to:

```
========================================
[0] Input: Suspicious login attempt from unknown IP address
  -> Predicted label: 1 (prob=0.87)
  Gating probs: [0.12 0.83 0.05]
  Top-2 experts: [1, 0] (scores=[0.83, 0.12])
  Expert 0 logits: [ -1.2   0.5 ]
  Expert 1 logits: [ 0.8 -1.4]
  ...
```

> **If you hit an OOM (out‑of‑memory) error** on a 4 GB GPU, try reducing the batch size or using `torch.cuda.empty_cache()` after inference.

---

## 10. (Optional) Docker‑based Reproducible Environment

If you prefer containerization, the following `Dockerfile` will spin up an identical environment in a few minutes.

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install system deps
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:

```bash
docker build -t cybermoe .
docker run --gpus all -p 8501:8501 cybermoe
```

---

## 11. Offline Mode (No Internet After First Run)

1. **Download the model once**  
   ```bash
    python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
   ```

2. **Set cache path** (so the script never tries to hit HF servers again)

   ```bash
   export TRANSFORMERS_CACHE=/path/to/local/cache
   ```

3. **Run the demo** – it will read from `TRANSFORMERS_CACHE` instead of downloading.

---

## 12. Quick Checklist

| ✔ | Item |
|---|------|
| ✅ | NVIDIA driver ≥ 535.x |
| ✅ | CUDA Toolkit 11.8 + cuDNN 8.6 |
| ✅ | Python 3.10+ in a virtual env |
| ✅ | PyTorch 2.1.x + CUDA (or CPU) |
| ✅ | Transformers 4.36+ |
| ✅ | Enough VRAM (≥ 4 GB for DistilBERT) |
| ✅ | Internet at least once for model download |

---

### Final Note

With the above setup, your local workstation will be able to run **CyberMoE** entirely offline, leveraging the GPU for inference. The Streamlit app includes RLHF feedback collection, a tiny reward model trainer, supervised fine-tuning, analytics, and import/export of feedback data. The fine-tuned model checkpoint persists across app restarts. The script is lightweight enough that it can even run on a modest CPU‑only machine, but it will be slower.

---

## Appendix: Git LFS for Large Artifacts

This repository uses Git Large File Storage (LFS) to track large artifacts (e.g., checkpoints and some datasets). If you plan to push or pull large files, set up LFS first:

```bash
git lfs install
git lfs pull     # to fetch LFS content for existing pointers
```

The `.gitattributes` includes patterns like `checkpoints/**`, `data/**`, `*.pt`, `*.csv`, and `*.jsonl`. New large files matching those patterns will be stored via LFS automatically. If a previously committed large file triggers GitHub’s 100MB limit, you may need to migrate it to LFS and force-push (coordinate with collaborators):

```bash
git lfs migrate import --include='checkpoints/*.pt'
git push --force-with-lease
```
