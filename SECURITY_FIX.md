# Security Fix — February 8, 2026

This commit applies four security hardening fixes to the CyberMoE demo application.
All changes have been tested and verified against existing smoke tests and model-loading tests.

---

## 1. Fix Insecure Deserialization in `torch.load` — **Critical**

**File:** `finetune_from_feedback.py` (line ~113)

**Problem:** `torch.load()` was called with `weights_only=False`, which uses Python's `pickle`
module internally. A tampered checkpoint file could execute arbitrary code on the host
when the model is loaded.

**Fix:** Changed to `weights_only=True`. The checkpoint only contains safe types (`dict`,
`OrderedDict`, `str`, `int`, `Tensor`) that are on PyTorch's default allowlist.
Removed the unnecessary `try/except TypeError` fallback for older PyTorch versions
since the project already targets PyTorch ≥ 2.0.

**Before:**
```python
try:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(ckpt_path, map_location=device)
```

**After:**
```python
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
```

---

## 2. Fix Shell Injection in `init_cache.sh` — **Medium**

**File:** `init_cache.sh` (lines 6–13)

**Problem:** The environment variable `$BACKBONE` was interpolated directly into a Python
heredoc via string concatenation (`"'"$BACKBONE"'"`). A crafted `CYBERMOE_BACKBONE`
value could break out of the string and inject arbitrary Python code.

**Fix:** Quoted the heredoc delimiter (`<<'PY'`) to prevent shell expansion inside the
heredoc block, and read the environment variable from within Python using
`os.environ.get()` instead.

**Before:**
```bash
BACKBONE=${CYBERMOE_BACKBONE:-distilbert-base-uncased}
python3 - <<PY
print("Downloading:", "'"$BACKBONE"'")
AutoTokenizer.from_pretrained("'"$BACKBONE"'", ...)
```

**After:**
```bash
export CYBERMOE_BACKBONE=${CYBERMOE_BACKBONE:-distilbert-base-uncased}
python3 - <<'PY'
import os
backbone = os.environ.get("CYBERMOE_BACKBONE", "distilbert-base-uncased")
AutoTokenizer.from_pretrained(backbone, ...)
```

---

## 3. Pin Dependencies in `requirements.txt` — **Medium**

**File:** `requirements.txt`

**Problem:** All seven dependencies were completely unpinned (e.g., just `torch`). This
made builds non-reproducible, and exposed the project to supply-chain attacks or
unexpected breaking changes when new package versions are released.

**Fix:** Pinned all packages to specific versions that match the Dockerfile's install
step. Added `pyarrow`, `torchvision`, and `torchaudio` to keep the file complete.

| Package | Pinned Version |
|---|---|
| torch | 2.0.1 |
| transformers | 4.36.2 |
| streamlit | 1.29.0 |
| scikit-learn | 1.3.2 |
| datasets | 2.14.6 |
| pandas | 2.0.3 |
| graphviz | 0.20.3 |
| pyarrow | 14.0.2 |
| torchvision | 0.15.2 |
| torchaudio | 2.0.2 |

---

## 4. Add Non-Root User to Dockerfile — **Low**

**File:** `Dockerfile`

**Problem:** The container had no `USER` directive, so the application ran as `root`
inside the container. If the app were compromised, the attacker would have root
privileges within the container.

**Fix:** Added a non-root `appuser` with `useradd` and switched to it with the `USER`
directive before the `EXPOSE` and `CMD` lines.

**Added:**
```dockerfile
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser
```

---

## Testing

All changes were verified with:

| Test | Result |
|---|---|
| `python smoke_test.py` | ✅ All tests passed |
| `python test_loading.py` | ✅ Model loaded successfully |
| `load_finetuned_model()` with `weights_only=True` | ✅ Checkpoint loads correctly |
| `py_compile` on all Python files | ✅ No syntax errors |
| `bash -n init_cache.sh` | ✅ Valid shell syntax |
