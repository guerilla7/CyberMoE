#!/usr/bin/env bash
set -euo pipefail
echo "[init] creating data dir"
mkdir -p data
echo "[init] pulling demo HF backbone into cache"
export CYBERMOE_BACKBONE=${CYBERMOE_BACKBONE:-distilbert-base-uncased}
python3 - <<'PY'
import os
from transformers import AutoModel, AutoTokenizer
backbone = os.environ.get("CYBERMOE_BACKBONE", "distilbert-base-uncased")
print("Downloading:", backbone)
AutoTokenizer.from_pretrained(backbone, cache_dir="./.cache")
AutoModel.from_pretrained(backbone, cache_dir="./.cache")
print("Done")
PY
echo "[init] init complete. small sample dataset is placed under ./data"
