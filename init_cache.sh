#!/usr/bin/env bash
set -euo pipefail
echo "[init] creating data dir"
mkdir -p data
echo "[init] pulling demo HF backbone into cache"
BACKBONE=${CYBERMOE_BACKBONE:-distilbert-base-uncased}
python3 - <<PY
from transformers import AutoModel, AutoTokenizer
print("Downloading:", "'"$BACKBONE"'")
AutoTokenizer.from_pretrained("'"$BACKBONE"'", cache_dir="./.cache")
AutoModel.from_pretrained("'"$BACKBONE"'", cache_dir="./.cache")
print("Done")
PY
echo "[init] init complete. small sample dataset is placed under ./data"
