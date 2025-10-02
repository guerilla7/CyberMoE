# apply_patch.ps1
# Purpose: Create Morpheus demo files for CyberMoE locally (no git required).
# Edit $OutDir to the target folder where files should be created.
# If Git is installed and you want to commit/push, set $DoGit = $true and fill $GitRepoUrl.
# Run in PowerShell (Windows). For bash scripts created here, run them under WSL/Git-Bash.

param(
    [string]$OutDir = ".",
    [string]$Username = "mpathak4",
    [bool]$DoGit = $false,
    [string]$GitRepoUrl = ""
)

if ($Username -eq "mpathak4") {
    Write-Host "Note: edit the script to set `\$Username` if you want to set Git remote names or messages." -ForegroundColor Yellow
}

# Ensure output dir exists
$absOut = Resolve-Path -Path $OutDir
New-Item -ItemType Directory -Path $absOut -Force | Out-Null

Set-Location $absOut

# ----------------------------
# morpheus_pipeline.yaml
# ----------------------------
$morpheus = @'
pipeline:
  name: morpheus_prefilter_demo
  description: GPU streaming prefilter that parses pcap, extracts features, and emits embeddings + suspicion_score
  stages:
    - id: pcap_reader
      type: pcap_reader
      params:
        path: ./data/sample.pcap
        replay_rate: 1000
    - id: flow_enricher
      type: zeek_enricher
      params:
        enrich_fields:
          - src_ip
          - dst_ip
          - src_port
          - dst_port
          - proto
          - payload_len
    - id: feature_extractor
      type: feature_extractor
      params:
        flow_features:
          - byte_histogram
          - packet_sizes
          - interarrival_stats
        text_features:
          - http_host
          - user_agent
    - id: streaming_encoder
      type: streaming_encoder
      params:
        model: demo-encoder-128
        precision: fp16
        batch_size: 64
    - id: suspicion_scorer
      type: anomaly_scorer
      params:
        model: gpu-anom-64
        threshold: 0.5
        output_fields:
          - suspicion_score
    - id: output_formatter
      type: json_formatter
      params:
        schema:
          - event_id
          - ts
          - source_type: "net"
          - src_ip
          - dst_ip
          - proto
          - embedding: embedding_array
          - suspicion_score
    - id: file_sink
      type: file_sink
      params:
        path: ./data/morpheus_out.jsonl
        max_batch_size: 1000
'@
Set-Content -Path "morpheus_pipeline.yaml" -Value $morpheus -Encoding UTF8
Write-Host "Wrote morpheus_pipeline.yaml"

# ----------------------------
# consumer_morpheus_to_cybermoe.py
# ----------------------------
$consumer = @'
#!/usr/bin/env python3
"""
Minimal consumer that converts Morpheus JSONL output into the canonical CyberMoE ingestion envelope.
Reads ./data/morpheus_out.jsonl and writes ./data/cybermoe_input.jsonl
"""
import json
import uuid
import time
from typing import Dict, Any

INPUT_SOURCE = "./data/morpheus_out.jsonl"
OUT_QUEUE = "./data/cybermoe_input.jsonl"

def morpheus_to_envelope(m_msg: Dict[str, Any]) -> Dict[str, Any]:
    event_id = m_msg.get("event_id") or str(uuid.uuid4())
    ts = m_msg.get("ts") or int(time.time() * 1000)
    src = m_msg.get("source_type", "net")
    embedding = m_msg.get("embedding")
    suspicion = float(m_msg.get("suspicion_score", 0.0))
    meta = {
        "src_ip": m_msg.get("src_ip"),
        "dst_ip": m_msg.get("dst_ip"),
        "proto": m_msg.get("proto"),
        "morpheus_score": suspicion,
        "raw_source": "morpheus_prefilter_demo",
    }
    raw = f"net|{meta.get('src_ip','-')}->{meta.get('dst_ip','-')}|proto={meta.get('proto','-')}|score={suspicion:.3f}"
    envelope = {
        "event_id": event_id,
        "ts": ts,
        "source_type": src,
        "raw": raw,
        "embedding": embedding,
        "meta": meta
    }
    return envelope

def consume_and_publish(input_path: str = INPUT_SOURCE, out_path: str = OUT_QUEUE):
    with open(input_path, "r") as fin, open(out_path, "a") as fout:
        for line in fin:
            try:
                m = json.loads(line)
            except Exception:
                continue
            env = morpheus_to_envelope(m)
            fout.write(json.dumps(env) + "\\n")

if __name__ == "__main__":
    consume_and_publish()
'@
Set-Content -Path "consumer_morpheus_to_cybermoe.py" -Value $consumer -Encoding UTF8
Write-Host "Wrote consumer_morpheus_to_cybermoe.py"

# ----------------------------
# docker-compose.demo.yml
# ----------------------------
$docker = @'
version: "3.8"
services:
  cybermoe:
    image: python:3.10-slim
    container_name: cybermoe
    build: .
    volumes:
      - ./:/app
      - ./data:/app/data
      - ${TRANSFORMERS_CACHE:-~/.cache/huggingface}/transformers:/root/.cache/huggingface/transformers
    working_dir: /app
    environment:
      - CYBERMOE_BACKBONE=${CYBERMOE_BACKBONE:-distilbert-base-uncased}
      - CYBERMOE_TOP_K=${CYBERMOE_TOP_K:-2}
    command: ["bash", "-lc", "python3 consumer_morpheus_to_cybermoe.py && python3 CyberMoe.py --demo"]
    deploy:
      resources:
        limits:
          memory: 8G
    profiles: ["cpu","gpu"]

  streamlit-app:
    image: python:3.10-slim
    container_name: cybermoe-streamlit
    build: .
    volumes:
      - ./:/app
      - ./data:/app/data
      - ${TRANSFORMERS_CACHE:-~/.cache/huggingface}/transformers:/root/.cache/huggingface/transformers
    working_dir: /app
    ports:
      - "8501:8501"
    command: ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    deploy:
      resources:
        limits:
          memory: 8G
    profiles: ["streamlit"]

  morpheus-prefilter:
    image: nvidia/morpheus:demo
    container_name: morpheus_prefilter
    volumes:
      - ./data:/data
      - ./morpheus_pipeline.yaml:/app/morpheus_pipeline.yaml
    command: ["bash", "-lc", "morpheus-run /app/morpheus_pipeline.yaml"]
    profiles: ["gpu"]
'@
Set-Content -Path "docker-compose.demo.yml" -Value $docker -Encoding UTF8
Write-Host "Wrote docker-compose.demo.yml"

# ----------------------------
# init_cache.sh
# ----------------------------
$init = @'
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
'@
Set-Content -Path "init_cache.sh" -Value $init -Encoding UTF8
# Mark executable bit if running in a unix-like environment (best-effort)
try {
    icacls "init_cache.sh" /grant Everyone:RX | Out-Null
} catch {
    # ignore on systems without icacls or no permissions
}
Write-Host "Wrote init_cache.sh"

# ----------------------------
# sample morpheus_out.jsonl
# ----------------------------
$sample = @'
{"event_id":"demo-1","ts":1700000000000,"source_type":"net","src_ip":"10.0.0.1","dst_ip":"10.0.0.2","proto":"TCP","embedding":[0.01,0.02,0.03,0.04,0.05],"suspicion_score":0.12}
{"event_id":"demo-2","ts":1700000001000,"source_type":"net","src_ip":"10.0.0.3","dst_ip":"10.0.0.4","proto":"HTTP","embedding":[0.11,0.12,0.13,0.14,0.15],"suspicion_score":0.78}
'@
if (-not (Test-Path -Path "data")) { New-Item -ItemType Directory -Path "data" | Out-Null }
Set-Content -Path "data/morpheus_out.jsonl" -Value $sample -Encoding UTF8
Write-Host "Wrote data/morpheus_out.jsonl"

# ----------------------------
# smoke_test.py
# ----------------------------
$smoke = @'
#!/usr/bin/env python3
"""
Smoke test that validates the minimal realtime loop:
reads ./data/cybermoe_input.jsonl produced by consumer and checks presence of expected fields.
Exits non-zero on failure.
"""
import json
import sys
from pathlib import Path

INPUT = Path("./data/cybermoe_input.jsonl")
if not INPUT.exists():
    print("Missing input queue: ./data/cybermoe_input.jsonl — run consumer first", file=sys.stderr)
    sys.exit(2)

required = {"event_id","ts","source_type","raw","embedding","meta"}
count = 0
with INPUT.open() as f:
    for line in f:
        try:
            o = json.loads(line)
        except Exception as e:
            print("Invalid JSON:", e, file=sys.stderr)
            sys.exit(3)
        if not required.issubset(set(o.keys())):
            print("Missing required keys in envelope:", set(o.keys()), file=sys.stderr)
            sys.exit(4)
        count += 1
if count == 0:
    print("No events found in queue", file=sys.stderr)
    sys.exit(5)
print(f"Smoke test passed: {count} events found")
sys.exit(0)
'@
Set-Content -Path "smoke_test.py" -Value $smoke -Encoding UTF8
Write-Host "Wrote smoke_test.py"

# ----------------------------
# Optional: Git commit & push if requested and git available
# ----------------------------
if ($DoGit) {
    $git = & git --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Git not found in PATH. Skipping git commit/push. Install git or push files via GitHub web UI." -ForegroundColor Yellow
    } else {
        # initialize repo if not already
        if (-not (Test-Path -Path ".git")) {
            git init
            git add .
            git commit -m "Add Morpheus prefilter demo files and smoke test"
            if ($GitRepoUrl) {
                git remote add origin $GitRepoUrl
                git branch -M feat/morpheus-prefilter-demo
                git push -u origin feat/morpheus-prefilter-demo
            } else {
                Write-Host "No GitRepoUrl provided; local commit created. Provide GitRepoUrl to push." -ForegroundColor Yellow
            }
        } else {
            git checkout -b feat/morpheus-prefilter-demo
            git add .
            git commit -m "Add Morpheus prefilter demo files and smoke test"
            if ($GitRepoUrl) {
                git remote add origin $GitRepoUrl 2>$null
                git push -u origin feat/morpheus-prefilter-demo
            } else {
                Write-Host "Repository is a git repo but no GitRepoUrl supplied; commit created locally." -ForegroundColor Yellow
            }
        }
    }
} else {
    Write-Host "Files created locally in `$(Resolve-Path .)`."
    Write-Host "If you want me to enable Git commit/push, re-run with -DoGit $true and -GitRepoUrl 'https://github.com/your/repo.git' (requires git in PATH)."
}

Write-Host "Done."