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
