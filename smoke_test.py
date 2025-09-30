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
    print("Missing input queue: ./data/cybermoe_input.jsonl â€” run consumer first", file=sys.stderr)
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
