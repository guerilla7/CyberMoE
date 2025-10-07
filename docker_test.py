import os
import torch
from finetune_from_feedback import finetune_from_feedback, load_finetuned_model

# Create test data
os.makedirs("data", exist_ok=True)
with open("data/test_feedback.jsonl", "w") as f:
    f.write("""{"user_input": "This is a test malicious input", "user_feedback": true, "pred_label": "Malicious"}""")

# Test fine-tuning
print("Testing fine-tuning...")
model = finetune_from_feedback(feedback_path="data/test_feedback.jsonl")
print("Fine-tuning successful!")

# Test loading
print("Testing checkpoint loading...")
loaded_model = load_finetuned_model()
print(f"Model loaded successfully: {loaded_model is not None}")

print("All tests passed!")

