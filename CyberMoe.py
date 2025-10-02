#!/usr/bin/env python3
"""
CyberMoE – Minimal Mixture‑of‑Experts for Adaptive Cybersecurity

Author:  Ron F. Del Rosario
Date:    2025‑09‑11
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from model import CyberMoE, train_model, TOP_K, DEVICE

# --------------------------------------------------------------------------- #
# 5️⃣ Demo – inference & explanation
# --------------------------------------------------------------------------- #
def demo(model):
    model.eval()  # inference mode

    # Expert names for clearer output
    expert_names = ["Network", "Malware", "Phishing"]

    # Example inputs (feel free to add your own)
    texts = [
        "Suspicious login attempt from unknown IP address",
        "New vulnerability discovered in Apache HTTP Server",
        "Phishing email with malicious attachment detected",
        "Normal user activity: accessing internal portal",
        "Malware sample shows code injection in system DLL"
    ]

    with torch.no_grad():
        final_logits, gating_probs, expert_logits = model(texts)

    # Convert to probabilities
    probs = torch.softmax(final_logits, dim=-1).cpu().numpy()

    # Print results
    for i, text in enumerate(texts):
        print("\n" + "=" * 80)
        print(f"[{i}] Input: {text}")

        # Final prediction
        pred = int(probs[i].argmax())
        print(f"  -> Predicted label: {pred} (prob={probs[i][pred]:.3f})")

        # Gating probabilities
        gate = gating_probs[i].cpu().numpy()
        print(f"  Gating probs: " + ", ".join([f"{name}={prob:.2f}" for name, prob in zip(expert_names, gate)]))

        # Top‑k experts (for explanation)
        topk_vals, topk_idx = torch.topk(gating_probs[i], TOP_K)
        topk_names = [expert_names[idx] for idx in topk_idx]
        print(f"  Top-{TOP_K} experts: {topk_names} (scores={topk_vals.numpy().tolist()})")

        # Expert logits
        for j, exp in enumerate(expert_logits[i].cpu().numpy()):
            print(f"    Expert {j} ({expert_names[j]}) logits: {exp}")
        print("\n--- Explanation of Output ---")
        print("Predicted label: 0 = benign, 1 = malicious")
        print("prob: Model's confidence in the predicted label (higher = more confident)")
        print("Gating probs: Probability assigned to each expert by the gating network (higher = more trusted)")
        print("Top-K experts: Indices of the experts most trusted for this input")
        print("Expert logits: Raw scores from each expert before fusion; higher value for index 1 means more likely malicious")


# --------------------------------------------------------------------------- #
# 6️⃣ Optional training loop (synthetic data)
# --------------------------------------------------------------------------- #
def train_demo():
    """
    Very small training demo on synthetic data.
    In practice you would use real labeled logs, malware samples,
    phishing emails, etc.  Each domain expert should be fine‑tuned
    on its own data first.
    """
    
    model = train_model()

    # After training, run the demo inference again
    print("\n--- Post‑training inference ---")
    
    model.eval()
    expert_names = ["Network", "Malware", "Phishing"]
    texts = [
        "Suspicious login attempt from unknown IP address",
        "New vulnerability discovered in Apache HTTP Server",
        "Phishing email with malicious attachment detected",
        "Normal user activity: accessing internal portal",
        "Malware sample shows code injection in system DLL"
    ]
    
    with torch.no_grad():
        final_logits, gating_probs, expert_logits = model(texts)

    probs = torch.softmax(final_logits, dim=-1).cpu().numpy()

    print("Demo input predictions after training:")
    for i, text in enumerate(texts):
        print("\n" + "=" * 80)
        print(f"[{i}] Input: {text}")
        pred = int(probs[i].argmax())
        print(f"  -> Predicted label: {pred} (prob={probs[i][pred]:.3f})")
        gate = gating_probs[i].cpu().numpy()
        print(f"  Gating probs: " + ", ".join([f"{name}={prob:.2f}" for name, prob in zip(expert_names, gate)]))
        topk_vals, topk_idx = torch.topk(gating_probs[i], TOP_K)
        topk_names = [expert_names[idx] for idx in topk_idx]
        print(f"  Top-{TOP_K} experts: {topk_names} (scores={topk_vals.numpy().tolist()})")

        # Also print the sparse expert logits to show which were used
        print("  Expert Logits (shows sparse activation):")
        for j, exp_logit in enumerate(expert_logits[i].cpu().numpy()):
            if exp_logit.any(): # Check if the expert was used
                print(f"    Expert {j} ({expert_names[j]}) logits: {exp_logit} <-- ACTIVATED")
            else:
                print(f"    Expert {j} ({expert_names[j]}) logits: {exp_logit} <-- SKIPPED")

    # Illustrative confusion matrix for the demo inputs
    true_demo_labels = [1, 1, 1, 0, 1] # Malicious, Malicious, Phishing, Benign, Malware
    final_preds = [int(p.argmax()) for p in probs]
    demo_acc = accuracy_score(true_demo_labels, final_preds)
    demo_cm = confusion_matrix(true_demo_labels, final_preds)
    print("\n" + "=" * 80)
    print(f"Accuracy on demo sentences: {demo_acc:.4f}")
    print(f"Confusion Matrix on demo sentences:\n{demo_cm}")


# --------------------------------------------------------------------------- #
# 7️⃣ Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # --- Choose one of the two options below ---

    # 1. Run a quick inference demo with an UNTRAINED model.
    #    Shows the architecture but predictions will be random.
    # model = CyberMoE().to(DEVICE)
    # demo(model)

    # 2. Run the synthetic TRAINING and inference demo.
    #    This trains the model on themed data, so its predictions are more realistic.
    #    This is the recommended way to see the model in action.
    train_demo()
