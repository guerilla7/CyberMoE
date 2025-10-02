#!/usr/bin/env python3
"""
CyberMoE – Interactive Streamlit Demo

Author:  Ron F. Del Rosario
Date:    2025‑09‑12
"""

import streamlit as st
import torch
import pandas as pd
from model import train_model, CyberMoE, TOP_K

# --------------------------------------------------------------------------- #
# Model Loading (with caching)
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_model():
    """Trains the model and caches it for future runs."""
    progress_bar = st.progress(0, text="Training model, please wait...")
    def update_progress(fraction):
        progress_bar.progress(fraction)
    
    model = train_model(progress_callback=update_progress)
    model.eval() # Set to inference mode
    progress_bar.empty()
    return model

# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="CyberMoE Demo", layout="wide")

st.title("🤖 CyberMoE: An Interactive Mixture-of-Experts Demo")
st.write("""
This demo showcases a sparse Mixture-of-Experts (MoE) model for cybersecurity text classification. 
Enter a security-related sentence below and see how the model analyzes it in real-time.
""")

# Load the model
model = load_model()
expert_names = ["Network", "Malware", "Phishing"]

# --- Input Area ---
st.header("Analyze a Security Event")

user_input = st.text_area(
    "Enter a sentence to classify:", 
    "Suspicious login attempt from unknown IP address",
    height=100
)

if st.button("Analyze", use_container_width=True):
    if user_input:
        with torch.no_grad():
            final_logits, gating_probs, expert_logits = model([user_input])
        st.session_state.analysis_results = (final_logits, gating_probs, expert_logits)
    else:
        st.warning("Please enter some text to analyze.")
        if 'analysis_results' in st.session_state:
            del st.session_state.analysis_results

if 'analysis_results' in st.session_state:
    final_logits, gating_probs, expert_logits = st.session_state.analysis_results
    # --- Results Display ---
    st.header("Analysis Results")

    col1, col2 = st.columns(2)

    # --- Column 1: Final Prediction & Gating Scores ---
    with col1:
        st.subheader("Final Prediction")
        probs = torch.softmax(final_logits, dim=-1).cpu().numpy().flatten()
        pred_label = probs.argmax()
        pred_text = "Malicious" if pred_label == 1 else "Benign"
        
        st.metric(label="Classification", value=pred_text, delta=f"{probs[pred_label]:.2%} confidence")
        
        st.subheader("Gating Network Scores")
        st.write("The gating network decides which expert is most relevant. Higher is better.")
        gating_df = pd.DataFrame({
            "Expert": expert_names,
            "Score": gating_probs.cpu().numpy().flatten()
        })
        st.bar_chart(gating_df, x="Expert", y="Score")

    # --- Column 2: Expert Activation ---
    with col2:
        st.subheader(f"Sparse Activation (Top-{TOP_K} Experts)")
        st.write("Only the most relevant experts are used to save computation.")

        topk_vals, topk_idx = torch.topk(gating_probs[0], TOP_K)
        topk_names = [expert_names[idx] for idx in topk_idx]

        activated_experts_df = pd.DataFrame({
            'Expert': topk_names,
            'Gating Score': topk_vals.cpu().numpy()
        })
        st.dataframe(activated_experts_df, use_container_width=True)

        st.subheader("Expert Outputs (Logits)")
        st.write("Raw scores from each expert. Skipped experts have `[0, 0]`.")
        
        expert_output_data = []
        for i, (name, logit_tensor) in enumerate(zip(expert_names, expert_logits[0])):
            logits = logit_tensor.cpu().numpy()
            status = "🔥 Activated" if logits.any() else "❄️ Skipped"
            expert_output_data.append({
                "Expert": name,
                "Logits [Benign, Malicious]": str(logits),
                "Status": status
            })
        
        expert_output_df = pd.DataFrame(expert_output_data)
        st.table(expert_output_df)

# --- Explanation Section ---
st.sidebar.title("💡 How It Works")
st.sidebar.info("""
1.  **Shared Encoder**: The input text is converted into a numerical representation by a single, shared `distilbert` model.

2.  **Gating Network**: A small neural network analyzes this representation and assigns a relevance score to each specialized expert (Network, Malware, Phishing).

3.  **Sparse Routing**: To be efficient, the model only activates the **Top-2** experts with the highest scores. The other experts are skipped entirely.

4.  **Final Decision**: The outputs of the activated experts are combined, weighted by their scores, to produce the final classification (Benign or Malicious).
""")

st.sidebar.title("⚙️ Controls")
if st.sidebar.button("Restart Demo (retrain model)"):
    st.cache_resource.clear()
    if 'analysis_results' in st.session_state:
        del st.session_state.analysis_results
    st.rerun()