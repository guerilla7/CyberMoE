#!/usr/bin/env python3
"""
CyberMoE – Interactive Streamlit Demo

Author:  Ron F. Del Rosario
Date:    2025‑09‑12
"""

import streamlit as st
import torch
import pandas as pd
from model import train_model, CyberMoE, NUM_EXPERTS

# --------------------------------------------------------------------------- #
# Model Loading (with caching)
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_model(weighted_loss, aux_loss_weight, top_k):
    """Trains the model and caches it for future runs."""
    progress_bar = st.progress(0, text="Training model, please wait...")
    def update_progress(fraction):
        progress_bar.progress(fraction)
    
    model = train_model(
        progress_callback=update_progress, 
        weighted_loss=weighted_loss, 
        aux_loss_weight=aux_loss_weight,
        top_k=top_k
    )
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

architecture_diagram = '''
digraph CyberMoE {
    rankdir=TB;
    graph [bgcolor="transparent"];
    node [shape=box, style="rounded,filled", fillcolor="#444444", fontname="sans-serif", fontcolor="white"];
    edge [fontname="sans-serif", color="white", fontcolor="white"];

    "Input Text" [shape=plaintext, fontcolor="white"];
    "Shared Encoder (BERT)" [fillcolor="#004080"];
    "Gating Network" [fillcolor="#808000"];
    "Top-K Routing" [shape=diamond, fillcolor="#b35900"];

    subgraph cluster_experts {
        label = "Specialized Experts";
        style = "rounded";
        bgcolor = "#333333";
        fontcolor = "white";
        "Expert 1 (Network)" [fillcolor="#800000"];
        "Expert 2 (Malware)" [fillcolor="#800000"];
        "Expert 3 (Phishing)" [fillcolor="#800000"];
        "Expert 4 (Cloud Security)" [fillcolor="#800000"];
        "Expert 5 (Web App Security)" [fillcolor="#800000"];
    }

    "Final Prediction" [shape=plaintext, fillcolor="#006400"];

    "Input Text" -> "Shared Encoder (BERT)";
    "Shared Encoder (BERT)" -> "Gating Network";
    "Shared Encoder (BERT)" -> "Expert 1 (Network)" [style=dashed, arrowhead=none];
    "Shared Encoder (BERT)" -> "Expert 2 (Malware)" [style=dashed, arrowhead=none];
    "Shared Encoder (BERT)" -> "Expert 3 (Phishing)" [style=dashed, arrowhead=none];
    "Shared Encoder (BERT)" -> "Expert 4 (Cloud Security)" [style=dashed, arrowhead=none];
    "Shared Encoder (BERT)" -> "Expert 5 (Web App Security)" [style=dashed, arrowhead=none];

    "Gating Network" -> "Top-K Routing";
    "Top-K Routing" -> "Expert 1 (Network)" [style=dotted];
    "Top-K Routing" -> "Expert 2 (Malware)" [style=dotted];
    "Top-K Routing" -> "Expert 3 (Phishing)" [style=dotted];
    "Top-K Routing" -> "Expert 4 (Cloud Security)" [style=dotted];
    "Top-K Routing" -> "Expert 5 (Web App Security)" [style=dotted];

    "Expert 1 (Network)" -> "Final Prediction" [style=dotted];
    "Expert 2 (Malware)" -> "Final Prediction" [style=dotted];
    "Expert 3 (Phishing)" -> "Final Prediction" [style=dotted];
    "Expert 4 (Cloud Security)" -> "Final Prediction" [style=dotted];
    "Expert 5 (Web App Security)" -> "Final Prediction" [style=dotted];
}
'''
with st.container(border=True):
    st.graphviz_chart(architecture_diagram)


# --- Sidebar Controls ---
st.sidebar.title("⚙️ Controls")

top_k = st.sidebar.slider("Number of Experts to Use (Top-K)", min_value=1, max_value=NUM_EXPERTS, value=2, step=1)
use_weighted_loss = st.sidebar.checkbox("Use Weighted Loss", value=True)
use_aux_loss = st.sidebar.checkbox("Use Auxiliary Load Balancing Loss", value=True)
aux_loss_weight = st.sidebar.number_input(
    "Auxiliary Loss Weight", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.01, 
    step=0.01, 
    disabled=not use_aux_loss
)

if not use_aux_loss:
    aux_loss_weight = 0.0

if st.sidebar.button("Restart Demo (retrain model)"):
    st.cache_resource.clear()
    if 'analysis_results' in st.session_state:
        del st.session_state.analysis_results
    st.rerun()

# Load the model
model = load_model(use_weighted_loss, aux_loss_weight, top_k)
expert_names = ["Network", "Malware", "Phishing", "Cloud Security", "Web App Security"]

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
            final_logits, gating_probs, expert_logits, domain_pred = model([user_input])
        st.session_state.analysis_results = (final_logits, gating_probs, expert_logits, domain_pred)
        st.session_state.user_input_for_analysis = user_input
    else:
        st.warning("Please enter some text to analyze.")
        if 'analysis_results' in st.session_state:
            del st.session_state.analysis_results

if 'analysis_results' in st.session_state:
    final_logits, gating_probs, expert_logits, domain_pred = st.session_state.analysis_results
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
        
        # Display predicted domain
        domain_probs = torch.softmax(domain_pred, dim=-1)[0]
        predicted_domain = expert_names[domain_probs.argmax()]
        st.metric(label="Predicted Domain", value=predicted_domain, delta=f"{domain_probs.max():.2%} confidence")
        
        st.subheader("Gating Network Scores")
        st.write("The gating network decides which expert is most relevant. Higher is better.")
        gating_df = pd.DataFrame({
            "Expert": expert_names,
            "Score": gating_probs.cpu().numpy().flatten()
        })
        st.bar_chart(gating_df, x="Expert", y="Score")

    # --- Column 2: Expert Activation ---
    with col2:
        st.subheader(f"Sparse Activation (Top-{top_k} Experts)")
        st.write("Only the most relevant experts are used to save computation.")

        topk_vals, topk_idx = torch.topk(gating_probs[0], top_k)
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
    st.header("🕵️‍♀️ Explain the Gating Decision")
    st.write("See which words in your input sentence were most influential for a specific expert's score.")
    
    expert_to_explain = st.selectbox("Select expert to explain:", expert_names)
    
    if st.button("Explain Routing"):
        if 'user_input_for_analysis' in st.session_state:
            text_to_explain = st.session_state.user_input_for_analysis
            word_importances = model.explain_gating(text_to_explain, expert_names.index(expert_to_explain))
            
            # Display as a table for simplicity
            explanation_df = pd.DataFrame(word_importances, columns=["Token", "Importance"])
            explanation_df = explanation_df[explanation_df.Token.isin(["[CLS]", "[SEP]"]) == False]
            st.dataframe(explanation_df)
        else:
            st.warning("Please analyze a sentence first.")


# --- Sidebar Explanation ---
st.sidebar.title("💡 How It Works")
st.sidebar.info(f"""
1.  **Shared Encoder**: The input text is converted into a numerical representation by a single, shared `bert-base-uncased` model.

2.  **Gating Network**: A small neural network analyzes this representation and assigns a relevance score to each specialized expert.

3.  **Sparse Routing**: To be efficient, the model only activates the **Top-{top_k}** experts with the highest scores. The other experts are skipped entirely.

4.  **Final Decision**: The outputs of the activated experts are combined, weighted by their scores, to produce the final classification.

**Advanced Training Features:**
- **Weighted Loss:** Gives more importance to the under-represented "Benign" class during training to improve accuracy.
- **Auxiliary Load Balancing Loss:** Encourages the gating network to use all experts more evenly, preventing it from relying too heavily on just a few.
""")