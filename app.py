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
from preprocessor import CyberPreprocessor
from feedback import log_feedback
from rlhf_reward_model import train_reward_model
from finetune_from_feedback import finetune_from_feedback

# --------------------------------------------------------------------------- #
# Model Loading (with caching)
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_model(weighted_loss, aux_loss_weight, top_k, use_pretraining):
    """Trains the model and caches it for future runs."""
    progress_bar = st.progress(0, text="Training model, please wait...")
    def update_progress(fraction):
        progress_bar.progress(fraction)
    
    model = train_model(
        progress_callback=update_progress, 
        weighted_loss=weighted_loss, 
        aux_loss_weight=aux_loss_weight,
        top_k=top_k,
        pretrain_experts=use_pretraining,
        pretrain_epochs=5,
        pretrain_lr=1e-4
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

# Pre-training control
use_pretraining = st.sidebar.checkbox("Use Expert Pre-training", value=True)
if use_pretraining:
    st.sidebar.markdown(
        """ℹ️ Pre-training helps each expert develop domain-specific knowledge
        before joint training. This can improve specialization and accuracy."""
    )

if not use_aux_loss:
    aux_loss_weight = 0.0

if st.sidebar.button("Restart Demo (retrain model)"):
    st.cache_resource.clear()
    if 'analysis_results' in st.session_state:
        del st.session_state.analysis_results
    st.rerun()

# --- RLHF Training Utilities ---
st.sidebar.markdown("---")
st.sidebar.title("🧪 RLHF Utilities")
if st.sidebar.button("Train Reward Model from Feedback"):
    with st.spinner("Training reward model from data/feedback.jsonl..."):
        try:
            save_path = train_reward_model()
            st.sidebar.success(f"Reward model trained and saved to: {save_path}")
        except Exception as e:
            st.sidebar.error(f"Failed to train reward model: {e}")

if st.sidebar.button("Fine-tune Model from Feedback"):
    with st.spinner("Fine-tuning CyberMoE from feedback..."):
        try:
            ft_model = finetune_from_feedback()
            st.session_state.override_model = ft_model.eval()
            st.sidebar.success("Fine-tuned model is now active for this session.")
        except Exception as e:
            st.sidebar.error(f"Failed to fine-tune from feedback: {e}")

# Load the model (use fine-tuned model if available)
if 'override_model' in st.session_state and st.session_state.override_model is not None:
    model = st.session_state.override_model
else:
    model = load_model(use_weighted_loss, aux_loss_weight, top_k, use_pretraining)
expert_names = ["Network", "Malware", "Phishing", "Cloud Security", "Web App Security"]

# --- Input Area ---
st.header("Analyze a Security Event")


# Initialize preprocessor
preprocessor = CyberPreprocessor()

user_input = st.text_area(
    "Enter a sentence to classify:", 
    "Suspicious login attempt from unknown IP address",
    height=100
)


if st.button("Analyze", use_container_width=True):
    if user_input:
        # Extract technical features
        preprocessed = preprocessor.process(user_input)
        st.session_state.preprocessed = preprocessed
        
        with torch.no_grad():
            outputs = model([user_input])
            final_logits = outputs['logits']
            gating_probs = outputs['routing_probs']
            expert_logits = outputs['expert_outputs']
            domain_pred = outputs['domain_logits']
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

    # --- Column 2: Technical Analysis ---
    with col2:
        st.subheader("🔍 Technical Features")
        if 'preprocessed' in st.session_state:
            features = st.session_state.preprocessed['features']
            
            # Create a clean display of detected features
            feature_data = []
            for feature_type, items in features.items():
                if items:  # Only show non-empty features
                    feature_data.append({
                        "Type": feature_type.upper(),
                        "Count": len(items),
                        "Values": ", ".join(items)
                    })
            
            if feature_data:
                feature_df = pd.DataFrame(feature_data)
                st.dataframe(feature_df, use_container_width=True)
            else:
                st.info("No technical features detected in input text")
            
            # Show domain relevance scores
            st.subheader("📊 Domain Relevance Scores")
            domain_scores = st.session_state.preprocessed['domain_scores'].numpy()
            domain_score_df = pd.DataFrame({
                "Domain": expert_names,
                "Relevance Score": domain_scores
            })
            st.bar_chart(domain_score_df.set_index("Domain"))
        
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

    # --- RLHF Feedback Section ---
    st.header("✅ Provide Feedback (RLHF)")
    st.write("Help improve the model by telling us if the analysis was correct.")

    with st.form("feedback_form", clear_on_submit=True):
        was_correct = st.radio("Was the classification correct?", ("Yes", "No"), index=0, horizontal=True)
        corrected_label = st.selectbox("If not correct, what should it be?", ("Benign", "Malicious"))
        user_notes = st.text_area("Additional notes (optional)", "")
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        # Collect current context for feedback logging
        probs = torch.softmax(final_logits, dim=-1).cpu().numpy().flatten()
        pred_label = int(probs.argmax())
        pred_conf = float(probs[pred_label])

        domain_probs = torch.softmax(domain_pred, dim=-1)[0].cpu().numpy().flatten()
        domain_idx = int(domain_probs.argmax())
        domain_conf = float(domain_probs[domain_idx])

        gating_scores = gating_probs[0].detach().cpu().numpy().tolist()
        experts_array = expert_logits[0].detach().cpu().numpy().tolist()
        features = st.session_state.preprocessed.get('features', {}) if 'preprocessed' in st.session_state else {}
        domain_scores = st.session_state.preprocessed.get('domain_scores', torch.tensor([])).detach().cpu().numpy().tolist() if 'preprocessed' in st.session_state else []

        feedback_record = {
            "user_input": st.session_state.get('user_input_for_analysis', ''),
            "pred_label": "Malicious" if pred_label == 1 else "Benign",
            "pred_confidence": pred_conf,
            "domain_pred": expert_names[domain_idx],
            "domain_confidence": domain_conf,
            "gating_scores": gating_scores,
            "expert_logits": experts_array,
            "features": features,
            "domain_scores": domain_scores,
            "user_feedback": True if was_correct == "Yes" else False,
            "correction": None if was_correct == "Yes" else corrected_label,
            "notes": user_notes,
        }

        try:
            log_feedback(feedback_record)
            st.success("Thanks! Your feedback was recorded.")
        except Exception as e:
            st.error(f"Failed to record feedback: {e}")


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