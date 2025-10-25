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
from feedback import log_feedback, DEFAULT_FEEDBACK_PATH
from rlhf_reward_model import train_reward_model
from finetune_from_feedback import finetune_from_feedback, load_finetuned_model

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

# --- Feedback Export/Import ---
st.sidebar.markdown("---")
st.sidebar.title("🗂️ Feedback Data")
from io import BytesIO
import os

# Export current feedback file if it exists
if os.path.exists(DEFAULT_FEEDBACK_PATH):
    try:
        with open(DEFAULT_FEEDBACK_PATH, "rb") as f:
            data_bytes = f.read()
        st.sidebar.download_button(
            label="Download feedback.jsonl",
            data=data_bytes,
            file_name="feedback.jsonl",
            mime="application/json",
            use_container_width=True
        )
    except Exception as e:
        st.sidebar.error(f"Failed to load feedback for download: {e}")
else:
    st.sidebar.info("No feedback.jsonl found yet.")

# Import feedback.jsonl (append or replace)
uploaded_file = st.sidebar.file_uploader("Upload feedback.jsonl", type=["jsonl", "json"], accept_multiple_files=False)
import_action = st.sidebar.radio("Import mode", ["Append", "Replace"], horizontal=True)
# Confirmation for destructive replace
confirm_replace = False
if import_action == "Replace":
    confirm_replace = st.sidebar.checkbox("Confirm replace (destructive)", value=False)

# Basic JSONL schema validation helper
def _validate_feedback_jsonl(content: bytes, max_errors: int = 5):
    import json
    try:
        text = content.decode("utf-8")
    except Exception as e:
        return False, None, {"errors": [f"File is not valid UTF-8: {e}"]}
    lines = text.splitlines()
    out_lines = []
    errors = []
    valid = 0
    def _norm_label(val):
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("benign", "malicious"):
                return "Malicious" if low == "malicious" else "Benign"
            if low in ("1", "true", "yes"):
                return "Malicious"
            if low in ("0", "false", "no"):
                return "Benign"
        if isinstance(val, (int, float)):
            return "Malicious" if int(val) == 1 else "Benign"
        return None
    def _norm_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("true", "yes", "1"): return True
            if low in ("false", "no", "0"): return False
        return None
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception as e:
            errors.append(f"Line {idx}: invalid JSON ({e})")
            if len(errors) >= max_errors:
                break
            continue
        if not isinstance(rec, dict):
            errors.append(f"Line {idx}: not a JSON object")
            if len(errors) >= max_errors:
                break
            continue
        # Required fields
        ui = rec.get("user_input")
        if not isinstance(ui, str) or not ui.strip():
            errors.append(f"Line {idx}: missing or invalid 'user_input' (string required)")
        uf = _norm_bool(rec.get("user_feedback"))
        if uf is None:
            errors.append(f"Line {idx}: missing or invalid 'user_feedback' (bool or 'true'/'false')")
        else:
            rec["user_feedback"] = uf
        # Optional normalization
        pl = rec.get("pred_label")
        if pl is not None:
            nl = _norm_label(pl)
            if nl is None:
                errors.append(f"Line {idx}: invalid 'pred_label' (expected Benign/Malicious)")
            else:
                rec["pred_label"] = nl
        corr = rec.get("correction")
        if corr is not None:
            nl = _norm_label(corr)
            if nl is None:
                errors.append(f"Line {idx}: invalid 'correction' (expected Benign/Malicious)")
            else:
                rec["correction"] = nl
        if errors and len(errors) >= max_errors:
            break
        if not errors or (errors and not errors[-1].startswith(f"Line {idx}:")):
            # Only count as valid if no new error added for this line
            valid += 1
            out_lines.append(json.dumps(rec, ensure_ascii=False))
    if errors:
        return False, None, {"errors": errors[:max_errors], "valid": valid}
    if valid == 0:
        return False, None, {"errors": ["No valid JSONL records found."], "valid": 0}
    normalized = ("\n".join(out_lines) + "\n").encode("utf-8")
    return True, normalized, {"valid": valid}

if uploaded_file is not None:
    try:
        os.makedirs(os.path.dirname(DEFAULT_FEEDBACK_PATH), exist_ok=True)
        content = uploaded_file.getvalue()
        ok, normalized, info = _validate_feedback_jsonl(content)
        if not ok:
            errs = info.get("errors", []) if isinstance(info, dict) else []
            st.sidebar.error("Upload failed schema check.\n" + "\n".join(errs))
        else:
            if import_action == "Replace":
                if not confirm_replace:
                    st.sidebar.warning("Check 'Confirm replace (destructive)' to overwrite existing feedback.jsonl.")
                else:
                    with open(DEFAULT_FEEDBACK_PATH, "wb") as f:
                        f.write(normalized)
                    st.sidebar.success(f"Replaced feedback.jsonl with {info.get('valid', 0)} records.")
            else:
                # Append mode: ensure file ends with newline before appending normalized JSONL
                needs_nl = False
                if os.path.exists(DEFAULT_FEEDBACK_PATH) and os.path.getsize(DEFAULT_FEEDBACK_PATH) > 0:
                    with open(DEFAULT_FEEDBACK_PATH, "rb") as rf:
                        try:
                            rf.seek(-1, os.SEEK_END)
                            needs_nl = rf.read(1) != b"\n"
                        except Exception:
                            needs_nl = True
                with open(DEFAULT_FEEDBACK_PATH, "ab") as f:
                    if needs_nl:
                        f.write(b"\n")
                    f.write(normalized)
                st.sidebar.success(f"Appended {info.get('valid', 0)} records to feedback.jsonl.")
    except Exception as e:
        st.sidebar.error(f"Failed to import feedback: {e}")

# Controls for persisted model
col_load, col_clear = st.sidebar.columns(2)
if col_load.button("Load Fine-Tuned"):
    loaded = load_finetuned_model()
    if loaded is not None:
        st.session_state.override_model = loaded
        st.sidebar.success("Fine-tuned model loaded.")
    else:
        st.sidebar.info("No fine-tuned checkpoint found.")

if col_clear.button("Clear Fine-Tuned"):
    st.session_state.override_model = None
    st.sidebar.info("Fine-tuned model cleared for this session.")

# --- Feedback Analytics (load helper) ---
@st.cache_data
def load_feedback_df(path: str = DEFAULT_FEEDBACK_PATH, _mtime: float | None = None):
    import json
    import os
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                rows.append(rec)
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Parse timestamp and normalize types
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Ensure lists exist even if missing
    for col in ["gating_scores", "expert_logits", "domain_scores"]:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
    return df

# Load the model (auto-load fine-tuned checkpoint if present; allow session override)
if 'override_model' in st.session_state and st.session_state.override_model is not None:
    model = st.session_state.override_model
else:
    persisted = load_finetuned_model()
    if persisted is not None:
        model = persisted
        st.sidebar.success("Loaded fine-tuned model from checkpoint.")
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

    # --- Feedback Analytics Section ---
    st.header("📈 Feedback Analytics")
    import os
    fb_mtime = os.path.getmtime(DEFAULT_FEEDBACK_PATH) if os.path.exists(DEFAULT_FEEDBACK_PATH) else None
    fb_df = load_feedback_df(_mtime=fb_mtime)
    if fb_df is None or fb_df.empty:
        st.info("No feedback collected yet. Submit feedback above to populate analytics.")
    else:
        # Basic KPIs
        total = len(fb_df)
        correct = int(fb_df.get("user_feedback", pd.Series([False]*total)).astype(bool).sum())
        incorrect = total - correct
        accuracy = (correct / total) if total > 0 else 0.0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Feedback", total)
        kpi2.metric("Correct", correct)
        kpi3.metric("Incorrect", incorrect)
        kpi4.metric("Observed Accuracy", f"{accuracy*100:.1f}%")

        # Distributions
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Prediction Labels")
            if "pred_label" in fb_df.columns:
                st.bar_chart(fb_df["pred_label"].value_counts())
            else:
                st.write("No prediction labels found.")
        with colB:
            st.subheader("Corrections (when incorrect)")
            if "correction" in fb_df.columns:
                st.bar_chart(fb_df[fb_df["user_feedback"] == False]["correction"].fillna("None").value_counts())
            else:
                st.write("No corrections logged.")

        colC, colD = st.columns(2)
        with colC:
            st.subheader("Predicted Domain")
            if "domain_pred" in fb_df.columns:
                st.bar_chart(fb_df["domain_pred"].value_counts())
            else:
                st.write("No domain predictions found.")
        with colD:
            st.subheader("Avg Confidence by Outcome")
            if "pred_confidence" in fb_df.columns and "user_feedback" in fb_df.columns:
                conf_grp = fb_df.groupby(fb_df["user_feedback"].astype(bool))["pred_confidence"].mean()
                conf_grp.index = conf_grp.index.map(lambda x: "Correct" if x else "Incorrect")
                st.bar_chart(conf_grp)
            else:
                st.write("Confidence not available.")

        # Average gating and domain scores
        with st.expander("Model Signals Averages"):
            try:
                # Expand gating scores into DataFrame columns
                gating_expanded = fb_df["gating_scores"].apply(lambda x: x if isinstance(x, list) else [])
                max_len = int(gating_expanded.apply(lambda l: len(l) if isinstance(l, list) else 0).max())
                gating_cols = pd.DataFrame(gating_expanded.apply(lambda l: (l + [0.0]*(max_len-len(l)))[:max_len] if isinstance(l, list) else [0.0]*max_len))
                # The above yields a DataFrame column of lists; convert properly
                gating_matrix = pd.DataFrame(gating_cols[0].tolist()) if max_len > 0 else pd.DataFrame()
                if not gating_matrix.empty:
                    gating_matrix.columns = [f"Expert {i+1}" for i in range(max_len)]
                    st.bar_chart(gating_matrix.mean())
                else:
                    st.write("No gating scores to aggregate.")
            except Exception:
                st.write("Failed to aggregate gating scores.")

            try:
                domain_expanded = fb_df["domain_scores"].apply(lambda x: x if isinstance(x, list) else [])
                dmax = int(domain_expanded.apply(lambda l: len(l) if isinstance(l, list) else 0).max())
                domain_cols = pd.DataFrame(domain_expanded.apply(lambda l: (l + [0.0]*(dmax-len(l)))[:dmax] if isinstance(l, list) else [0.0]*dmax))
                domain_matrix = pd.DataFrame(domain_cols[0].tolist()) if dmax > 0 else pd.DataFrame()
                if not domain_matrix.empty:
                    domain_matrix.columns = expert_names[:dmax]
                    st.bar_chart(domain_matrix.mean())
                else:
                    st.write("No domain scores to aggregate.")
            except Exception:
                st.write("Failed to aggregate domain scores.")

        # Time-series accuracy
        if "timestamp" in fb_df.columns and fb_df["timestamp"].notna().any():
            st.subheader("Accuracy Over Time")
            ts = fb_df.dropna(subset=["timestamp"])\
                .assign(correct=fb_df["user_feedback"].astype(bool))\
                .set_index("timestamp")["correct"]\
                .resample("1H").mean().fillna(0.0)
            st.line_chart(ts)

        # Recent feedback table
        st.subheader("Most Recent Feedback")
        show_cols = [c for c in ["timestamp","user_input","pred_label","pred_confidence","user_feedback","correction","domain_pred","domain_confidence","notes"] if c in fb_df.columns]
        st.dataframe(fb_df.sort_values(by="timestamp", ascending=False)[show_cols].head(25), use_container_width=True)


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

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>MIT License | Copyright © 2025 Ron F. del Rosario</p>
    </div>
    """,
    unsafe_allow_html=True
)