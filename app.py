"""Credit Card Fraud Detection app (Streamlit).

Run:  streamlit run app.py
Needs model.pkl and sample_transactions.csv in the same folder.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(HERE / "model.pkl")


@st.cache_data
def load_samples():
    return pd.read_csv(HERE / "sample_transactions.csv")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Probability of fraud for each row. The model was trained on the raw
    columns (Time, V1..V28, Amount), so no scaling is applied."""
    return model.predict_proba(df[FEATURES])[:, 1]


model = load_model()
samples = load_samples()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Fraud threshold", 0.05, 0.95, 0.50, 0.05,
        help="A transaction is flagged as fraud when its fraud probability is at or above this value. "
             "Lower = catches more fraud but more false alarms.",
    )
    st.caption("Model: Logistic Regression\n\nInputs: Time, V1–V28 (PCA features), Amount")

st.title("💳 Credit Card Fraud Detection")
st.write("Check a single transaction or upload a whole file. The model returns the probability that a transaction is fraudulent.")

tab_single, tab_batch, tab_about = st.tabs(["Single transaction", "Batch (CSV)", "About"])

# ---------- Single transaction ----------
with tab_single:
    st.subheader("Check one transaction")
    mode = st.radio("Input", ["Pick a sample transaction", "Enter values manually"], horizontal=True)

    if mode == "Pick a sample transaction":
        kind = st.selectbox("Sample type", ["Any", "Legitimate (Class 0)", "Fraud (Class 1)"])
        pool = samples
        if kind.startswith("Legit"):
            pool = samples[samples["Class"] == 0]
        elif kind.startswith("Fraud"):
            pool = samples[samples["Class"] == 1]
        idx = st.number_input("Sample number", 0, len(pool) - 1, 0, step=1)
        row = pool.iloc[int(idx)]
        values = {f: float(row[f]) for f in FEATURES}
        c1, c2 = st.columns(2)
        c1.metric("Amount", f"${values['Amount']:,.2f}")
        c2.metric("Time (sec since first transaction)", f"{values['Time']:,.0f}")
        true_label = int(row["Class"])
    else:
        true_label = None
        c1, c2 = st.columns(2)
        time_v = c1.number_input("Time (seconds)", 0.0, 1_000_000.0, 50000.0)
        amount_v = c2.number_input("Amount ($)", 0.0, 100_000.0, 50.0)
        values = {"Time": time_v, "Amount": amount_v}
        with st.expander("Advanced: PCA features V1–V28 (default 0)"):
            cols = st.columns(4)
            for i in range(1, 29):
                values[f"V{i}"] = cols[(i - 1) % 4].number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

    if st.button("🔍 Check transaction", type="primary"):
        prob = float(predict_proba(pd.DataFrame([values]))[0])
        is_fraud = prob >= threshold
        if is_fraud:
            st.error(f"🚨 **Likely FRAUD** — fraud probability {prob:.1%}")
        else:
            st.success(f"✅ **Looks legitimate** — fraud probability {prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        if true_label is not None:
            st.caption(f"Actual label in dataset: **{'Fraud' if true_label else 'Legitimate'}**")

# ---------- Batch ----------
with tab_batch:
    st.subheader("Check many transactions")
    st.write("Upload a CSV with columns `Time, V1…V28, Amount` (an optional `Class` column is used to measure accuracy).")
    use_demo = st.checkbox("Use the built-in sample file instead", value=False)
    up = None if use_demo else st.file_uploader("CSV file", type="csv")

    data = samples if use_demo else (pd.read_csv(up) if up is not None else None)
    if data is not None:
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            out = data.copy()
            out["fraud_probability"] = predict_proba(out)
            out["prediction"] = np.where(out["fraud_probability"] >= threshold, "Fraud", "Legit")

            n_flag = int((out["prediction"] == "Fraud").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Transactions", f"{len(out):,}")
            m2.metric("Flagged as fraud", f"{n_flag:,}")
            m3.metric("Flag rate", f"{n_flag / len(out):.2%}")

            if "Class" in out.columns:
                y, pred = out["Class"].astype(int), (out["prediction"] == "Fraud").astype(int)
                tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                st.markdown("**Performance vs. the `Class` column**")
                a, b, c, d = st.columns(4)
                a.metric("Precision", f"{prec:.1%}"); b.metric("Recall", f"{rec:.1%}")
                c.metric("Frauds caught", f"{tp} / {tp + fn}"); d.metric("False alarms", f"{fp}")
                st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual Legit", "Actual Fraud"],
                                          columns=["Predicted Legit", "Predicted Fraud"]))

            show = out.sort_values("fraud_probability", ascending=False)
            cols_first = ["prediction", "fraud_probability", "Amount", "Time"] + (["Class"] if "Class" in out.columns else [])
            st.dataframe(show[cols_first + [c for c in show.columns if c not in cols_first]].head(500),
                         use_container_width=True)
            st.download_button("⬇️ Download all results (CSV)", out.to_csv(index=False).encode(),
                               "fraud_predictions.csv", "text/csv")

# ---------- About ----------
with tab_about:
    st.subheader("About this model")
    st.markdown(
        """
- **Algorithm:** Logistic Regression (scikit-learn)
- **Data:** the public credit-card dataset (284,807 transactions, only 492 frauds ≈ 0.17%), where
  `V1–V28` are anonymised PCA components, plus `Time` and `Amount`.
- **Threshold:** change it in the sidebar. The model is trained on very imbalanced data, so the threshold
  trades off *catching more fraud* against *more false alarms*.
- **Note:** this is a learning/demo project, not a production fraud system.
"""
    )