import io
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "risk_clause_model.joblib"
DATA_PATH = APP_DIR / "sample_clauses.csv"

RISK_LABELS = {
    "payment_risk": "Payment Risk",
    "termination_risk": "Termination Risk",
    "liability_risk": "Liability Risk",
    "confidentiality_risk": "Confidentiality Risk",
    "indemnity_risk": "Indemnity Risk",
    "no_apparent_risk": "No Apparent Risk",
}

EXPLANATIONS = {
    "payment_risk": "The clause may involve unclear payment timing, fees, penalties, or payment obligations.",
    "termination_risk": "The clause may allow termination with limited notice or under broad conditions.",
    "liability_risk": "The clause may create broad, uncapped, or one-sided liability obligations.",
    "confidentiality_risk": "The clause may provide unclear or potentially insufficient confidentiality protections.",
    "indemnity_risk": "The clause may create broad obligations to compensate or defend another party.",
    "no_apparent_risk": "The model did not identify a clear match to the risk categories it was trained on. This does not guarantee that the clause is safe.",
}

def split_clauses(text):
    text = text.replace("\r", "\n")
    # First split on numbered clause headings, then on paragraph boundaries / sentence endings.
    chunks = re.split(
        r"(?im)(?=^\s*(?:\d+(?:\.\d+)*[.)]?\s+|section\s+\d+|article\s+[ivx\d]+))",
        text
    )
    clauses = []
    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if not chunk:
            continue
        # Break long blocks into sentence-level units while keeping short clauses together.
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z])", chunk)
        for piece in pieces:
            piece = piece.strip(" •\t")
            if len(piece) >= 25:
                clauses.append(piece)
    return clauses

def extract_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix == ".txt":
        return data.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            return "\n".join(page.get_text("text") for page in doc)
        except Exception as exc:
            raise ValueError(f"Could not read this PDF. If it is a scanned image PDF, OCR is not included. Details: {exc}")
    if suffix == ".docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    paragraphs.append(" | ".join(cell.text for cell in row.cells))
            return "\n".join(paragraphs)
        except Exception as exc:
            raise ValueError(f"Could not read this DOCX file: {exc}")
    raise ValueError("Unsupported file type. Upload a .txt, .pdf, or .docx file.")

def train_model():
    df = pd.read_csv(DATA_PATH)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode", sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(df["text"].astype(str), df["label"].astype(str))
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return train_model()

def main():
    st.set_page_config(page_title="Legal Contract Risk-Clause Detection", page_icon="⚖️", layout="wide")
    st.title("⚖️ Legal Contract Risk-Clause Detection Using NLP")
    st.write("Upload a contract to extract clauses and screen them for potentially risky language using a TF-IDF + Logistic Regression NLP model.")
    st.warning(
        "Educational prototype only — not legal advice. Predictions depend on a small illustrative training dataset and may be wrong. "
        "A clause marked “No Apparent Risk” is not guaranteed to be safe. Have important contracts reviewed by a qualified lawyer."
    )

    with st.sidebar:
        st.header("How it works")
        st.markdown("1. Upload PDF, DOCX, or TXT\n2. Extract and split text into clauses\n3. Predict a risk category for each clause\n4. Review highlighted results and export CSV")
        st.caption("The included dataset is a small demonstration dataset, not a validated legal benchmark.")

    upload = st.file_uploader("Upload a contract", type=["pdf", "docx", "txt"])
    demo = st.checkbox("Use built-in demo contract instead")
    if demo:
        contract_text = """1. Payment. The customer must pay all invoices within 30 days of receipt. Late payments may incur a fee of 5 percent per month.
2. Termination. The provider may terminate this agreement at any time without prior notice and without giving a reason.
3. Limitation of Liability. The supplier's total liability shall be unlimited for any loss, whether direct, indirect, incidental, or consequential.
4. Confidentiality. Each party must protect confidential information and may disclose it only when required by law.
5. Indemnification. The customer shall indemnify, defend, and hold harmless the provider from all claims, losses, damages, costs, and expenses arising from any use of the services.
6. Notices. Notices under this agreement must be sent in writing to the addresses listed by each party."""
        st.info("Demo contract loaded. Select Analyze Contract to see example results.")
    elif upload:
        try:
            contract_text = extract_text(upload)
        except ValueError as exc:
            st.error(str(exc))
            return
    else:
        contract_text = ""

    with st.expander("Preview extracted contract text", expanded=False):
        st.text_area("Extracted text", contract_text, height=220)

    if st.button("Analyze Contract", type="primary", disabled=not bool(contract_text.strip())):
        clauses = split_clauses(contract_text)
        if not clauses:
            st.error("No clauses long enough to analyze were found. Try a text-based PDF or DOCX, or upload a TXT file.")
            return
        model = get_model()
        predictions = model.predict(clauses)
        probabilities = model.predict_proba(clauses)
        confidence = probabilities.max(axis=1)
        results = pd.DataFrame({
            "Clause #": range(1, len(clauses) + 1),
            "Clause Text": clauses,
            "Predicted Category": [RISK_LABELS.get(x, x.replace("_", " ").title()) for x in predictions],
            "Model Confidence": [round(float(x) * 100, 1) for x in confidence],
            "Explanation": [EXPLANATIONS.get(x, "Review this clause carefully.") for x in predictions],
        })
        st.session_state["results"] = results
        st.session_state["predictions_raw"] = list(predictions)

    if "results" in st.session_state:
        results = st.session_state["results"]
        st.subheader("Analysis Summary")
        total = len(results)
        flagged = int((results["Predicted Category"] != "No Apparent Risk").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Clauses analyzed", total)
        c2.metric("Potentially flagged", flagged)
        c3.metric("Categories found", results.loc[results["Predicted Category"] != "No Apparent Risk", "Predicted Category"].nunique())

        counts = results["Predicted Category"].value_counts().rename_axis("Category").reset_index(name="Count")
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Category breakdown**")
            st.dataframe(counts, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Clause-level results**")
            st.dataframe(results[["Clause #", "Predicted Category", "Model Confidence"]], use_container_width=True, hide_index=True)

        st.subheader("Flagged Clauses")
        flagged_rows = results[results["Predicted Category"] != "No Apparent Risk"]
        if flagged_rows.empty:
            st.info("No clauses were flagged in the selected risk categories. This is not a guarantee that the contract is safe.")
        else:
            for _, row in flagged_rows.iterrows():
                with st.expander(f"Clause {row['Clause #']} — {row['Predicted Category']} ({row['Model Confidence']}% model confidence)"):
                    st.write(row["Clause Text"])
                    st.caption(row["Explanation"])
                    st.caption("Model confidence is not the same as legal risk severity or prediction accuracy.")
        st.subheader("All Clauses")
        st.dataframe(results, use_container_width=True, hide_index=True)
        csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download analysis as CSV", data=csv_bytes, file_name="contract_risk_analysis.csv", mime="text/csv")
        st.download_button("Download extracted clauses as TXT", data="\n\n".join(f"{i+1}. {c}" for i, c in enumerate(results["Clause Text"])).encode("utf-8"), file_name="extracted_clauses.txt", mime="text/plain")

if __name__ == "__main__":
    main()
