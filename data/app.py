import streamlit as st
import pandas as pd
import os
import subprocess
import platform
from openai import OpenAI

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Sepsis Atlas",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# API AUTH & CONFIG
# =====================================================
API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

MODEL = "google/gemini-2.0-flash-001"

# =====================================================
# DATA LOADING
# =====================================================
FILE = "Sepsis_Evidence_Table.csv"
PDF_FOLDER = "pdf"

@st.cache_data
def load_data():
    if not os.path.exists(FILE):
        return None
    try:
        df = pd.read_csv(FILE, on_bad_lines="skip")
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        return df
    except Exception as e:
        st.error(f"CSV ERROR: {e}")
        return None

df = load_data()

def open_pdf_manual(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        st.error(f"Error opening PDF: {e}")

# =====================================================
# SESSION STATE
# =====================================================
if "report" not in st.session_state:
    st.session_state.report = ""
if "matched_df" not in st.session_state:
    st.session_state.matched_df = None

# =====================================================
# MAIN UI
# =====================================================
st.title("🏥 Sepsis Atlas")
st.markdown("### AI-Powered Clinical Evidence Synthesizer")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("📋 Patient Clinical Profile")
    patient_desc = st.text_area(
        "Describe Patient",
        height=400,
        placeholder="Input patient vitals or ask a clinical audit question..."
    )
    run_btn = st.button("🔬 Run Evidence Audit", type="primary", use_container_width=True)

with right:
    st.subheader("📊 Grounded Clinical Evidence")

    if run_btn and patient_desc:
        if df is None:
            st.error("Evidence table missing.")
            st.stop()

        with st.spinner("Executing Deterministic Audit..."):
            # 1. HARD-STOP KEYWORD CHECK (The "Crazy Claim" Filter)
            query = patient_desc.lower()
            keyword_bank = ["lactate", "sofa", "apache", "il-6", "lymphocyte", "vasopressor", "shock", "icu", "mortality", "auc", "odds ratio"]
            keywords = [k for k in keyword_bank if k in query]

            # Trigger immediate failure for non-clinical queries
            if not keywords:
                st.session_state.report = "## ⚠️ AUDIT FAILED\n**Reason:** No validated clinical biomarkers or protocols detected. The Atlas refuses to synthesize data for non-indexed claims."
                st.session_state.matched_df = pd.DataFrame()
                st.rerun()

            # 2. FILTER DATA BASED ON KEYWORDS
            mask = pd.Series(False, index=df.index)
            cols = ["study_name", "predictor", "population", "effect_size", "notes", "outcome"]
            for col in [c for c in cols if c in df.columns]:
                mask |= df[col].astype(str).str.lower().str.contains('|'.join(keywords), na=False)
            
            matched_df = df[mask].head(6)
            st.session_state.matched_df = matched_df
            evidence_text = matched_df.to_string(index=False)

            # 3. DETERMINISTIC CLINICAL AUDITOR PROMPT
            SYSTEM_PROMPT = """
You are a Clinical Data Auditor. Your ONLY purpose is to extract statistics from the provided DATA.

STRICT PROTOCOL:
1. DATA LOCK: Use ONLY the provided DATA.
2. NO ECHOING: Do NOT repeat the user's patient description or hypothetical claims.
3. MANDATORY STATS: You must populate the following columns for every match found:
   - Predictor Variables (Thresholds/Scores)
   - Outcome Definitions (Timeframes)
   - Effect Sizes (Exact AUC, OR, or HR values)
   - Cohort Description (Sample size and Type)
   - Source Anchor (Specific Page or Section)

4. SAFETY: If the query mentions a protocol (e.g. Blueberry, HOBI) NOT in the DATA, you MUST return only: "CLAIM UNSUPPORTED: No clinical evidence found in the repository."

## EVIDENCE TABLE
| Study | Predictor | Outcome | Effect Size (AUC/OR) | Cohort | Source Anchor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |

## AUDIT VERDICT
- **Risk Level:** [High/Low]
- **Clinical Match:** [Summary of match based on the statistical data provided].
"""

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=0, # No creativity allowed
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"PATIENT QUERY: {patient_desc}\n\nDATA:\n{evidence_text}"}
                    ]
                )
                st.session_state.report = response.choices[0].message.content
                st.rerun()
            except Exception as e:
                st.error(f"API ERROR: {e}")

    # OUTPUT DISPLAY
    if st.session_state.report:
        st.markdown(st.session_state.report)

        if st.session_state.matched_df is not None and not st.session_state.matched_df.empty:
            st.divider()
            st.subheader("📄 Source Evidence PDFs")
            cols = st.columns(3)
            for i, (_, row) in enumerate(st.session_state.matched_df.iterrows()):
                filename = row.get("source_file") or f"{row.get('study_name','study')}.pdf"
                full_path = os.path.join(PDF_FOLDER, str(filename))
                with cols[i % 3]:
                    st.caption(row.get("study_name", "Study"))
                    if os.path.exists(full_path):
                        if st.button(f"📂 Open PDF {i+1}", key=f"pdf_{i}"):
                            open_pdf_manual(full_path)

st.caption("Research-use only. Deterministic evidence synthesis.")
