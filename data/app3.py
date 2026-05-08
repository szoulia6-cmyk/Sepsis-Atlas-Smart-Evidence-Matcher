import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import subprocess
import platform

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sepsis Atlas Dashboard", page_icon="🏥", layout="wide")

# --- AUTH & SETUP ---
API_KEY = "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"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
FILE = "Sepsis_Evidence_Table_3.csv"
PDF_FOLDER = "pdf/articles" 

# --- SIDEBAR: MANUAL PDF VIEWER ---
with st.sidebar:
    st.header("📚 Evidence Library")
    if os.path.exists(FILE):
        df = pd.read_csv(FILE, sep=None, engine="python", on_bad_lines="skip")
        
        # Identify core columns dynamically
        study_col = next((c for c in df.columns if 'study' in c.lower() or 'name' in c.lower()), df.columns[0])
        file_col = next((c for c in df.columns if 'file' in c.lower() or 'source' in c.lower()), None)
        
        st.subheader("1. Select Study")
        selected_study = st.selectbox("Choose a study from data:", df[study_col].unique())
        
        st.subheader("2. View Original Paper")
        if os.path.exists(PDF_FOLDER):
            available_pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
            filename = st.selectbox("Select PDF to open:", ["Select a file..."] + available_pdfs)
            
            file_path = os.path.join(PDF_FOLDER, filename)
            if filename != "Select a file..." and os.path.exists(file_path):
                if st.button("📂 Open Local PDF", use_container_width=True):
                    try:
                        curr_os = platform.system()
                        if curr_os == "Darwin": subprocess.call(('open', file_path))
                        elif curr_os == "Windows": os.startfile(file_path)
                        else: subprocess.call(('xdg-open', file_path))
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.error("Folder 'pdf' not found.")
        st.divider()
        st.write(f"Total Database Rows: {len(df)}")
    else:
        st.error("CSV file not found.")

# --- MAIN INTERFACE ---
st.title("🏥 Sepsis Atlas: Smart Evidence Matcher")
st.markdown("---")

col_in, col_out = st.columns([1, 1.5], gap="large")

with col_in:
    st.subheader("📋 Patient Clinical Profile")
    patient_desc = st.text_area("Enter clinical details:", height=300)
    run_btn = st.button("🔍 Run Evidence Audit", use_container_width=True, type="primary")

with col_out:
    
    if run_btn and patient_desc:
        with st.spinner("Analyzing top evidence..."):
            # Build existing columns list safely
            potential_cols = [study_col, 'predictor', 'outcome', 'effect_size', 'performance']
            if file_col: potential_cols.append(file_col)
            
            existing_cols = [c for c in potential_cols if c in df.columns]
            
            # Send top 3 rows to AI
            top_3_evidence = df[existing_cols].head(3).to_string(index=False)
            
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {
                        "role": "system",
                        "content" : (
                            "You are a Clinical Evidence Synthesizer specialized in biomarker selection for sepsis mortality risk stratification.\n\n"

                            "TASK:\n"
                            "From the provided DATA, extract and compare predictors of mortality (preferably 28-day mortality when available).\n\n"

                            "STRICT RULES:\n"
                            "1. Only use information explicitly present in the DATA.\n"
                            "2. Do NOT infer or hallucinate missing values.\n"
                            "3. Preserve exact wording for study names and predictors.\n"
                            "4. If a field is missing → write \"Not reported\".\n"
                            "5. Each predictor must be a separate row (even if from the same study).\n\n"

                            "METRIC HANDLING:\n"
                            "- Extract ALL reported effect sizes and performance metrics.\n"
                            "- Identify the BEST metric per predictor using this priority:\n"
                            "  1. AUC / AUROC / C-index / C-statistic\n"
                            "  2. Hazard Ratio (HR)\n"
                            "  3. Odds Ratio (OR)\n"
                            "  4. Relative Risk (RR)\n"
                            "  5. Accuracy / Sensitivity / Specificity\n"
                            "- Store this separately for ranking.\n\n"

                            "YOU MUST EXTRACT:\n"
                            "- Study name\n"
                            "- Population\n"
                            "- Sample size\n"
                            "- Predictor\n"
                            "- Outcome (e.g., 28-day mortality, ICU mortality)\n"
                            "- Model (e.g., logistic regression, Cox regression)\n"
                            "- Effect size\n"
                            "- Performance (AUC, sensitivity, etc.)\n"
                            "- Adjustment (variables adjusted for, if reported)\n"
                            "- Relevance to target population (High / Moderate / Low based on similarity to ICU adult sepsis)\n"
                            "- Source (filename if available)\n\n"

                            "OUTPUT FORMAT:\n\n"

                            "### 📊 STUDY-LEVEL EVIDENCE TABLE\n"
                            "| Study | Population | Sample Size | Predictor | Outcome | Model | Effect Size | Performance | Adjustment | Relevance to Target Population | Source |\n\n"

                            "### 🏆 RANKED PREDICTORS\n"
                            "| Predictor | Best Metric | Value | Study Notes |\n\n"

                            "RANKING RULES:\n"
                            "- Rank predictors primarily by BEST metric (AUC/C-index highest first).\n"
                            "- If AUC not available, use HR/OR magnitude.\n"
                            "- Consider consistency across studies.\n"
                            "- Prefer predictors validated in ICU sepsis populations.\n\n"

                            "### 🔍 KEY INSIGHTS\n"
                            "- Which predictors are strongest overall?\n"
                            "- Which are most consistent across studies?\n"
                            "- Differences between biomarkers vs clinical scores\n\n"

                            "### ⚠️ LIMITATIONS\n"
                            "- Missing data\n"
                            "- Heterogeneous populations\n"
                            "- Differences in study design\n"
                            )
                    },
                    {
                        "role": "user",
                        "content": f"DATA:\n{top_3_evidence}\n\nPATIENT:\n{patient_desc}"
                    }
                ],
                temperature=0
            )
            st.markdown(response.choices[0].message.content)
            st.success("🏁 Audit Complete.")
    else:
        st.info("Awaiting patient data...")

st.caption("Disclaimer: This tool is for research/demo purposes. Verify with a physician.")
