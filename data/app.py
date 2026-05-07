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
FILE = "Sepsis_Evidence_Table.csv"
PDF_FOLDER = "pdf" 

# --- SIDEBAR: MANUAL PDF VIEWER ---
with st.sidebar:
    st.header("📚 Evidence Library")
    if os.path.exists(FILE):
        df = pd.read_csv(FILE)
        
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
    st.subheader("📊 Grounded Evidence Report")
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
                        "content": (
                            "You are a Clinical Auditor. You must use the EXACT study names provided in the DATA.\n\n"
                            "STRICT RULES:\n"
                            "1. Never use shorthand like 'Sepsis-3 study' or 'Pediatric study'.\n"
                            "2. Always write the FULL study name exactly as it appears in the DATA.\n"
                            "3. Immediately follow every study name with its filename in parentheses from the data, e.g., 'Full Study Name (filename.pdf)'.\n\n"
                            "OUTPUT HEADERS:\n"
                            "### 1. CONFIDENCE LEVEL\n\n"
                            "### 2. GAP ANALYSIS\n\n"
                            "### 3. CLINICAL MATCHES\n\n"
                            "### 4. STUDY DETAILS\n\n"
                            "### 5. SCIENTIFIC SUMMARY\n\n"
                            "### 6. ESTIMATED MORTALITY RISK & CORRELATION"
                        )
                    },
                    {"role": "user", "content": f"DATA:\n{top_3_evidence}\n\nPATIENT:\n{patient_desc}"}
                ],
                temperature=0 # Ensures the AI sticks to the exact text provided
            )
            st.markdown(response.choices[0].message.content)
            st.success("🏁 Audit Complete.")
    else:
        st.info("Awaiting patient data...")

st.caption("Disclaimer: This tool is for research/demo purposes. Verify with a physician.")