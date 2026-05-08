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

# --- SIDEBAR ---
with st.sidebar:
    st.header("📚 Evidence Library")
    if os.path.exists(FILE):
        # LOAD AND CLEAN COLUMNS IMMEDIATELY
        df = pd.read_csv(FILE)
        # This removes spaces and makes everything lowercase (e.g., 'Source File' -> 'source_file')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Identify the critical columns after cleaning
        file_col = 'source_file' if 'source_file' in df.columns else (next((c for c in df.columns if 'file' in c), None))
        study_col = 'study_name' if 'study_name' in df.columns else df.columns[0]
        
        st.subheader("1. Select Study")
        selected_study = st.selectbox("Choose a study:", df[study_col].unique())
        study_info = df[df[study_col] == selected_study].iloc[0]
        
        st.subheader("2. Link to Source")
        filename = str(study_info.get(file_col, '')).strip() if file_col else ""
        
        if os.path.exists(PDF_FOLDER):
            available_pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
            if not filename or filename not in available_pdfs or filename == "nan":
                st.warning("🔗 Auto-link failed. Select manually:")
                filename = st.selectbox("Select PDF:", ["Select a file..."] + available_pdfs)
            
            file_path = os.path.join(PDF_FOLDER, filename)
            if filename != "Select a file..." and os.path.exists(file_path):
                st.success(f"📄 Linked: {filename}")
                if st.button("📂 Open Local PDF", use_container_width=True):
                    try:
                        curr_os = platform.system()
                        if curr_os == "Darwin": subprocess.call(('open', file_path))
                        elif curr_os == "Windows": os.startfile(file_path)
                        else: subprocess.call(('xdg-open', file_path))
                    except Exception as e:
                        st.error(f"Error: {e}")
        st.divider()
        st.write(f"Total Rows: {len(df)}")
    else:
        st.error("CSV file not found.")

# --- MAIN INTERFACE ---
st.title("🏥 Sepsis Atlas: Smart Evidence Matcher")
st.markdown("---")

col_in, col_out = st.columns([1, 1.5], gap="large")

with col_in:
    st.subheader("📋 Patient Profile")
    patient_desc = st.text_area("Enter clinical details:", height=200)
    run_btn = st.button("🔍 Run Evidence Audit", use_container_width=True, type="primary")

with col_out:
    st.subheader("📊 Grounded Evidence Report")
    if run_btn and patient_desc:
        with st.spinner("Processing..."):
            # BUILD SAFE COLUMN LIST
            # We only pick columns that we KNOW are in the cleaned dataframe
            potential = [study_col, 'predictor', 'outcome', 'effect_size', 'performance', file_col]
            existing_cols = [c for c in potential if c and c in df.columns]
            
            # CRASH-FREE SELECTOR
            top_3_evidence = df[existing_cols].head(3).to_string(index=False)
            
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {"role": "system", "content": "You are a Clinical Auditor. Use provided data. Include (filename) in study mentions. Format with 6 clear headers: 1.Confidence, 2.Gaps, 3.Matches, 4.Details, 5.Summary, 6.Mortality/Correlation."},
                    {"role": "user", "content": f"DATA:\n{top_3_evidence}\n\nPATIENT:\n{patient_desc}"}
                ]
            )
            st.markdown(response.choices[0].message.content)
            st.success("🏁 Audit Complete.")
    else:
        st.info("Awaiting patient data...")
