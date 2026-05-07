import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Setup & Authentication
load_dotenv()

API_KEY = "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" 

client = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=API_KEY
)

FILE = "Sepsis_Evidence_Table.csv"

def patient_consult(patient_description):
    if not os.path.exists(FILE):
        print(f"❌ Error: {FILE} not found.")
        return

    try:
        df = pd.read_csv(FILE)
        evidence_context = df.to_string()

        # 3. Loosened Logic: Relevance & Gap Analysis
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a Clinical Evidence Matcher. Your ONLY source of truth is the provided CSV DATA.\n\n"
                        "STRICT RULES:\n"
                        "1. RELEVANCE RANKING: If a 100% match isn't found, identify the CLOSEST study based on clinical keywords (infection site, scores, metrics).\n"
                        "2. MATCH CONFIDENCE: Label the match as HIGH, MEDIUM, or LOW based on how closely the patient aligns with the study population.\n"
                        "3. GAP ANALYSIS: Explicitly list 'Gaps/Inaccuracies' (e.g., age difference, different ICU setting, missing metrics).\n"
                        "4. NO GUESSED DATA: Use exact numbers from the 'effect_size' or 'performance' columns. If a metric is missing, write 'Metric Not Reported'.\n"
                        "5. EXTREME OUTLIERS: Only if the query is totally unrelated to sepsis or the table (e.g., 'broken arm') should you say 'NO RELEVANT EVIDENCE'.\n\n"
                        "Output Format:\n"
                        "### 🏆 Closest Study: [Study Name]\n"
                        "**Confidence:** [Level] - [Reasoning]\n"
                        "**Evidence Found:** [Stats/Mortality/Performance]\n"
                        "**Gaps/Inaccuracies:** [List why this study might not perfectly apply to this patient]"
                    )
                },
                {
                    "role": "user", 
                    "content": f"CSV DATA:\n{evidence_context}\n\nTARGET PATIENT:\n{patient_description}"
                }
            ],
            temperature=0 
        )
        
        output = response.choices[0].message.content
        print("\n" + "="*60)
        print("🔍 SEPSIS ATLAS: RELEVANCE REPORT")
        print("="*60)
        print(output)
        print("="*60)

    except Exception as e:
        print(f"\n❌ API Error: {e}")

if __name__ == "__main__":
    print("\n" + "*"*40)
    print("--- Sepsis Atlas: Smart Match Mode ---")
    print("Type 'exit' to close.")
    print("*"*40)
    
    while True:
        patient_info = input("\nDescribe Patient: ")
        if patient_info.lower() == 'exit': break
        if not patient_info.strip(): continue
        patient_consult(patient_info)