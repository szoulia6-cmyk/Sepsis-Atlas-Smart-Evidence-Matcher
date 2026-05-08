import os
import json
import pandas as pd
from openai import OpenAI
import PyPDF2

# =====================================================
# CONFIG & AUTH
# =====================================================
API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

PDF_FOLDER = "pdf"
OUTPUT_CSV = "Sepsis_Evidence_Table_3.csv"

# =====================================================
# HELPERS
# =====================================================
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages[:5]: # Take first 5 pages
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"❌ Error reading PDF {pdf_path}: {e}")
        return ""

def analyze_paper(filename, text):
    print(f"🔬 AI Analyzing: {filename}...")
    
    system_message = (
        "You are a clinical research data extraction system specialized in sepsis mortality studies.\n\n"

        "TASK:\n"
        "Extract structured study data for biomarker and risk stratification analysis.\n\n"

        "STRICT RULES:\n"
        "1. Only extract information explicitly present in the text.\n"
        "2. Do NOT infer or hallucinate missing values.\n"
        "3. If a field is not found, write \"Not reported\".\n"
        "4. Preserve exact wording for study names and predictors.\n"
        "5. Return ONLY a valid JSON object (no explanations).\n\n"

        "YOU MUST EXTRACT THESE FIELDS:\n"
        "- study_name\n"
        "- population\n"
        "- sample_size\n"
        "- predictor (main biomarker or clinical score studied)\n"
        "- outcome (e.g., 28-day mortality, ICU mortality)\n"
        "- model (e.g., logistic regression, Cox regression, machine learning model)\n"
        "- effect_size (e.g., OR, HR with values)\n"
        "- performance (e.g., AUC, sensitivity, specificity)\n"
        "- adjustment (variables adjusted for in multivariable analysis)\n"
        "- validation_method (e.g., ROC analysis, cross-validation, external validation)\n"
        "- study_design (e.g., retrospective cohort, prospective study)\n"
        "- source_anchor (must include page number and section if available, e.g., \"Page 3 - Results section\")\n\n"

        "SOURCE ANCHOR RULES:\n"
        "- Always include page number if visible in text.\n"
        "- Include section name if identifiable (Abstract, Methods, Results).\n"
        "- If exact page not visible, approximate location (e.g., \"early section\", \"results paragraph\").\n\n"

        "OUTPUT FORMAT (STRICT JSON):\n"
        "{\n"
        "  \"study_name\": \"...\",\n"
        "  \"population\": \"...\",\n"
        "  \"sample_size\": \"...\",\n"
        "  \"predictor\": \"...\",\n"
        "  \"outcome\": \"...\",\n"
        "  \"model\": \"...\",\n"
        "  \"effect_size\": \"...\",\n"
        "  \"performance\": \"...\",\n"
        "  \"adjustment\": \"...\",\n"
        "  \"validation_method\": \"...\",\n"
        "  \"study_design\": \"...\",\n"
        "  \"source_anchor\": \"...\"\n"
        "}"
        )

    user_message = f"Document: {filename}\n\nContent: {text[:15000]}"

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={ "type": "json_object" } 
        )
        
        data = json.loads(response.choices[0].message.content)
        
        # --- THE INTEGRATED FIX ---
        # 1. Handle if AI returns a list [ {data} ]
        if isinstance(data, list):
            data = data[0]
        
        # 2. Handle if AI nests it inside a key like {"study": {data}}
        if isinstance(data, dict) and len(data) == 1:
            val = list(data.values())[0]
            if isinstance(val, dict):
                data = val

        # 3. Attach filename for the dashboard links
        data['source_file'] = filename
        return data

    except Exception as e:
        print(f"❌ AI Error on {filename}: {e}")
        return None

# =====================================================
# MAIN EXECUTION
# =====================================================
def run_analysis():
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ '{PDF_FOLDER}' folder not found.")
        return

    all_results = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

    for pdf in pdf_files:
        raw_text = extract_text_from_pdf(os.path.join(PDF_FOLDER, pdf))
        if raw_text.strip():
            result = analyze_paper(pdf, raw_text)
            if result:
                all_results.append(result)
        else:
            print(f"⚠️ {pdf} is empty or unreadable.")

    if all_results:
        df = pd.DataFrame(all_results)
        # Use quoting=1 to prevent commas in clinical text from breaking columns
        df.to_csv(OUTPUT_CSV, index=False, quoting=1)
        print(f"\n✅ SUCCESS: {len(all_results)} studies extracted to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_analysis()
