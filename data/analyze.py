import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def analyze_paper(filename, text):
    print(f"🔬 Extracting Evidence: {filename}...")
    
    # SYSTEM PROMPT: Forces the clinical structure
    system_message = (
        "You are a clinical data extractor. Your goal is to support 'Counterfactual Mortality Estimation'. "
        "Extract data into a JSON object with these keys: "
        "study_name, population, sample_size, predictor, outcome, timing, method, effect_size, performance, source_anchor, notes. "
        "If a value is not mentioned, write 'not reported'. Do not hallucinate."
    )

    user_message = f"Document: {filename}\n\nContent: {text[:15000]}"

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={ "type": "json_object" } # Ensures we get a table-ready format
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error on {filename}: {e}")
        return None