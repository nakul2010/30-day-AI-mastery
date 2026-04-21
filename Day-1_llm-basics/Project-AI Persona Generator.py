from groq import Groq
import json
import os
from dotenv import load_dotenv
from matplotlib import text

load_dotenv()

client = Groq(api_key=os.getenv("groq_api_key"))

def generate_persona(industry, use_case, tone):
    prompt = f"""
Create an AI assistant persona for a {industry} company.

Use case: {use_case}
Tone: {tone}

Return ONLY valid JSON.
Do not add explanations, markdown, or extra text.

Format:
{{
  "name": "...",
  "role": "...",
  "system_prompt": "...",
  "example_questions": ["...", "...", "...", "...", "..."],
  "example_responses": ["...", "...", "...", "...", "..."]
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    
    def extract_json(text):
        try:
            return json.loads(text)
        except:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start:end])
                except:
                    return None
            return None
        
    content = response.choices[0].message.content
    return extract_json(content)


examples = [
    ("fintech startup", "onboarding new users", "friendly but professional"),
    ("law firm", "answering client questions", "formal and precise"),
    ("fitness app", "motivating users to work out", "energetic and encouraging"),
]

all_personas = []

for industry, use_case, tone in examples:
    persona = generate_persona(industry, use_case, tone)

    if persona:
        all_personas.append(persona)

        print(f"\n=== {persona['name']} ===")
        print(f"Role: {persona['role']}")
        print(f"System Prompt Preview: {persona['system_prompt'][:100]}...")
    else:
        print("Skipped invalid JSON response")

with open("personas.json", "w", encoding="utf-8") as f:
    json.dump(all_personas, f, indent=4, ensure_ascii=False)

print("\nPersonas saved to personas.json")