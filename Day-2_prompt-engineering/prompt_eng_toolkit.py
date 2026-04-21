from groq import Groq
import os 
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("groq_api_key"))

def llm(prompt, system="You are a helpful assistant", temperature=0.7):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

# ---- TECHNIQUE 1: Zero-shot vs Few-shot comparison ----
zero_shot = "Classify sentiment: 'The delivery was late but product is good'"

few_shot = """
Classify sentiment:
"Absolutely love it!" → Positive  
"Complete garbage" → Negative
"It does the job" → Neutral
"Came damaged but support helped" → Mixed

Classify: "The delivery was late but product is good" →
"""

print("Zero-shot:", llm(zero_shot))
print("Few-shot:", llm(few_shot))

# ---- TECHNIQUE 2: Chain of Thought ----
problem = """
An AI startup has 3 engineers. Each engineer can build 2 features per week.
The product needs 18 features to launch. They have 4 weeks.
Think step by step. Can they launch on time? What's the risk?
"""
print("\nCoT Answer:", llm(problem))

# ---- TECHNIQUE 3: Structured JSON output ----
import json

def analyze_startup_idea(idea):
    prompt = f"""
    Idea: {idea}
    """

    system = """You are a JSON generator.
    You MUST return ONLY valid JSON.
    No explanations, no markdown, no extra text."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    result = response.choices[0].message.content
    return json.loads(result)

analysis = analyze_startup_idea("An AI tool that writes personalized cold emails for sales teams")
print("\nStartup Analysis:")
print(json.dumps(analysis, indent=2))