from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("groq_api_key"))

def test(prompt, temperature=0.7, label=""):
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}],
        temperature=temperature
    )
    result = response.choices[0].message.content
    print(f"\n[{label}]\n{result}\n{'-'*50}")
    return result

test("What are AI agents?", label="Basic")
test("You are a senior engineer. Explain AI agents to a beginner in 3 bullet points.", label="Role + Constraint")
test("Explain AI agents. Think step by step.", label="Chain of Thought")