from groq import Groq
import json 
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("groq_api_key"))

def llm(prompt, system="You are a helpful assistant", temperature=0.7):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

def compare_prompts(task, prompts_dict):
    """
    Compare multiple prompting strategies on the same task.
    """
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}")
    
    for strategy_name, prompt in prompts_dict.items():
        print(f"\n--- {strategy_name} ---")
        print(llm(prompt, temperature=0.3))
        print()

# Same task, 4 different strategies
task = "Explain what AI agents are"

strategies = {
    "Basic (no engineering)": 
        "What are AI agents?",
    
    "Role + Constraint": 
        """You are a senior AI engineer explaining to a smart 
        16-year-old. Use one real analogy. Max 80 words.""",
    
    "Few-shot structured": 
        """Explain a tech concept using this format:
        Concept: APIs
        One-liner: Waiters that carry requests between apps and kitchens
        Real example: When you pay via Razorpay, an API talks to your bank
        
        Now explain: AI Agents""",
    
    "Chain of thought + Role": 
        """You are building a course on AI for beginners. 
        First identify what the student probably already knows.
        Then build on that.
        Then give one practical example from daily life.
        Explain: AI Agents"""
}

compare_prompts(task, strategies)