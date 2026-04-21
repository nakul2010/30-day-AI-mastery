from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import json
import re
import sys

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

client = Groq(api_key=os.getenv("groq_api_key"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

def llm(prompt, temperature=0.7, system="You are a helpful AI assistant."):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

# ════════════════════════════════════════════
# FEATURE 1: Smart Content Generator
# ════════════════════════════════════════════
def generate_content(topic, platform, audience):
    formats = {
        "blog": "800-word blog post with introduction, 3 sections, conclusion",
        "linkedin": "150-word professional post with strong hook",
        "twitter": "3 tweet variations under 280 characters each",
        "email": "email with subject line, 100-word body, CTA"
    }
    
    result = llm(f"""
    Topic: {topic}
    Platform format: {formats.get(platform, platform)}
    Audience: {audience}
    
    Write the content directly. No meta-commentary.
    """, temperature=0.7)
    
    return result

# ════════════════════════════════════════════
# FEATURE 2: Document Q&A (RAG)
# ════════════════════════════════════════════
def setup_knowledge_base(documents, collection_name="portfolio_kb"):
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    
    collection = chroma_client.create_collection(collection_name)
    embeddings = embedding_model.encode(documents).tolist()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    return collection

def ask_knowledge_base(collection, question):
    q_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=3)
    context = "\n".join(results['documents'][0])
    
    answer = llm(f"""Answer using ONLY this context:
{context}

Question: {question}
If answer not in context, say "Not found in knowledge base."
""", temperature=0.0)
    return answer

# ════════════════════════════════════════════
# FEATURE 3: Startup Idea Analyzer
# ════════════════════════════════════════════
def analyze_idea(idea):
    result = llm(f"""
    Analyze this startup idea and return ONLY valid JSON:
    Idea: {idea}
    
    {{
        "verdict": "promising/risky/avoid",
        "market_size": "large/medium/small",
        "competition": "high/medium/low",
        "moat": "what makes this defensible",
        "top_3_risks": ["risk1", "risk2", "risk3"],
        "first_step": "most important thing to do this week",
        "score": "1-10"
    }}
    """, temperature=0.0)

    # LLMs sometimes wrap JSON in extra text or markdown fences.
    cleaned = result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # Safe fallback so the app does not crash during demo runs.
    return {
        "verdict": "risky",
        "market_size": "unknown",
        "competition": "unknown",
        "moat": "Could not parse model output.",
        "top_3_risks": [
            "Invalid JSON response from model",
            "Output formatting drift",
            "Need stricter output constraints"
        ],
        "first_step": "Retry analysis with stricter prompt or model JSON mode.",
        "score": "N/A"
    }

# ════════════════════════════════════════════
# FEATURE 4: Multi-Persona Chat
# ════════════════════════════════════════════
personas = {
    "mentor": "You are a senior AI engineer with 10 years experience. Give practical, direct advice.",
    "critic": "You are a devil's advocate. Challenge every idea, find weaknesses, push back hard.",
    "investor": "You are a VC investor. Evaluate everything through ROI and market size lens.",
    "coach": "You are a motivational coach. Be encouraging, action-oriented, and energetic."
}

def chat_with_persona(message, persona_name):
    system = personas.get(persona_name, personas["mentor"])
    return llm(message, temperature=0.7, system=system)

# ════════════════════════════════════════════
# MAIN DEMO — Run All Features
# ════════════════════════════════════════════
def main():
    print("\n" + "="*55)
    print("🤖 AI CONTENT & ANALYSIS SUITE")
    print("="*55)
    
    # Demo 1: Content Generation
    print("\n📝 FEATURE 1: Content Generator")
    print("─"*40)
    content = generate_content(
        topic="AI tools for college students",
        platform="linkedin",
        audience="Indian students and young professionals"
    )
    print(content)
    
    # Demo 2: RAG Knowledge Base
    print("\n\n📚 FEATURE 2: Document Q&A")
    print("─"*40)
    docs = [
        "This AI Suite was built by Nakul as part of a 30-day AI mastery program.",
        "The suite uses Groq's Llama 3.3 70B model for all language tasks.",
        "RAG feature uses ChromaDB for vector storage and SentenceTransformers for embeddings.",
        "The tool supports content generation for blog, LinkedIn, Twitter and email formats.",
        "Built in Python using LangChain concepts, Groq API, and ChromaDB."
    ]
    kb = setup_knowledge_base(docs)
    answer = ask_knowledge_base(kb, "What model does this suite use?")
    print(f"Q: What model does this suite use?\nA: {answer}")
    
    # Demo 3: Startup Analyzer
    print("\n\n💡 FEATURE 3: Startup Idea Analyzer")
    print("─"*40)
    analysis = analyze_idea(
        "An app that helps Indian street food vendors manage orders and accept UPI payments using AI"
    )
    print(json.dumps(analysis, indent=2))
    
    # Demo 4: Multi-Persona Chat
    print("\n\n🎭 FEATURE 4: Multi-Persona Responses")
    print("─"*40)
    question = "Should I drop college to focus on building AI projects?"
    for persona in ["mentor", "critic", "investor"]:
        print(f"\n[{persona.upper()}]")
        print(chat_with_persona(question, persona))

if __name__ == "__main__":
    main()