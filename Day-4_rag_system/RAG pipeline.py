from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize everything
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_knowledge_base")
groq_client = Groq(api_key=os.getenv("groq_api_key"))

# ── PHASE 1: INDEX YOUR DOCUMENTS ──────────────────────────
# Simulating a company knowledge base
documents = [
    "Our refund policy allows returns within 30 days of purchase with original receipt.",
    "Customer support is available Monday to Saturday, 9 AM to 6 PM IST.",
    "We offer three pricing plans: Basic at ₹999/month, Pro at ₹2499/month, Enterprise at ₹7999/month.",
    "The Basic plan includes 5 users and 10GB storage.",
    "The Pro plan includes 25 users, 100GB storage, and priority support.",
    "Enterprise plan includes unlimited users, 1TB storage, dedicated account manager.",
    "To reset your password, go to Settings > Security > Reset Password.",
    "Our mobile app is available on both iOS and Android platforms.",
    "We use 256-bit SSL encryption to protect all customer data.",
    "Annual subscriptions receive a 20% discount compared to monthly billing.",
]

# Convert documents to embeddings and store
print("Indexing documents...")
embeddings = embedding_model.encode(documents).tolist()
ids = [f"doc_{i}" for i in range(len(documents))]

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)
print(f"{len(documents)} documents indexed\n")

# ── PHASE 2: RETRIEVAL + GENERATION ────────────────────────
def rag_answer(question):
    # Step 1: Convert question to embedding
    question_embedding = embedding_model.encode([question]).tolist()
    
    # Step 2: Find most similar documents
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3  # retrieve top 3 most relevant chunks
    )
    
    # Step 3: Extract retrieved context
    retrieved_chunks = results['documents'][0]
    context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])
    
    # Step 4: Build prompt with context injected
    prompt = f"""You are a helpful customer support assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    # Step 5: LLM generates answer from context
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # precise answers for support bot
    )
    
    answer = response.choices[0].message.content
    
    # Show what was retrieved (great for understanding/debugging)
    print(f"\nQuestion: {question}")
    print(f"\nRetrieved chunks:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f" {i}. {chunk[:80]}...")
    print(f"\n Answer: {answer}")
    print(f"\n{'─'*55}")
    
    return answer

# Test your RAG system
questions = [
    "What is the refund policy?",
    "How much does the Pro plan cost?",
    "When is customer support available?",
    "Is there a discount for annual plans?",
    "What programming languages do you support?"  # not in docs
]

for q in questions:
    rag_answer(q)