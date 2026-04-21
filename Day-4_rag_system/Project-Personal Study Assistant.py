from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("study_notes")
groq_client = Groq(api_key=os.getenv("groq_api_key"))

# Your actual study notes — add your own content here
study_notes = [
    "Transformers use self-attention to process all tokens simultaneously unlike RNNs.",
    "BERT is an encoder-only transformer trained on masked language modeling.",
    "GPT is a decoder-only transformer trained on next token prediction.",
    "Fine-tuning means taking a pre-trained model and training it further on specific data.",
    "RAG combines retrieval systems with generative models to add external knowledge.",
    "LangChain is a framework that chains LLM calls and tools into pipelines.",
    "Temperature controls randomness in LLM outputs. 0 = deterministic, 1+ = creative.",
    "Embeddings convert text to vectors where similar meanings are geometrically close.",
    "ChromaDB is a local vector database that stores and searches embeddings.",
    "Prompt engineering is the practice of crafting inputs to get desired LLM outputs.",
    "Few-shot prompting gives examples in the prompt to guide model behavior.",
    "Chain of thought prompting makes models reason step by step before answering.",
]

# Index notes
embeddings = embedding_model.encode(study_notes).tolist()
collection.add(
    documents=study_notes,
    embeddings=embeddings,
    ids=[f"note_{i}" for i in range(len(study_notes))]
)

def study_assistant(question):
    q_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=3)
    context = "\n".join(results['documents'][0])
    
    prompt = f"""You are a study assistant helping a student learn AI concepts.
Use the notes below to answer clearly. Use simple language and analogies.

NOTES:
{context}

QUESTION: {question}"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    print(f"\n {question}")
    print(f" {response.choices[0].message.content}\n")

# Add your own study notes and test it
study_assistant("What is the difference between BERT and GPT?")
study_assistant("How does RAG work?")
study_assistant("When should I use low temperature?")