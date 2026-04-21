from sentence_transformers import SentenceTransformer
import numpy as np

# Free embedding model — runs locally
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text to embeddings
texts = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Pizza is a popular Italian food",
    "Dogs are loyal animals",
    "Neural networks are inspired by the brain"
]

embeddings = model.encode(texts)

print(f"Each text becomes a vector of {len(embeddings[0])} numbers")
print(f"Embedding for first text (first 5 numbers): {embeddings[0][:5]}")

# Calculate similarity between texts
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare: ML text vs Deep Learning text (should be HIGH similarity)
sim1 = cosine_similarity(embeddings[0], embeddings[1])
# Compare: ML text vs Pizza text (should be LOW similarity)  
sim2 = cosine_similarity(embeddings[0], embeddings[2])

print(f"\nSimilarity: 'ML' vs 'Deep Learning': {sim1:.3f}")
print(f"Similarity: 'ML' vs 'Pizza': {sim2:.3f}")
print("\nHigher number = more similar meaning")