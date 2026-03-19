import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# Initialize client
client = genai.Client(api_key="AIzaSyBreO3TNW6vhcwlUUTAUNr74F38XgD-1tY")
# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load chunks
chunks = np.load("chunks.npy", allow_pickle=True)


# Retrieve relevant chunks
def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Generate answer
def generate_answer(query):
    context = retrieve(query)
    context_text = "\n\n".join(context)

    prompt = f"""
You are an AI assistant for Bank of Maharashtra loan products.

Rules:
- Use ONLY the context
- Do not add extra information
- If not found, say "Information not available"
- Answer in structured format with headings
- Do not infer or assume anything beyond context.
Context:
{context_text}

Question:
{query}

Answer clearly:
"""

    response = client.models.generate_content(
       model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
# Run loop
while True:
    query = input("\nAsk your question (or type 'exit'): ")
    
    if query.lower() == "exit":
        break

    answer, context = generate_answer(query)

    print("\nRelevant Context:")
    for i, c in enumerate(context):
        print(f"\nChunk {i+1}:\n{c}")

    print("\nAnswer:", answer)
