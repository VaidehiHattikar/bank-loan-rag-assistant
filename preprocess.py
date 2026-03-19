from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your data
with open("data/Raw Loans.txt", "r",encoding="latin-1") as f:
    text = f.read()

# Step 1: Split into chunks
chunks = text.split("======")  # using your separators
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

# Step 2: Generate embeddings
embeddings = model.encode(chunks)

# Step 3: Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index + chunks
faiss.write_index(index, "faiss_index.index")

np.save("chunks.npy", np.array(chunks))