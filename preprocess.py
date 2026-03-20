from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data/Raw Loans.txt", "r",encoding="latin-1") as f:
    text = f.read()

chunks = text.split("======")  # using your separators
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "faiss_index.index")

np.save("chunks.npy", np.array(chunks))
