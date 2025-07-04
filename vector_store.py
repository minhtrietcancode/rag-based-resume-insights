import faiss
import pickle
import os
import numpy as np
from typing import List
from embed_chunk import embed_chunk

INDEX_FILE = "vector_store/resume_index.faiss"
META_FILE = "vector_store/resume_metadata.pkl"
DIM = 384  # Dimension of embeddings for all-MiniLM-L6-v2

# Create or load FAISS index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(DIM)

# Load metadata if available
if os.path.exists(META_FILE):
    with open(META_FILE, 'rb') as f:
        metadata = pickle.load(f)
else:
    metadata = []

def add_chunks(vectors: List[List[float]], chunks: List[str]):
    global index, metadata
    arr = np.array(vectors).astype('float32')
    index.add(arr)
    metadata.extend(chunks)

def save_index():
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, 'wb') as f:
        pickle.dump(metadata, f)

def search_top_k(query_vector: List[float], k: int = 3):
    arr = np.array([query_vector]).astype('float32')
    distances, indices = index.search(arr, k)
    results = [(metadata[i], distances[0][j]) for j, i in enumerate(indices[0]) if i < len(metadata)]
    return results

# Just a small example for testing
if __name__ == '__main__':
    # Add new chunks and save index
    chunks = [
        "EDUCATION\nSoutheastern Louisiana University (SLU)...",
        "EXPERIENCE\nWorked as marketing intern at XYZ Corp...",
        "SKILLS\nPython, SQL, Excel, Tableau",
        "I do not have any skill"
    ]
    vectors = [embed_chunk(c) for c in chunks]
    add_chunks(vectors, chunks)
    save_index()

    # Perform a search
    query = embed_chunk("What are the candidate's skills?")
    results = search_top_k(query)

    for chunk, score in results:
        print("\nScore:", score)
        print(chunk)
