import faiss
import pickle
import os
import numpy as np
from typing import List
from embed_chunk import embed_chunk

# Define file paths for storing the FAISS index and associated metadata.
INDEX_FILE = "vector_store/resume_index.faiss"
META_FILE = "vector_store/resume_metadata.pkl"
# Define the dimension of the embeddings. This must match the output dimension of the SentenceTransformer model.
DIM = 384  # Dimension of embeddings for all-MiniLM-L6-v2

# Initialize the FAISS index. If an index file already exists, load it; otherwise, create a new flat L2 index.
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    # Initialize a new FAISS index for L2 (Euclidean) distance similarity search.
    index = faiss.IndexFlatL2(DIM)

# Load existing metadata (chunks of text) if the metadata file exists; otherwise, initialize an empty list.
if os.path.exists(META_FILE):
    with open(META_FILE, 'rb') as f:
        metadata = pickle.load(f)
else:
    metadata = []

def add_chunks(vectors: List[List[float]], chunks: List[str]):
    """
    Adds new vectors (embeddings) and their corresponding text chunks to the FAISS index and metadata store.

    Args:
        vectors (List[List[float]]): A list of embedding vectors to be added.
        chunks (List[str]): A list of text chunks corresponding to the vectors.
    """
    global index, metadata # Declare intent to modify global variables.
    # Convert the list of vectors to a NumPy array with float32 data type, as required by FAISS.
    arr = np.array(vectors).astype('float32')
    # Add the vectors to the FAISS index.
    index.add(arr)
    # Extend the global metadata list with the new chunks.
    metadata.extend(chunks)

def save_index():
    """
    Saves the current state of the FAISS index and the associated metadata to disk.
    This ensures persistence of the vector store.
    """
    # Write the FAISS index to its designated file path.
    faiss.write_index(index, INDEX_FILE)
    # Serialize and save the metadata list using pickle.
    with open(META_FILE, 'wb') as f:
        pickle.dump(metadata, f)

def search_top_k(query_vector: List[float], k: int = 3):
    """
    Performs a similarity search on the FAISS index to find the top K most relevant chunks
    based on a given query vector.

    Args:
        query_vector (List[float]): The embedding vector of the query.
        k (int): The number of top similar results to retrieve (default is 3).

    Returns:
        List[tuple]: A list of tuples, each containing a text chunk and its similarity score (distance).
                     Results are filtered to ensure valid indices.
    """
    # Convert the query vector to a NumPy array with float32 data type, wrapped in an array for batching.
    arr = np.array([query_vector]).astype('float32')
    # Perform the search: distances and indices of the k nearest neighbors.
    distances, indices = index.search(arr, k)
    # Compile results, mapping indices back to their original text chunks and scores.
    # Ensures that only valid indices (within the bounds of metadata) are included.
    results = [(metadata[i], distances[0][j]) for j, i in enumerate(indices[0]) if i < len(metadata)]
    return results

# Example usage and testing block for the vector store functionalities.
if __name__ == '__main__':
    # Sample text chunks to be added to the vector store.
    chunks = [
        "EDUCATION\nSoutheastern Louisiana University (SLU)...",
        "EXPERIENCE\nWorked as marketing intern at XYZ Corp...",
        "SKILLS\nPython, SQL, Excel, Tableau",
        "I do not have any skill"
    ]
    # Generate embeddings for the sample chunks using the embed_chunk function.
    vectors = [embed_chunk(c) for c in chunks]
    # Add the generated vectors and chunks to the FAISS index and metadata.
    add_chunks(vectors, chunks)
    # Save the updated index and metadata to disk.
    save_index()

    # Define a query and generate its embedding for searching.
    query = embed_chunk("What are the candidate's skills?")
    # Perform a search to retrieve the top relevant chunks.
    results = search_top_k(query)

    # Print the search results, showing each relevant chunk and its similarity score.
    for chunk, score in results:
        print("\nScore:", score)
        print(chunk)
