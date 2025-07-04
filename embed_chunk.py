from typing import List
from sentence_transformers import SentenceTransformer
import re

# Load the Sentence-Transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_chunk(text: str) -> str:
    """
    Optional: Clean a chunk of resume text before embedding.
    - Removes excessive separators.
    - Collapses multiple newlines.
    """
    text = re.sub(r"-{3,}", " ", text)          # Replace long dashes
    text = re.sub(r"\n{2,}", "\n", text)        # Collapse multiple newlines
    return text.strip()

def embed_chunk(chunk: str) -> List[float]:
    """
    Embed a single cleaned text chunk using Sentence-Transformers.

    Args:
        chunk (str): A chunk of text from a resume.

    Returns:
        List[float]: The embedding vector.
    """
    if not chunk:
        return []

    cleaned = clean_chunk(chunk)
    embedding = model.encode(cleaned, convert_to_numpy=True).tolist()
    return embedding

if __name__ == '__main__':
    # Quick local test
    sample = (
        "EDUCATION\n"
        "---------\n"
        "Southeastern Louisiana University (SLU) Hammond, LA\n"
        "Bachelor of Arts in Marketing; Spanish minor May 2021\n"
        "Major GPA: 3.50/4.00; Overall GPA: 3.65/4.00"
    )
    vec = embed_chunk(sample)
    print(f"Embedding length: {len(vec)}\nSample values: {vec[:5]}")
