from typing import List
from sentence_transformers import SentenceTransformer
import re

# Initialize the Sentence-Transformers model for embedding text.
# The 'all-MiniLM-L6-v2' model is chosen for its balance of performance and efficiency.
model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_chunk(text: str) -> str:
    """
    Cleans a given text chunk by removing excessive separators and collapsing multiple newlines.
    This prepares the text for more effective embedding.

    Args:
        text (str): The raw text chunk to be cleaned.

    Returns:
        str: The cleaned text chunk.
    """
    # Replace three or more hyphens with a single space to remove section separators.
    text = re.sub(r"-{3,}", " ", text)          # Replace long dashes
    # Collapse two or more newline characters into a single newline to normalize spacing.
    text = re.sub(r"\n{2,}", "\n", text)        # Collapse multiple newlines
    # Remove leading/trailing whitespace from the cleaned text.
    return text.strip()

def embed_chunk(chunk: str) -> List[float]:
    """
    Generates a numerical vector embedding for a given text chunk using the Sentence-Transformers model.
    The chunk is first cleaned to optimize embedding quality.

    Args:
        chunk (str): A text chunk from a resume, intended for embedding.

    Returns:
        List[float]: A list of floats representing the embedding vector.
                     Returns an empty list if the input chunk is empty.
    """
    # Return an empty list immediately if the chunk is empty to avoid processing overhead.
    if not chunk:
        return []

    # Clean the text chunk before generating its embedding.
    cleaned = clean_chunk(chunk)
    # Encode the cleaned chunk into a dense vector, converting the output to a Python list.
    embedding = model.encode(cleaned, convert_to_numpy=True).tolist()
    return embedding
