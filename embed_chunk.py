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

if __name__ == '__main__':
    # This block demonstrates the functionality of the embed_chunk function with a sample text.
    # It serves as a quick local test to verify the embedding process.
    sample = (
        "EDUCATION\n"
        "---------\n"
        "Southeastern Louisiana University (SLU) Hammond, LA\n"
        "Bachelor of Arts in Marketing; Spanish minor May 2021\n"
        "Major GPA: 3.50/4.00; Overall GPA: 3.65/4.00"
    )
    # Generate the embedding for the sample text.
    vec = embed_chunk(sample)
    # Print the length of the generated embedding and the first five values for verification.
    print(f"Embedding length: {len(vec)}\nSample values: {vec[:5]}")
