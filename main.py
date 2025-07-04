import argparse
import os
import anthropic

from convert_pdf_image import pdf_to_image
from extract_text_from_image import extract_resume_structure
from chunk import chunk_resume_text
from embed_chunk import embed_chunk
from vector_store import add_chunks, save_index, search_top_k


def generate_answer_with_claude(question: str, top_k: int = 3):
    """Generate answer using Claude API"""
    # Initialize Claude client
    client = anthropic.Anthropic(
        api_key=os.getenv("PUT_YOUR_ANTHROPIC_API_KEY_HERE")
    )
    
    # Retrieve relevant chunks
    retrieved = search_top_k(question, top_k)
    
    # Prepare context from retrieved chunks
    context = "\n\n".join([chunk for chunk, score in retrieved])
    
    # Create prompt for Claude
    prompt = f"""Based on the following resume information, please answer the question.

Resume Information:
{context}

Question: {question}

Please provide a comprehensive answer based on the resume information provided."""
    
    # Call Claude API
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text, retrieved


def main(pdf_path: str, question: str, top_k: int = 3):
    # 1. Convert PDF to image
    print(f"Converting PDF '{pdf_path}' to image...")
    image_path = pdf_to_image(pdf_path)
    print(f"Image saved at: {image_path}\n")

    # 2. Extract structured text and headers
    print("Extracting text and headers from image...")
    formatted_text, headers = extract_resume_structure(image_path)
    print(f"Detected headers: {headers}\n")

    # 3. Chunk the text by headers
    print("Chunking resume text by headers...")
    chunks = chunk_resume_text(formatted_text, headers)
    print(f"Generated {len(chunks)} chunks.\n")

    # 4. Embed chunks and add to vector store
    print("Embedding chunks and updating vector store...")
    vectors = [embed_chunk(c) for c in chunks]
    # Ensure storage directory exists
    os.makedirs('vector_store', exist_ok=True)
    add_chunks(vectors, chunks)
    save_index()
    print("Index saved to 'vector_store/'.\n")

    # 5. Generate answer using Claude API
    print(f"Retrieving top {top_k} relevant chunks and generating answer...\n")
    answer, retrieved = generate_answer_with_claude(question, top_k=top_k)

    # 6. Output
    print("=== Answer ===")
    print(answer)
    print("\n=== Retrieved Chunks ===")
    for idx, (chunk, score) in enumerate(retrieved, 1):
        print(f"\n[{idx}] (score: {score:.4f})\n{chunk}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAG-based Resume Q&A pipeline")
    parser.add_argument('pdf_path', type=str, help='Path to the resume PDF file')
    parser.add_argument('question', type=str, help='Question to ask about the resume')
    parser.add_argument('--top_k', type=int, default=3, help='Number of chunks to retrieve')
    args = parser.parse_args()
    main(args.pdf_path, args.question, args.top_k)