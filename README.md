# RAG-based Resume Insights: Your AI-Powered Career Assistant

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview

Navigating the job market often involves tailoring your resume for each application, a time-consuming and iterative process. The **RAG-based Resume Insights** is an innovative AI-powered tool designed to streamline this by providing intelligent, real-time feedback on your resume.

Born from the personal challenges of countless job applications and resume refinements, this project leverages **Retrieval-Augmented Generation (RAG)** techniques to enable users to ask natural language questions about their resume and receive accurate, context-aware answers. It acts as your personal career coach, offering AI-driven insights and recommendations to help you optimize your resume, highlight key skills, and improve your chances of success.

### Why This Tool?

*   **Intelligent Insights:** Get answers to specific questions about your resume's content, strengths, and areas for improvement.
*   **Faster Feedback:** Reduce the time spent on manual resume reviews.
*   **Optimized for Success:** Receive AI-driven recommendations to make your resume more impactful and ATS-friendly.
*   **Empowerment:** Gain confidence in your resume by understanding how an AI might interpret it.

## How It Works: The RAG Pipeline

The RAG-based Resume Insights tool operates through a sophisticated pipeline to deliver accurate and relevant answers. Here's a step-by-step breakdown of the process:

1.  **PDF to Image Conversion:** Resumes come in various formats and structures, making direct text extraction inconsistent and prone to losing formatting. To overcome this, the tool first converts the input PDF resume into high-resolution images. This ensures that the original layout and structure are preserved, which is crucial for accurate text extraction.

2.  **Structured Text Extraction:** Once converted to images, advanced Optical Character Recognition (OCR) is used to extract text. This process not only pulls out the raw text but also intelligently identifies and retains the structural elements, such as headers and sections, as they appear in the visual layout.

3.  **Resume Chunking by Headers:** The extracted text is then divided into smaller, logical "chunks." This chunking process is primarily guided by the detected headers (e.g., "Experience," "Education," "Skills"). This ensures that each chunk contains semantically related information, improving the relevance of retrieval.

4.  **Embedding and Vector Storage (FAISS):** Each text chunk is transformed into a dense numerical vector (an embedding) using a Sentence-Transformer model. These embeddings capture the semantic meaning of the text. All these vectors are then stored in a highly efficient vector database called FAISS (Facebook AI Similarity Search).

5.  **Retrieval-Augmented Generation (RAG):** When a user asks a question, the question itself is converted into an embedding. This query embedding is then used to perform a lightning-fast similarity search within the FAISS vector database. The system retrieves the "top K" most relevant chunks (the context) from your resume that are semantically similar to your question.

6.  **LLM-Powered Answering:** Finally, the retrieved relevant chunks, along with your original question, are fed as context to a powerful Large Language Model (LLM), specifically Anthropic's Claude API. The LLM then synthesizes this information to generate a comprehensive, accurate, and context-aware answer, enhancing both the speed and accuracy of the feedback you receive.

## Repository Structure

```
rag-based-resume-ats/
├── README.md
├── requirements.txt
└── src/
    ├── chunk.py
    ├── convert_pdf_image.py
    ├── embed_chunk.py
    ├── extract_text_from_image.py
    ├── main.py
    └── vector_store.py
```

*   `README.md`: This comprehensive guide to the project.
*   `requirements.txt`: Lists all Python dependencies required to run the project.
*   `src/`:
    *   `main.py`: The main entry point of the application. It orchestrates the entire RAG pipeline, from PDF conversion and text extraction to chunking, embedding, and generating answers using the Anthropic Claude API.
    *   `convert_pdf_image.py`: Handles the conversion of PDF resume files into image formats (PNG) for further processing. This is crucial for OCR-based text extraction.
    *   `extract_text_from_image.py`: Utilizes OCR (EasyOCR) to extract structured text and identify potential headers from the converted resume images. It intelligently reconstructs the resume's content.
    *   `chunk.py`: Responsible for dividing the extracted resume text into smaller, manageable "chunks" based on identified headers. This enhances the relevance of information retrieved during the RAG process.
    *   `embed_chunk.py`: Transforms the text chunks into numerical vector embeddings using a Sentence-Transformer model. These embeddings are essential for semantic search and similarity calculations.
    *   `vector_store.py`: Manages the FAISS vector store, which efficiently stores and allows for rapid similarity searches on the embedded resume chunks. It handles adding new chunks, saving the index, and retrieving relevant information.

## Setup & Usage Instructions

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.9+**
*   **pip** (Python package installer)
*   **An Anthropic API Key**: This project uses the Anthropic Claude API for generating intelligent responses. You'll need to set up an environment variable for your API key.

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/rag-based-resume-ats.git
    cd rag-based-resume-ats
    ```

    *(Note: Replace `your-username/rag-based-resume-ats.git` with the actual repository URL if you've forked or cloned from a different source.)*

2.  **Create a Virtual Environment (Recommended):

    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:

    *   **On Windows:**

        ```bash
        .\venv\Scripts\activate
        ```

    *   **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables

Set your Anthropic API key as an environment variable named `ANTHROPIC_API_KEY`.

*   **On Windows (Command Prompt):**

    ```bash
    set ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
    ```

*   **On Windows (PowerShell):**

    ```powershell
    $env:ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
    ```

*   **On macOS/Linux:**

    ```bash
    export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
    ```

    *(Replace `"YOUR_ANTHROPIC_API_KEY"` with your actual API key.)*

### Usage

Once the setup is complete, you can run the `main.py` script to analyze a resume and ask questions.

```bash
python src/main.py "path/to/your/resume.pdf" "What are my key achievements and skills?" --top_k 5
```

*   Replace `"path/to/your/resume.pdf"` with the actual path to your resume PDF file.
*   Replace `"What are my key achievements and skills?"` with the question you want to ask about your resume.
*   `--top_k`: (Optional) Specifies the number of top relevant chunks to retrieve for generating the answer (default is 3).

Example:

```bash
python src/main.py "Business_Resume.pdf" "Summarize my work experience."
```

The tool will output the AI-generated answer based on your resume, along with the retrieved chunks that formed the basis of the answer.

## License

This project is licensed under the MIT License - see the LICENSE file for details.