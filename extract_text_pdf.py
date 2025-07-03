import PyPDF2

def extract_text_from_pdf(pdf_path):
    # Open the PDF file in binary reading mode
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text_pages = []

        # Iterate through all pages and extract text
        for page in reader.pages:
            text_pages.append(page.extract_text() or "")

    # Join individual page texts into one string
    return "\n\n".join(text_pages)


