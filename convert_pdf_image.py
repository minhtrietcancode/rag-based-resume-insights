import fitz  # PyMuPDF
import os
import easyocr

def pdf_to_image(pdf_path):
    """
    Convert a PDF file to PNG images.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of file paths for the generated PNG images
    """
    doc = fitz.open(pdf_path)
    
    # Extract filename without extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    image_paths = []
    page_count = len(doc)  # Store page count before closing
    
    for page_num in range(page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for quality
        
        # Generate image filename
        image_filename = f'{pdf_name}_page_{page_num + 1}.png'
        pix.save(image_filename)
        image_paths.append(image_filename)
    
    doc.close()
    print(f"Converted {page_count} pages to PNG files with prefix '{pdf_name}'")
    
    return image_paths[0]
