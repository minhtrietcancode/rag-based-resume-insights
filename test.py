from convert_pdf_image import pdf_to_image
from extract_text_from_image import extract_resume_structure

# Load the pdf and convert to image 
pdf_path = "sample pdf\Business_Resume.docx.pdf"
image_path = pdf_to_image(pdf_path)


# Extract text and header from the image 
format_text, headers = extract_resume_structure(image_path)

# check the result 
print(format_text)
print(headers)


