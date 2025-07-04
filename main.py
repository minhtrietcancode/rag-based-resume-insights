from convert_pdf_image import pdf_to_image
from extract_text_from_image import extract_resume_structure
from chunk import chunk_resume_text
from embed_chunk import embed_chunk

'''
Note the process once again
- pdf path --> convert to image --> return an image path 
- image path --> extract the text from it + headers --> chunk by headers
- with each chunk --> embed it using sentence transformer 
- and then store these vectors of all the chunks in a database 
'''
