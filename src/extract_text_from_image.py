import easyocr
from PIL import Image

def extract_resume_structure(image_path, lang_list=['en'], gpu=False, threshold=10):
    """
    Extracts structured text and identifies potential headers from a resume image using EasyOCR.
    It processes the image, groups text into lines, and heuristically detects headers.

    Args:
        image_path (str): The file path to the resume image (e.g., PNG).
        lang_list (list): A list of language codes for OCR processing (default is English).
        gpu (bool): A flag indicating whether to use GPU for OCR acceleration (default is False).
        threshold (int): The Y-axis grouping threshold to determine if text belongs to the same line.
                         A smaller threshold means stricter line grouping.

    Returns:
        formatted_text (str): The extracted and formatted text content of the resume, with headers highlighted.
        headers (list): A list of strings, where each string is a detected header line.
    """
    # Initialize the EasyOCR reader with specified languages and GPU preference.
    reader = easyocr.Reader(lang_list, gpu=gpu)
    # Perform OCR on the image to get bounding box, text, and confidence for each detected element.
    results = reader.readtext(image_path)

    # Annotate results with the vertical midpoint of each text bounding box.
    # This helps in sorting and grouping text into lines.
    annotated = []
    for bbox, text, _ in results:
        mid_y = sum(pt[1] for pt in bbox) / 4  # Calculate the average Y-coordinate of the bounding box.
        annotated.append((mid_y, bbox, text))
    # Sort the annotated text elements based on their vertical position (mid_y).
    annotated.sort(key=lambda x: x[0])

    # Group sorted text elements into lines based on a Y-axis threshold.
    # This logic reconstructs lines of text from individual word detections.
    lines, current_line, last_y = [], [], None
    for mid_y, bbox, text in annotated:
        if last_y is None or abs(mid_y - last_y) < threshold:
            # If it's the first element or close to the last element's Y-position, add to current line.
            current_line.append((bbox, text, mid_y))
        else:
            # Otherwise, start a new line.
            lines.append(current_line)
            current_line = [(bbox, text, mid_y)]
        last_y = mid_y  # Update the last Y-position for the next iteration.
    if current_line:
        # Add any remaining elements as the last line.
        lines.append(current_line)

    # Sort words within each line by their X-coordinate and concatenate them to form coherent lines of text.
    structured_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda x: min(pt[0] for pt in x[0])) # Sort words in line by their starting X-coordinate.
        line_text = ' '.join(word for _, word, _ in line_sorted) # Join words to form a line of text.
        structured_lines.append(line_text)

    # Heuristically detect headers based on capitalization and word count.
    headers = []
    formatted_text_lines = []
    for line in structured_lines:
        words = line.split()
        # Calculate the ratio of uppercase characters in the line.
        upper_ratio = sum(1 for c in line if c.isupper()) / max(len(line), 1)
        # A line is considered a header if a high percentage of its characters are uppercase and it's relatively short.
        is_header = upper_ratio > 0.6 and len(words) < 6
        if is_header:
            headers.append(line)
            # Format headers with surrounding newlines and a separator for readability.
            formatted_text_lines.append(f"\n{line}\n" + "-" * len(line))
        else:
            formatted_text_lines.append(line)

    # Join all formatted lines to create the final resume text.
    formatted_text = '\n'.join(formatted_text_lines)
    return formatted_text, headers
