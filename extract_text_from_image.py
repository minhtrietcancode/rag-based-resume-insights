import easyocr
from PIL import Image

def extract_resume_structure(image_path, lang_list=['en'], gpu=False, threshold=10):
    """
    Extracts structured text and headers from a resume image using EasyOCR.

    Args:
        image_path (str): Path to the resume image file.
        lang_list (list): List of language codes for OCR.
        gpu (bool): Whether to use GPU acceleration.
        threshold (int): Y-axis grouping threshold for line separation.

    Returns:
        formatted_text (str): Resume content as formatted string.
        headers (list): List of detected header lines.
    """
    reader = easyocr.Reader(lang_list, gpu=gpu)
    results = reader.readtext(image_path)

    # Annotate results with vertical position
    annotated = []
    for bbox, text, _ in results:
        mid_y = sum(pt[1] for pt in bbox) / 4
        annotated.append((mid_y, bbox, text))
    annotated.sort(key=lambda x: x[0])

    # Group into lines based on Y-axis threshold
    lines, current_line, last_y = [], [], None
    for mid_y, bbox, text in annotated:
        if last_y is None or abs(mid_y - last_y) < threshold:
            current_line.append((bbox, text, mid_y))
        else:
            lines.append(current_line)
            current_line = [(bbox, text, mid_y)]
        last_y = mid_y
    if current_line:
        lines.append(current_line)

    # Sort words in lines and concatenate
    structured_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda x: min(pt[0] for pt in x[0]))
        line_text = ' '.join(word for _, word, _ in line_sorted)
        structured_lines.append(line_text)

    # Detect headers heuristically
    headers = []
    formatted_text_lines = []
    for line in structured_lines:
        words = line.split()
        upper_ratio = sum(1 for c in line if c.isupper()) / max(len(line), 1)
        is_header = upper_ratio > 0.6 and len(words) < 6
        if is_header:
            headers.append(line)
            formatted_text_lines.append(f"\n{line}\n" + "-" * len(line))
        else:
            formatted_text_lines.append(line)

    formatted_text = '\n'.join(formatted_text_lines)
    return formatted_text, headers
