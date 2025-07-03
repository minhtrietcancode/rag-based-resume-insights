from extract_text_pdf import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

COMMON_HEADERS = [
    # Education section
    "Education", "Academic Background", "Educational Background", 
    "Academic Qualifications", "University", "School",
    
    # Experience section
    "Experience", "Work Experience", "Professional Experience", 
    "Employment History", "Career", "Work History", "Employment",
    "Professional Background", "Job Experience",
    
    # Skills section
    "Skills", "Technical Skills", "Core Competencies", "Technologies",
    "Programming Languages", "Software Skills", "Tools", "Languages",
    "Technical Proficiencies", "Expertise", "Competencies",
    
    # Projects section
    "Projects", "Personal Projects", "Academic Projects", "Key Projects",
    "Notable Projects", "Selected Projects", "Portfolio",
    
    # Achievements section
    "Honors", "Awards", "Honors and Awards", "Achievements", 
    "Recognition", "Accomplishments", "Distinctions",
    
    # Certifications section
    "Certifications", "Certificates", "Professional Certifications",
    "Licenses", "Credentials",
    
    # Other common sections
    "Publications", "Research", "Activities", "Extracurricular",
    "Volunteer", "Volunteering", "Leadership", "Interests",
    "Summary", "Objective", "Profile", "About", "Contact",
    "References", "Additional Information", "Other", "Miscellaneous"
]

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.6
MAX_HEADER_WORDS = 5

# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize model and embeddings
model = SentenceTransformer(MODEL_NAME)
common_headers_lower = [header.lower() for header in COMMON_HEADERS]
header_embeddings = model.encode(common_headers_lower)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def preprocess_line(line: str) -> str:
    """
    Clean and preprocess a line for better matching.
    
    Args:
        line: Raw line from text
        
    Returns:
        Cleaned and normalized line
    """
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', line.strip())
    
    # Remove common decorative elements
    cleaned = re.sub(r'^[-=_•\*\+]+\s*', '', cleaned)
    cleaned = re.sub(r'\s*[-=_•\*\+]+$', '', cleaned)
    
    # Remove trailing colons
    cleaned = re.sub(r':+$', '', cleaned)
    
    return cleaned.strip().lower()

def is_potential_header(line: str) -> bool:
    """
    Quick filtering to eliminate obviously non-header lines.
    Improves performance by avoiding embedding computation for non-headers.
    
    Args:
        line: Line to check
        
    Returns:
        True if line could potentially be a header
    """
    cleaned = preprocess_line(line)
    
    # Skip empty lines
    if not cleaned:
        return False
    
    # Skip if more than MAX_HEADER_WORDS (optimization)
    word_count = len(cleaned.split())
    if word_count > MAX_HEADER_WORDS:
        return False
    
    return True

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def calculate_header_similarity(line: str, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float, str]:
    """
    Calculate similarity between a line and known headers with optimized matching.
    
    Args:
        line: The line to check
        threshold: Minimum similarity score to consider as header
        
    Returns:
        Tuple of (is_header, max_similarity_score, best_matching_header)
    """
    cleaned_line = preprocess_line(line)
    
    # Quick filter - avoid expensive embedding computation
    if not is_potential_header(cleaned_line):
        return False, 0.0, ""
    
    # Exact match check (fastest)
    for header in common_headers_lower:
        if cleaned_line == header:
            return True, 1.0, header
    
    # Substring match check
    for header in common_headers_lower:
        if cleaned_line in header or header in cleaned_line:
            # High score for close substring matches
            if abs(len(cleaned_line) - len(header)) <= 2:
                return True, 0.95, header
    
    # Compute embeddings only if quick checks fail
    line_embedding = model.encode([cleaned_line])
    similarities = cosine_similarity(line_embedding, header_embeddings)[0]
    
    # Find best match
    max_similarity = np.max(similarities)
    best_match_idx = np.argmax(similarities)
    best_matching_header = common_headers_lower[best_match_idx]
    
    is_header = max_similarity >= threshold
    
    return is_header, max_similarity, best_matching_header

def detect_headers_in_text(text: str, threshold: float = DEFAULT_THRESHOLD) -> List[Dict]:
    """
    Detect all headers in the given text with performance optimizations.
    
    Args:
        text: Full resume text
        threshold: Similarity threshold for header detection
        
    Returns:
        List of dictionaries with header information (high confidence only)
    """
    lines = text.splitlines()
    detected_headers = []
    
    # Pre-filter lines to avoid processing non-header lines
    potential_lines = []
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line and len(stripped_line.split()) <= MAX_HEADER_WORDS:
            potential_lines.append((i, stripped_line))
    
    # Process potential header lines
    for line_index, line in potential_lines:
        is_header, similarity, best_match = calculate_header_similarity(line, threshold)
        
        if is_header:
            # Determine confidence level
            if similarity > 0.8:
                confidence = 'High'
            elif similarity > 0.7:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            detected_headers.append({
                'line_index': line_index,
                'original_line': line,
                'cleaned_line': preprocess_line(line),
                'similarity_score': similarity,
                'best_match': best_match,
                'confidence': confidence
            })
    
    # Return only high confidence headers
    return [header for header in detected_headers if header['confidence'] == 'High']

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to demonstrate header detection."""
    sample_pdf = "sample pdf/Minh_Triet_Pham_Resume (1).pdf"
    
    try:
        extracted_text = extract_text_from_pdf(sample_pdf)
        headers = detect_headers_in_text(extracted_text)
        
        print(f"Found {len(headers)} high-confidence headers:")
        for header in headers:
            print(f"Line {header['line_index']}: {header['original_line']}")
            print(f"  Similarity: {header['similarity_score']:.3f}")
            print(f"  Best match: {header['best_match']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error processing PDF: {e}")

if __name__ == "__main__":
    main()