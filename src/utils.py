import re
import nltk
from PyPDF2 import PdfReader

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_resume(text):
    """Advanced cleaning array optimized for resume noise extraction."""
    text = re.sub(r"http\S+\s*", " ", text)  # Remove URLs
    text = re.sub(r"RT|cc", " ", text)      # Remove RT and cc marks
    text = re.sub(r"#\S+", " ", text)       # Remove hashtags
    text = re.sub(r"@\S+", " ", text)       # Remove mentions
    
    # Cleaned punctuation string using raw string pattern matching
    text = re.sub(r'[^\w\s]', ' ', text)
    
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"\s+", " ", text).strip()     # Remove duplicate extra spaces
    return text.lower()

def extract_text_from_pdf(pdf_file):
    """Headless parsing of binary PDF buffers."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text