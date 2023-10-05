from pypdf import PdfReader
from io import BytesIO
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import string

def extract_text_from_pdf(uploaded_file: BytesIO) -> str:
    """
    Extracts the text from the given PDF file.
    """
    pdf_file = PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_file.pages)):
        page = pdf_file.pages[page_num]
        text += page.extract_text()
    return text

def validity_check(text: str, n = 5) -> bool:
    """
    Checks if the text actually contains non-stopwords, and greater than n words
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if w.lower() not in stop_words]
    filtered_text = [w for w in filtered_text if w not in string.punctuation]
    print(filtered_text)
    return len(filtered_text) > 5


def first_n_words(text: str, n=50) -> str:
    """
    Prints the first n words of a string.
    """
    words = text.split()
    return ' '.join(words[:n])