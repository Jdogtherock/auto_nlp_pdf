import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pdf_preprocessing import *
import pytest

@pytest.fixture
def pdf_file():
    """
    Defines the pdf file used for testing.
    """
    with open('samples/pdf_1k_w_images.pdf', 'rb') as f:
        yield BytesIO(f.read())

def test_extract_text_from_pdf(pdf_file):
    """
    Tests the pdf text extraction.
    """
    text = extract_text_from_pdf(pdf_file)
    assert text != ""
    assert isinstance(text, str)
    #print(f"length of text: {len(text)}")

def test_validity_check():
    """
    Tests that the function recognizes invalid pdfs. Prevents pdfs with no words from being uploaded.
    """
    text1 = ""
    text2 = ".?!"
    text3 = " "
    text4 = "and the is of to"
    with open('samples/invalid_pdf.pdf', 'rb') as f:
        text5 = BytesIO(f.read())
    text5 = extract_text_from_pdf(text5)
    assert validity_check(text1) == False
    assert validity_check(text2) == False
    assert validity_check(text3) == False
    assert validity_check(text4) == False
    assert validity_check(text5) == False

def test_first_n_words(pdf_file):
    """
    Test getting the first n words of the pdf.
    """
    text = extract_text_from_pdf(pdf_file)
    first_50_words = first_n_words(text)
    assert len(first_50_words.split()) == 50
    #print(f'first 50 words: {first_50_words}')
