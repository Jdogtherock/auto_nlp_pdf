import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pdf_preprocessing import *
from pdf_eda import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pytest
from PIL import Image

@pytest.fixture
def text():
    """
    Returns the text from the PDF used for testing.
    """
    # Open the PDF as a binary file
    with open("samples/pdf_1k_w_images.pdf", "rb") as file:
        pdf_content = BytesIO(file.read())

    return extract_text_from_pdf(pdf_content)

def test_eda(text):
    results = combined_eda(text)
    assert results["Number of Characters"] > 0 # checks if num chars exists, and is correct
    assert results["Number of Words"] > 1000 # we know the text has > 1,000 words, so this checks for existence and full correctness
    assert results["Number of Sentences"] > 0
    assert len(results["Top 10 Frequent Words, No Stopwords"]) == 10 # asserts the dict of top 10 words is actually 10
    assert isinstance(results["Top 10 Frequent Words, No Stopwords"], dict) # asserts the top 10 freq is a dict
    assert results["Average Word Length"] > 0
    assert results["Average Sentence Length"] > 0
    #print(results)

def test_eda_exact():
    results = combined_eda("Apples Apples Apples. Bananas Bananas. Pear.")
    # 9 Words (Punctuation Included)
    # 44 Characters (Whitespace Included)
    # 3 Sentences
    # Top Frequent: Apples, Bananas, Pear
    assert results["Number of Characters"] == 44 # checks if num chars exists, and is correct
    assert results["Number of Words"] == 9 # we know the text has > 1,000 words, so this checks for existence and full correctness
    assert results["Number of Sentences"] == 3
    assert len(results["Top 3 Frequent Words, No Stopwords"]) == 3
    assert isinstance(results["Top 3 Frequent Words, No Stopwords"], dict) # asserts the top 10 freq is a dict
    assert results["Average Word Length"] == 6
    assert results["Average Sentence Length"] == 3
    #print(results)

def test_pos_chart_output(text):
    result = pos_chart(text)
    # Check if result is of type io.BytesIO
    assert isinstance(result, io.BytesIO)
    # Use PIL to open the BytesIO object and check the format
    image = Image.open(result)
    assert image.format == "PNG"

def test_bar_chart_output(text):
    result = top_bar_chart(text)
    assert isinstance(result, io.BytesIO)
    image = Image.open(result)
    assert image.format == "PNG"




# to convert to unit tests, manually calculate the answers, and use assertions to assert the result is the same.
# these are just functional tests, i trust the code, i just want to make sure its returning what its supposed to.