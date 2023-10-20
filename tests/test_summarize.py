import sys
import os
from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pdf_preprocessing import *
from tools import *
from summarize import *
import pytest

text_apple = extract_text_from_pdf('samples/Apple_60_Sentences.pdf')

def test_summarize_text():
    result = summarize_text(text_apple)
    assert isinstance(result, str)
    assert len(result) < len(text_apple)
    assert len(result) > 0