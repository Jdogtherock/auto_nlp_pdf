import sys
import os
from typing import *

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pdf_preprocessing import *
from tools import *
from NER import *
import pytest

text_apple = extract_text_from_pdf('samples/Apple_60_Sentences.pdf')

def test_NER():
    result = ner(text_apple)
    assert isinstance(result, dict)
    assert len(result) < len(text_apple)
    assert len(result) >= 0
    for key, values in result.items():
        for value in values:
            assert not value.startswith('#')
