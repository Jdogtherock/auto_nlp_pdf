import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pdf_preprocessing import *
from tools import *
from zero_shot import *
import pytest

text = extract_text_from_pdf('samples/Apple_60_Sentences.pdf')

def test_classify():
    """
    tests the getting the dictionary of classifications, as well as the most frequent classification
    """
    chunks = chunk_text(text, chunk_size, tokenizer)
    assert isinstance(chunks, list)
    assert all(len(chunk) <= chunk_size and len(chunk) > 0 for chunk in chunks)

    pred_labels = get_classifications(text)
    assert isinstance(pred_labels, dict)

    pred_label = classify(pred_labels)
    assert isinstance(pred_label, str)

    print(pred_label)


