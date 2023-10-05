from transformers import pipeline, AutoTokenizer
from tools import *
from typing import *
from collections import Counter
import nltk

model = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
summarize = pipeline("summarization", model=model)
chunk_size = 512

def summarize_text(text: str) -> str:
    """
    Summarizes each chunk of text, then outputs the concatenation.
    """
    chunks = chunk_text(text, chunk_size, tokenizer)
    output = ''
    for chunk in chunks:
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarize(text, min_length = 0, max_length = 128)
        output += '\t' + summary[0]['summary_text'] + '\n'
    return output