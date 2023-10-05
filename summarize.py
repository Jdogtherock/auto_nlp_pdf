from transformers import pipeline, AutoTokenizer
from tools import *
from typing import *
from collections import Counter
import nltk

model = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
chunk_size = 512

import requests

API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def summarize_query(text: str) -> str:
    """
    Used to summarize a chunk of text using API
    """
    output = query({
	"inputs": text,
    })
    return output

def summarize_text(text: str) -> str:
    """
    Summarizes each chunk of text, then outputs the concatenation.
    """
    chunks = chunk_text(text, chunk_size, tokenizer)
    output = ''
    for chunk in chunks:
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarize_query(text)[0]["summary_text"]
        output += '\t' + summary + '\n\n'
    return output

