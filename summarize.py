from transformers import AutoTokenizer
from tools import *
from typing import *
from collections import Counter
import os
import json
import requests

model = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
chunk_size = 512

ENDPOINT = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
KEY = os.getenv("API_KEY")
headers = {"Authorization": f"Bearer {KEY}"}

def query(payload):
    data = json.dumps(payload)
    response = requests.post(ENDPOINT, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

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

