from transformers import pipeline, AutoTokenizer
from collections import *
from typing import *
from tools import *
import nltk
import os


CANDIDATE_LABELS = [
    "Entertainment",
    "STEM",
    "Health",
    "Politics",
    "Business",
    "Environment",
    "Culture",
    "Recreation",
    "Lifestyle",
    "Educational",
]

model = 'cross-encoder/nli-distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
chunk_size = 512

import requests

API_URL = "https://api-inference.huggingface.co/models/cross-encoder/nli-distilroberta-base"
API_KEY = os.getenv("API_KEY")
headers = {"Authorization": API_KEY} # need to make the token a secret

def query(payload):
    """
    Defines the method to call the zero_shot API
    """
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def zero_shot_query(text):
    """
    Sends the input text to the zero_shot API, receives candidate label scores
    """
    output = query({
    "inputs": text,
    "parameters": {"candidate_labels": CANDIDATE_LABELS},
    })
    return output

def get_zero_shot_classifications(text: str) -> Dict[str, int]:
    """
    Tokenizes the input, then loops through the tokens in chunks, classifying each chunk, and storing the result. Returns the counter of predicted labels
    """
    chunks = chunk_text(text, chunk_size, tokenizer)
    pred_labels = defaultdict(int)
    for chunk in chunks:
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        pred_label = zero_shot_query(text)["labels"][0]
        pred_labels[pred_label] += 1
    return pred_labels

def zero_shot_classify(text: str) -> str:
    """
    Full Zero Shot Classification
    """
    pred_labels = get_zero_shot_classifications(text)
    return max(pred_labels, key=pred_labels.get)






