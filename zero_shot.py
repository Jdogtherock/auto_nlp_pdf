from transformers import pipeline, AutoTokenizer
from collections import *
from typing import *
from tools import *
import nltk

LABELS = [
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
classifier = pipeline("zero-shot-classification", model=model)
chunk_size = 512

def get_classifications(text: str) -> Dict[str, int]:
    """
    Tokenizes the input, then loops through the tokens in chunks, classifying each chunk, and storing the result. Returns the counter of predicted labels
    """
    chunks = chunk_text(text, chunk_size, tokenizer)
    pred_labels = defaultdict(int)
    for chunk in chunks:
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        pred_label = classifier(text, LABELS)['labels'][0]
        pred_labels[pred_label] += 1
    return pred_labels

def classify(pred_labels: Dict[str, int]) -> str:
    """
    Receives the counter of predicted labels and outputs the most frequent. Broke apart the methods in order to unit test better
    """
    return max(pred_labels, key=pred_labels.get)





