from transformers import AutoTokenizer
from transformers import pipeline
from tools import *
from typing import *
import collections
import os

model ="dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
chunk_size = 512

import requests

API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
API_KEY = os.getenv("API_KEY")
headers = {"Authorization": API_KEY}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def ner_chunk(text):
    output = query({
	"inputs": text,
    })
    return output

def ner(text: str) -> Dict[str, Set[str]]:
    """
    Predicts the named entities and returns them in type of entity:[words that are of that entity type] dictionary format
    """
    ners = collections.defaultdict(set)
    chunks = chunk_text(text, chunk_size, tokenizer)
    for chunk in chunks:
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        ner_results = ner_chunk(text)
        for entry in ner_results:
            entity_group = entry.get("entity_group")
            word = entry.get("word")
            if entity_group and word:
                if word[0] == '#':
                    continue
                if entity_group[0] == 'O':
                    continue
                ners[entity_group].add(word)
    return ners

