from transformers import AutoTokenizer
from tools import *
from typing import *
import collections
import os
import requests
import json

model ="dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
chunk_size = 512

ENDPOINT = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
KEY = os.getenv("API_KEY")
headers = {"Authorization": f"Bearer {KEY}"}

def query(payload):
    data = json.dumps(payload)
    response = requests.post(ENDPOINT, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def ner_query(text):
    output = query({
	"inputs": text,
    })
    print(output)
    return output

def ner(text: str) -> Dict[str, Set[str]]:
    """
    Predicts the named entities and returns them in type of entity:[words that are of that entity type] dictionary format
    """
    ners = collections.defaultdict(set)
    chunks = chunk_text(text, chunk_size, tokenizer)
    for chunk in chunks:
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        ner_results = ner_query(text)
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

