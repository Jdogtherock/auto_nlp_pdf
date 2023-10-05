from transformers import AutoTokenizer
from transformers import pipeline
from tools import *
from typing import *
import collections


model ="dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
chunk_size = 512

def ner(text: str) -> Dict[str, Set[str]]:
    """
    Predicts the named entities and returns them in type of entity:[words that are of that entity type] dictionary format
    """
    ners = collections.defaultdict(set)
    chunks = chunk_text(text, chunk_size, tokenizer)
    for chunk in chunks:
        text = tokenizer.decode(chunk, skip_special_tokens=True)
        ner_results = nlp(text)
        for element in ner_results:
            type = element['entity']
            word = element['word']
            if word[0] == '#':
                continue
            if type[0] == 'O':
                continue
            ners[type].add(word)
    return ners


