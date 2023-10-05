# functions that multiple tasks use, such as chunk_text
from typing import *

def chunk_text(text: str, chunk_size: int, tokenizer) -> List[List[int]]:
    """
    Takes text and splits it into input id chunks based on the chunk_size
    """
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunks.append(input_ids[i:i+chunk_size])
    return chunks