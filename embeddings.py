from transformers import AutoTokenizer, AutoModel
import torch
import polars as pl

def is_string(col):
    try:
        float(col[0])
        return False
    except ValueError:
        return True

# using bge-base-en model, top ranked 768-dim embedding model for MTEB
# might change to bge-small if too slow, or too much space used

EMBEDDING = 'BAAI/bge-base-en'
TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING)
MODEL = AutoModel.from_pretrained(EMBEDDING)

# INPUT: DATAFRAME WITH POSSIBLE TEXT FEATURES
# OUTPUT: DATAFRAME WITH THE TEXT FEATURES CONVERTED TO SENTENCE EMBEDDINGS

def get_sentence_embedding(sentences):
    """Return the embedding for a given sentence."""
    inputs = TOKENIZER(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = MODEL(**inputs)
        sentence_embeddings = outputs[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def embed(df):
    """Convert text columns in given dataframe to sentence embeddings."""
    for column in df.columns:
        if is_string(df[column]):
            embeddings = get_sentence_embedding(df[column].to_list())
            df = df.with_columns(pl.Series(embeddings.tolist()).alias(column))
    return df
