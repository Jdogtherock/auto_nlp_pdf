import polars as pl
import torch
import torch.nn.functional as F

import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from embeddings import is_string, get_sentence_embedding, embed

import pytest

def test_is_string():
    df = pl.DataFrame({
        'a': ['hello', 'world', 'hey'],
        'b': [1.2, 3.4, 5.6]
    })

    # Test 1: Column of strings
    assert is_string(df['a']) == True

    # Test 2: Column of floats
    assert is_string(df['b']) == False

def test_get_sentence_embedding():
    # Test 1: Simple sentence embedding
    sentence = ["Hello world"]
    embedding = get_sentence_embedding(sentence)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape[0] == 1  # We passed one sentence

    # Test 2: Multiple sentences
    sentences = ["Hello world", "This is a test"]
    embeddings = get_sentence_embedding(sentences)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 2  # We passed two sentences

def test_embed():
    # Test 1: DataFrame with string columns
    df = pl.DataFrame({'a': ["Hello", "World"], 'b': ["This", "Test"]})
    new_df = embed(df)
    # Print the length of each embedding for column 'a'
    #print("Length of embedding for column 'a':", len(new_df['a'][0]))
    # Print the length of each embedding for column 'b'
    #print("Length of embedding for column 'b':", len(new_df['b'][0]))
    assert new_df.shape == df.shape  # Should have the same shape as input

    # Test 2: DataFrame with mixed columns
    df = pl.DataFrame({'a': ["Hello", "World"], 'b': [1.2, 3.4]})
    new_df = embed(df)
    # Print the length of each embedding for column 'a'
    #print("Length of embedding for column 'a':", len(new_df['a'][0]))
    assert new_df.shape == df.shape  # Should have the same shape as input
    assert isinstance(new_df['b'][0], float)  # 'b' column should remain unchanged

    # Test 3: Compare embeddings of different sentences
    # 3.1 Two identical sentences
    df_identical = pl.DataFrame({'sentences': ["Hello world", "Hello world"]})
    embeddings_identical = embed(df_identical)['sentences']
    similarity_identical = F.cosine_similarity(torch.tensor(embeddings_identical[0]), torch.tensor(embeddings_identical[1]), dim=0).item()
    #print(f"Cosine similarity between identical sentences: {similarity_identical:.4f}")

    # 3.2 Two very different sentences
    df_different = pl.DataFrame({'sentences': ["Hello world", "Apples are tasty"]})
    embeddings_different = embed(df_different)['sentences']
    similarity_different = F.cosine_similarity(torch.tensor(embeddings_different[0]), torch.tensor(embeddings_different[1]), dim=0).item()
    #print(f"Cosine similarity between different sentences: {similarity_different:.4f}")

    # 3.3 Two similar sentences
    df_similar = pl.DataFrame({'sentences': ["Hello world", "Hi world"]})
    embeddings_similar = embed(df_similar)['sentences']
    similarity_similar = F.cosine_similarity(torch.tensor(embeddings_similar[0]), torch.tensor(embeddings_similar[1]), dim=0).item()
    #print(f"Cosine similarity between similar sentences: {similarity_similar:.4f}")
