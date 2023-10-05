# Exploratory Data Analysis for extracted text from pdf

import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import matplotlib.pyplot as plt
import io

def text_characteristics(text):
    """Returns the number of characters, words, and sentences in the given text."""
    num_characters = len(text)
    num_words = len(word_tokenize(text))
    num_sentences = len(sent_tokenize(text))
    return {
        "Number of Characters": num_characters,
        "Number of Words": num_words,
        "Number of Sentences": num_sentences
    }

def top_frequent_words(text, n=10):
    """Returns the top n most frequent words (excluding stopwords) in the given text."""
    words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    frequency = Counter(filtered_words)
    actual_n = min(n, len(frequency))
    return {"Top {} Frequent Words, No Stopwords".format(actual_n): dict(frequency.most_common(n))}

def top_bar_chart(text, n=10):
    """Generate a bar chart of the top n words' frequency and return as a BytesIO object."""
    # Get the top n frequent words
    word_data = top_frequent_words(text, n)
    title, data = list(word_data.items())[0]
    words, counts = zip(*data.items())

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='skyblue')
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xticks(rotation=65)  # Adjust rotation of x-labels
    plt.subplots_adjust(bottom=0.3)  # Adjust bottom margin to make room for x-labels

    # Save plot to BytesIO object and return
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return buf

def average_word_length(text):
    """Returns the average length of words in the given text."""
    words = [word for word in word_tokenize(text) if word.isalpha()]
    avg_length = sum([len(word) for word in words]) / len(words)
    return {"Average Word Length": round(avg_length, 2)}

def average_sentence_length(text):
    """Returns the average number of words in sentences of the given text."""
    sentences = sent_tokenize(text)
    avg_length = sum([len(word_tokenize(sentence)) for sentence in sentences]) / len(sentences)
    return {"Average Sentence Length": round(avg_length, 2)}

# Top 10 Bar Chart

def pos_chart(text):
    # Tokenize and POS-tag
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    # Count POS tags
    pos_counts = Counter(tag for word, tag in pos_tags)
    # Simplify and aggregate some of the tags
    simplified_counts = {"Noun": 0, "Verb": 0, "Adjective": 0, "Adverb": 0, "Other": 0}
    for tag, count in pos_counts.items():
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            simplified_counts["Noun"] += count
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            simplified_counts["Verb"] += count
        elif tag in ['JJ', 'JJR', 'JJS']:
            simplified_counts["Adjective"] += count
        elif tag in ['RB', 'RBR', 'RBS']:
            simplified_counts["Adverb"] += count
        else:
            simplified_counts["Other"] += count
    # Plot pie chart
    plt.figure(figsize=(10, 5))
    plt.pie(simplified_counts.values(), labels=simplified_counts.keys(), autopct='%1.1f%%', startangle=140, colors=['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink'])
    plt.title("Part-of-Speech Distribution")
    plt.axis("equal")
    # Save plot to BytesIO object and return
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

def combined_eda(text):
    """Combine results from all EDA functions for easy display in Streamlit."""
    results = {}
    results.update(text_characteristics(text))
    results.update(top_frequent_words(text))
    results.update(average_word_length(text))
    results.update(average_sentence_length(text))
    return results

