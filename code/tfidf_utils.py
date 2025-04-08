import json
import math
import re
from collections import Counter

def tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())

def compute_tf(tokens: list) -> dict:
    tf_counter = Counter(tokens)
    total_tokens = len(tokens)
    tf = {term: count / total_tokens for term, count in tf_counter.items()}
    return tf

def compute_idf(documents_tokens: list) -> dict:
    N = len(documents_tokens)
    idf = {}
    for tokens in documents_tokens:
        unique_tokens = set(tokens)
        for term in unique_tokens:
            idf[term] = idf.get(term, 0) + 1
    for term, df in idf.items():
        idf[term] = math.log(N / df)
    return idf

def compute_tfidf(documents: list) -> list:
    tokenized_docs = [tokenize(doc) for doc in documents]
    idf = compute_idf(tokenized_docs)
    tfidf_documents = []
    for tokens in tokenized_docs:
        tf = compute_tf(tokens)
        tfidf = {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}
        tfidf_documents.append(tfidf)
    return tfidf_documents

def compute_tfidf_for_recipes(train_json_path: str):
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    documents = []
    recipe_ids = []
    for recipe in train_data:
        recipe_id = recipe.get('id')
        recipe_ids.append(recipe_id)
        ingredient_texts = [ing['name'] for ing in recipe.get('ingredients', [])]
        combined_text = " ".join(ingredient_texts)
        documents.append(combined_text)

    tfidf_vectors = compute_tfidf(documents)
    recipe_tfidf = list(zip(recipe_ids, tfidf_vectors))
    return recipe_tfidf

if __name__ == "__main__":
    train_json_path = r"C:\Users\Field Luo\Documents\Natural Language Process\Final Project\train.json"
    results = compute_tfidf_for_recipes(train_json_path)
    for i, (r_id, tfidf_vec) in enumerate(results[:3], 1):
        print(f"Recipe ID: {r_id}")
        print("TF-IDF Vector (term => score):")
        for term, score in list(tfidf_vec.items())[:10]:
            print(f"  {term}: {score:.4f}")
        print("--------------------------------------------------")
