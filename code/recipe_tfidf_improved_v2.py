import json
import os
import math
import re
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Set
import time
from spellchecker import SpellChecker
import difflib

class OptimizedRecipeSearchEngine:
    def __init__(self,
                 json_path: str = "cleaned_data/train.json",
                 cache_dir: str = "tfidf_cache",
                 title_weight: float = 2.0,
                 ingredients_weight: float = 1.5,
                 directions_weight: float = 0.5):
        """
        Initialize the recipe search engine and load or compute TF-IDF vectors.

        Args:
            json_path: Path to the JSON file containing recipes.
            cache_dir: Directory to cache TF-IDF data.
            title_weight: Weight multiplier for the title section.
            ingredients_weight: Weight multiplier for the ingredients section.
            directions_weight: Weight multiplier for the directions section.
        """
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.title_weight = title_weight
        self.ingredients_weight = ingredients_weight
        self.directions_weight = directions_weight

        # Initialize the spell checker for typo correction
        self.spell = SpellChecker(distance=1)

        # Basic stopwords to filter out common non-informative words
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'of',
            'to', 'from', 'with', 'by', 'for', 'at', 'is', 'are',
            'cup', 'cups', 'tablespoon', 'tablespoons', 'teaspoon',
            'teaspoons', 'recipe', 'about', 'until', 'tsp','tsps'
        }

        # Load all recipes into memory
        self.recipes = self._load_recipes()

        # Compute or load cached TF-IDF vectors
        start = time.time()
        self.recipe_tfidf, self.vocabulary = self._load_or_compute_tfidf()
        elapsed = time.time() - start
        print(f"TF-IDF ready in {elapsed:.2f}s | Vocabulary size: {len(self.vocabulary)} terms")

    def _load_recipes(self) -> Dict[str, dict]:
        """Load recipes from JSON and map by recipe ID."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {r['id']: r for r in data}

    def _get_cache_path(self) -> str:
        """Generate a cache filename based on current weights and JSON filename."""
        os.makedirs(self.cache_dir, exist_ok=True)
        base = os.path.basename(self.json_path)
        config = f"_tw{self.title_weight:.1f}_iw{self.ingredients_weight:.1f}_dw{self.directions_weight:.1f}"
        return os.path.join(self.cache_dir, f"{base}{config}.pkl")

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, remove non-alpha, split on whitespace, filter stopwords."""
        if not text:
            return []
        lowered = text.lower()
        cleaned = re.sub(r'[^a-z\s]', ' ', lowered)
        tokens = [tok for tok in cleaned.split() if tok and tok not in self.stopwords]
        return tokens

    def _correct_spelling(self, tokens: List[str]) -> List[str]:
        """Correct spelling using SpellChecker and fallback to fuzzy match on vocabulary."""
        corrected = []
        for tok in tokens:
            # primary correction
            corr = self.spell.correction(tok)
            if corr and corr != tok:
                corrected.append(corr)
            else:
                # fallback fuzzy match against vocabulary
                matches = difflib.get_close_matches(tok, self.vocabulary, n=1, cutoff=0.7)
                corrected.append(matches[0] if matches else tok)
        return corrected

    def _extract_sections(self, recipe: dict) -> Dict[str, str]:
        """Extract raw text from title, ingredients, and directions."""
        title = recipe.get('title', '')
        ingredients = ' '.join(
            ing.get('name', '') if isinstance(ing, dict) else str(ing)
            for ing in recipe.get('ingredients', [])
        )
        directions = ' '.join(recipe.get('directions', []))
        return {'title': title, 'ingredients': ingredients, 'directions': directions}

    def _load_or_compute_tfidf(self) -> Tuple[List[Tuple[str, Dict[str, float]]], Set[str]]:
        """Load TF-IDF from cache or compute it from scratch."""
        cache_file = self._get_cache_path()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass

        tokenized = {}
        doc_freq = Counter()
        total = len(self.recipes)
        for rid, rec in self.recipes.items():
            secs = self._extract_sections(rec)
            toks = {s: self._tokenize(txt) for s, txt in secs.items()}
            tokenized[rid] = toks
            unique = set(t for toks in toks.values() for t in toks)
            for t in unique:
                doc_freq[t] += 1

        vocab = {t for t, f in doc_freq.items() if 3 <= f <= total * 0.7}
        idf = {t: math.log(total / doc_freq[t]) for t in vocab}

        tfidf_list = []
        for rid, secs in tokenized.items():
            vec = {}
            for section, toks in secs.items():
                weight = getattr(self, f"{section}_weight")
                length = max(1, len(toks))
                for t in toks:
                    if t in vocab:
                        tf = toks.count(t) / length
                        vec[t] = vec.get(t, 0) + tf * idf[t] * weight
            tfidf_list.append((rid, vec))

        with open(cache_file, 'wb') as f:
            pickle.dump((tfidf_list, vocab), f)
        return tfidf_list, vocab

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Compute cosine similarity based on common keys."""
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        mag = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
        return dot / mag if mag else 0.0

    def search(self, query: str, top_n: int = 5, debug: bool = False) -> List[dict]:
        """Search for recipes matching the query.

        Args:
            query: User input string.
            top_n: Number of top results to return.
            debug: If True, print tokenization and correction steps.
        """
        tokens = self._tokenize(query)
        corrected = self._correct_spelling(tokens)
        if debug:
            print(f"Original tokens: {tokens}")
            print(f"Corrected tokens: {corrected}")

        qvec = {}
        for t in corrected:
            if t in self.vocabulary:
                qvec[t] = qvec.get(t, 0) + 1
        length = len(corrected) or 1
        for t in qvec:
            qvec[t] /= length

        if not qvec:
            return []

        scores = []
        for rid, vec in self.recipe_tfidf:
            sim = self._cosine(qvec, vec)
            if sim > 0:
                scores.append((rid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rid, sc in scores[:top_n]:
            rec = self.recipes[rid]
            ings = [ing.get('name','') if isinstance(ing, dict) else str(ing)
                    for ing in rec.get('ingredients', [])]
            steps = rec.get('directions', [])
            results.append({
                'id': rid,
                'title': rec.get('title',''),
                'score': round(sc, 4),
                'ingredients': ings[:8],
                'steps': steps
            })
        return results

    def generate_test_queries(self) -> Dict[str, List[str]]:
        """Generate a set of test queries for edge-case validation."""
        base = ['chicken', 'garlic', 'salt', 'pepper', 'onion', 'spaghetti']
        return {
            'typos': [b[:2] + 'ik' + b[2:] for b in base],
            'empty': [''],
            'partial': ['chicken garlic', 'salt pepper'],
            'unit_noise': ['2cupschicken', '1/2tbsp garlic'],
            'extra_noise': ['please show me frid chiken recipe?']
        }


def main():
    engine = OptimizedRecipeSearchEngine()
    # This is just used for testing
    '''
    print("\n=== Test Query Processing ===")
    tests = engine.generate_test_queries()
    for cat, qs in tests.items():
        print(f"\nCategory: {cat}")
        for q in qs:
            print(f"Query: '{q}'")
            engine.search(q, debug=True)
    '''
    print("\n=== Interactive Search ===")
    print("Type 'q' to quit, or enter keywords to search recipes.")
    while True:
        inp = input("\nSearch recipes: ").strip()
        if not inp or inp.lower() == 'q':
            break
        results = engine.search(inp, debug=True)
        if not results:
            print("No matching recipes.")
        else:
            print(f"Found {len(results)} results:")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} (score {r['score']})")
                print("   Ingredients:", ", ".join(r['ingredients']))
                print("   Steps:")
                for step in r['steps']:
                    print(f"     - {step}")

if __name__ == '__main__':
    main()
