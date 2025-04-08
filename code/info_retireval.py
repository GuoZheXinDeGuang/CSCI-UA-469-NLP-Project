import json
import math
import os
from tfidf_utils import compute_tfidf_for_recipes, tokenize, compute_tf

class RecipeSearchEngine:
    def __init__(self, json_path: str, cache_dir: str = "tfidf_cache"):
        """
        Initialize recipe search engine
        :param json_path: Path to recipes JSON file
        :param cache_dir: Cache directory name
        """
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.recipes = self._load_recipes()
        self.recipe_tfidf = self._load_or_compute_tfidf()
        self.vocabulary = self._build_vocabulary()
        
    def _load_recipes(self) -> dict:
        """Load recipes and create ID-to-recipe mapping"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        return {recipe['id']: recipe for recipe in recipes}
    
    def _get_cache_path(self) -> str:
        """Get cache file path"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        base_name = os.path.basename(self.json_path)
        return os.path.join(self.cache_dir, f"{base_name}.tfidf_cache")
    
    def _load_or_compute_tfidf(self):
        """Load from cache or compute TF-IDF"""
        cache_path = self._get_cache_path()
        
        # Load from cache if exists
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"Loaded TF-IDF data from cache: {cache_path}")
                return [(item['recipe_id'], item['tfidf_vec']) for item in cached_data]
            except Exception as e:
                print(f"Error loading cache, recomputing TF-IDF: {e}")
        
        # Compute TF-IDF and save to cache
        print("Computing TF-IDF vectors (this may take a while for large datasets)...")
        recipe_tfidf = compute_tfidf_for_recipes(self.json_path)
        
        # Prepare cache data
        cache_data = [{
            'recipe_id': r_id,
            'tfidf_vec': tfidf_vec
        } for r_id, tfidf_vec in recipe_tfidf]
        
        # Save to cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"Saved TF-IDF data to cache: {cache_path}")
        
        return recipe_tfidf
    
    def _build_vocabulary(self) -> set:
        """Build vocabulary set"""
        vocabulary = set()
        for _, tfidf_vec in self.recipe_tfidf:
            vocabulary.update(tfidf_vec.keys())
        return vocabulary
    
    def _cosine_similarity(self, vec_a: dict, vec_b: dict) -> float:
        """Compute cosine similarity between two TF-IDF vectors"""
        # Get all unique terms
        terms = set(vec_a.keys()).union(set(vec_b.keys()))
        
        # Compute dot product
        dot_product = sum(vec_a.get(term, 0) * vec_b.get(term, 0) for term in terms)
        
        # Compute magnitudes
        mag_a = math.sqrt(sum(val**2 for val in vec_a.values()))
        mag_b = math.sqrt(sum(val**2 for val in vec_b.values()))
        
        # Avoid division by zero
        if mag_a * mag_b == 0:
            return 0.0
        
        return dot_product / (mag_a * mag_b)
    
    def search(self, query: str, top_n: int = 5) -> list:
        """
        Search recipes by ingredients
        :param query: Ingredients query string (e.g., "chicken garlic")
        :param top_n: Number of results to return
        :return: List of matching recipes, sorted by relevance
        """
        # Tokenize and compute TF for query
        query_tokens = tokenize(query)
        query_tf = compute_tf(query_tokens)
        
        # Build query vector using terms in vocabulary
        query_vec = {
            term: query_tf.get(term, 0)
            for term in query_tokens if term in self.vocabulary
        }
        
        # Compute similarity scores
        scores = []
        for recipe_id, recipe_vec in self.recipe_tfidf:
            similarity = self._cosine_similarity(query_vec, recipe_vec)
            if similarity > 0:
                scores.append((recipe_id, similarity))
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = []
        for recipe_id, score in scores[:top_n]:
            recipe = self.recipes[recipe_id]
            results.append({
                'id': recipe_id,
                'title': recipe['title'],
                'score': round(score, 4),
                'ingredients': [ing['name'] for ing in recipe['ingredients']][:8],  # Show max 8 ingredients
                'directions': recipe.get('directions', [])[:3]  # Show first 3 steps
            })
        
        return results

def main():
    # Initialize with your JSON file path
    json_path = "/Users/danielliu/Desktop/Spring 2025/CSCI-UA 469/Project/cleaned_data/train.json"
    engine = RecipeSearchEngine(json_path)
    
    print("Recipe Search Engine - Find recipes by ingredients")
    print("Enter ingredients you want to use (space separated), or 'q' to quit")
    
    while True:
        user_input = input("\nSearch recipes (enter ingredients): ").strip()
        
        if user_input.lower() == 'q':
            break
        
        if not user_input:
            print("Please enter valid ingredients!")
            continue
        
        results = engine.search(user_input)
        
        if not results:
            print("No matching recipes found.")
        else:
            print(f"\nFound {len(results)} matching recipes:")
            for i, recipe in enumerate(results, 1):
                print(f"\n{i}. {recipe['title']} (Match score: {recipe['score']:.3f})")
                print("Main ingredients:")
                for ing in recipe['ingredients']:
                    print(f"  - {ing}")
                if recipe['directions']:
                    print("Cooking steps:")
                    for step in recipe['directions']:
                        print(f"  - {step}")

if __name__ == "__main__":
    main()