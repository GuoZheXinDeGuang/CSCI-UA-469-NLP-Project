import json
import os
import math
import re
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Set
import time

class OptimizedRecipeSearchEngine:
    def __init__(self, json_path: str = "cleaned_data/train.json", 
                 cache_dir: str = "tfidf_cache",
                 title_weight: float = 2.0,
                 ingredients_weight: float = 1.5,
                 directions_weight: float = 0.5):
        """
        Initialize recipe search engine with performance-optimized TF-IDF
        
        Args:
            json_path: Path to recipes JSON file
            cache_dir: Cache directory name
            title_weight: Weight for terms in recipe title
            ingredients_weight: Weight for terms in ingredients
            directions_weight: Weight for terms in directions
        """
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.title_weight = title_weight
        self.ingredients_weight = ingredients_weight
        self.directions_weight = directions_weight
        
        # Simple cooking stopwords (smaller set for performance)
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'of', 'to', 'from',
            'with', 'by', 'for', 'at', 'is', 'are', 'cup', 'cups', 'tablespoon', 
            'tablespoons', 'teaspoon', 'teaspoons', 'recipe', 'about', 'until'
        }
        
        # Load recipes
        self.recipes = self._load_recipes()
        
        # Compute or load TF-IDF
        start_time = time.time()
        self.recipe_tfidf, self.vocabulary = self._load_or_compute_tfidf()
        elapsed = time.time() - start_time
        print(f"TF-IDF ready in {elapsed:.2f} seconds")
        print(f"Vocabulary size: {len(self.vocabulary)} terms")
    
    def _load_recipes(self) -> Dict[str, dict]:
        """Load recipes and create ID-to-recipe mapping"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        return {recipe['id']: recipe for recipe in recipes}
    
    def _get_cache_path(self) -> str:
        """Get cache file path with configuration parameters"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Configuration string for cache filename
        config_str = f"_tw{self.title_weight:.1f}_iw{self.ingredients_weight:.1f}_dw{self.directions_weight:.1f}"
        
        base_name = os.path.basename(self.json_path)
        return os.path.join(self.cache_dir, f"{base_name}{config_str}.pickle")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Fast tokenization with minimal processing
        """
        if not text:
            return []
            
        # Convert to lowercase and replace non-alphabetic chars
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Split by whitespace and filter stopwords
        return [t for t in text.split() if t and t not in self.stopwords]
    
    def _extract_recipe_text(self, recipe: dict) -> Dict[str, str]:
        """Extract text from different parts of the recipe"""
        # Extract title
        title = recipe.get('title', '')
        
        # Extract ingredients (fast method)
        ingredients_text = ' '.join(
            ing.get('name', '') if isinstance(ing, dict) else str(ing)
            for ing in recipe.get('ingredients', [])
        )
        
        # Extract directions
        directions_text = ' '.join(recipe.get('directions', []))
        
        return {
            'title': title,
            'ingredients': ingredients_text,
            'directions': directions_text
        }
    
    def _load_or_compute_tfidf(self) -> Tuple[List[Tuple[str, Dict[str, float]]], Set[str]]:
        """Load or compute TF-IDF with performance optimizations"""
        cache_path = self._get_cache_path()
        
        # Try loading from cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Loaded TF-IDF from cache: {cache_path}")
                return cached_data
            except Exception as e:
                print(f"Error loading cache: {e}, recomputing...")
        
        print("Computing TF-IDF vectors (optimized version)...")
        start_time = time.time()
        
        # Preprocess all recipes and collect document frequencies
        tokenized_sections = {}
        doc_freq = Counter()
        all_terms = set()
        total_recipes = len(self.recipes)
        
        # Process in batches for progress reporting
        batch_size = max(1, total_recipes // 10)
        
        for i, (recipe_id, recipe) in enumerate(self.recipes.items()):
            # Progress reporting
            if i % batch_size == 0:
                progress = i / total_recipes * 100
                elapsed = time.time() - start_time
                print(f"Processing recipes: {progress:.1f}% ({i}/{total_recipes}), time: {elapsed:.1f}s")
            
            # Extract text from recipe sections
            sections = self._extract_recipe_text(recipe)
            
            # Tokenize each section
            sections_tokens = {
                section: self._tokenize(text)
                for section, text in sections.items()
            }
            
            # Store tokenized content
            tokenized_sections[recipe_id] = sections_tokens
            
            # Update document frequency (count each term once per recipe)
            unique_terms = set()
            for section_tokens in sections_tokens.values():
                unique_terms.update(section_tokens)
            
            for term in unique_terms:
                doc_freq[term] += 1
                all_terms.add(term)
        
        print(f"Initial terms count: {len(all_terms)}")
        
        # Filter vocabulary (keep terms appearing in 3-70% of documents)
        min_doc_freq = 3
        max_doc_freq = total_recipes * 0.7
        
        vocabulary = {
            term for term, freq in doc_freq.items()
            if min_doc_freq <= freq <= max_doc_freq
        }
        
        print(f"Filtered vocabulary size: {len(vocabulary)}")
        
        # Compute IDF values for filtered vocabulary
        idf_values = {}
        for term in vocabulary:
            idf_values[term] = math.log(total_recipes / doc_freq[term])
        
        # Compute TF-IDF vectors
        recipe_tfidf = []
        
        for recipe_id, sections in tokenized_sections.items():
            tfidf_vector = {}
            
            # Process title
            for term in sections['title']:
                if term in vocabulary:
                    # Term count / section length
                    tf = sections['title'].count(term) / max(1, len(sections['title']))
                    tfidf_vector[term] = tfidf_vector.get(term, 0) + tf * idf_values[term] * self.title_weight
            
            # Process ingredients
            for term in sections['ingredients']:
                if term in vocabulary:
                    tf = sections['ingredients'].count(term) / max(1, len(sections['ingredients']))
                    tfidf_vector[term] = tfidf_vector.get(term, 0) + tf * idf_values[term] * self.ingredients_weight
            
            # Process directions (with lower weight)
            for term in sections['directions']:
                if term in vocabulary:
                    tf = sections['directions'].count(term) / max(1, len(sections['directions']))
                    tfidf_vector[term] = tfidf_vector.get(term, 0) + tf * idf_values[term] * self.directions_weight
            
            recipe_tfidf.append((recipe_id, tfidf_vector))
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump((recipe_tfidf, vocabulary), f)
        
        print(f"TF-IDF computation completed in {time.time() - start_time:.2f} seconds")
        print(f"Saved to cache: {cache_path}")
        
        return recipe_tfidf, vocabulary
    
    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Fast cosine similarity calculation using only common terms"""
        # Find common terms (for performance)
        common_terms = set(vec_a.keys()) & set(vec_b.keys())
        
        if not common_terms:
            return 0.0
        
        # Calculate dot product for common terms only
        dot_product = sum(vec_a[term] * vec_b[term] for term in common_terms)
        
        # Calculate magnitudes
        mag_a = math.sqrt(sum(vec_a[term] ** 2 for term in vec_a))
        mag_b = math.sqrt(sum(vec_b[term] ** 2 for term in vec_b))
        
        if mag_a * mag_b == 0:
            return 0.0
        
        return dot_product / (mag_a * mag_b)
    
    def search(self, query: str, top_n: int = 5) -> List[dict]:
        """Search recipes by query string"""
        # Tokenize and filter query
        query_tokens = self._tokenize(query)
        
        # Create query vector
        query_vec = {}
        for term in query_tokens:
            if term in self.vocabulary:
                query_vec[term] = query_vec.get(term, 0) + 1
        
        # Normalize query vector
        query_length = len(query_tokens)
        if query_length > 0:
            for term in query_vec:
                query_vec[term] /= query_length
        
        # If no valid query terms, return empty results
        if not query_vec:
            return []
        
        # Calculate similarities
        scores = []
        for recipe_id, recipe_vec in self.recipe_tfidf:
            similarity = self._cosine_similarity(query_vec, recipe_vec)
            if similarity > 0:
                scores.append((recipe_id, similarity))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = []
        for recipe_id, score in scores[:top_n]:
            recipe = self.recipes[recipe_id]
            
            # Format ingredients
            ingredients = []
            for ing in recipe.get('ingredients', []):
                if isinstance(ing, dict):
                    ingredients.append(ing.get('name', ''))
                else:
                    ingredients.append(str(ing))
            
            results.append({
                'id': recipe_id,
                'title': recipe.get('title', ''),
                'score': round(score, 4),
                'ingredients': ingredients[:8],  # Show max 8 ingredients
                'directions': recipe.get('directions', [])[:3]  # Show first 3 steps
            })
        
        return results

    def get_similar_recipes(self, recipe_id: str, top_n: int = 5) -> List[dict]:
        """Find recipes similar to the specified recipe"""
        if recipe_id not in self.recipes:
            return []
        
        # Find TF-IDF vector for the reference recipe
        reference_vec = None
        for rid, vec in self.recipe_tfidf:
            if rid == recipe_id:
                reference_vec = vec
                break
        
        if not reference_vec:
            return []
        
        # Calculate similarities
        scores = []
        for rid, vec in self.recipe_tfidf:
            if rid != recipe_id:  # Skip the reference recipe
                similarity = self._cosine_similarity(reference_vec, vec)
                if similarity > 0:
                    scores.append((rid, similarity))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = []
        for rid, score in scores[:top_n]:
            recipe = self.recipes[rid]
            results.append({
                'id': rid,
                'title': recipe.get('title', ''),
                'score': round(score, 4),
                'ingredients': [ing.get('name', '') if isinstance(ing, dict) else str(ing)
                               for ing in recipe.get('ingredients', [])][:5]
            })
        
        return results


def main():
    """Main function to run the search engine interactively"""
    # Initialize with optimized parameters
    engine = OptimizedRecipeSearchEngine(
        json_path="cleaned_data/train.json",
        title_weight=2.0,
        ingredients_weight=1.5,
        directions_weight=0.5
    )
    
    print("\n===== Optimized Recipe Search Engine =====")
    print("Search by ingredients or keywords")
    print("Commands:")
    print("  'q' to quit")
    print("  'similar:[recipe_id]' to find similar recipes")
    
    while True:
        user_input = input("\nSearch recipes: ").strip()
        
        if not user_input or user_input.lower() == 'q':
            print("Goodbye!")
            break
        
        # Check for special commands
        if user_input.startswith('similar:'):
            recipe_id = user_input[8:].strip()
            results = engine.get_similar_recipes(recipe_id)
            if not results:
                print(f"No recipe found with ID '{recipe_id}' or no similar recipes.")
                continue
                
            print(f"\nRecipes similar to '{engine.recipes.get(recipe_id, {}).get('title', recipe_id)}':")
            
        else:
            # Regular search
            results = engine.search(user_input)
        
        # Display results
        if not results:
            print("No matching recipes found.")
        else:
            print(f"\nFound {len(results)} matching recipes:")
            for i, recipe in enumerate(results, 1):
                print(f"\n{i}. {recipe['title']} (Match score: {recipe['score']:.3f})")
                
                print("Main ingredients:")
                for ing in recipe['ingredients'][:5]:
                    print(f"  - {ing}")
                
                if recipe.get('directions'):
                    print("First cooking step:")
                    print(f"  - {recipe['directions'][0]}")


if __name__ == "__main__":
    main()