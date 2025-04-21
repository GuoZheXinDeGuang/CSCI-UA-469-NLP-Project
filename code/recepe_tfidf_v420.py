import json, os, math, re, pickle, time, difflib
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from spellchecker import SpellChecker
from semantic_search import SemanticSearch          
# pip install sentence-transformers faiss-cpu pyspellchecker

class OptimizedRecipeSearchEngine:
    """Hybrid recipe search: TF‑IDF  +  dense ingredient similarity."""

    # Dietary restriction → forbidden ingredient keywords
    DIETARY_FORBIDDEN = {
        "vegan": {
            "egg", "milk", "butter", "cheese", "cream", "yogurt",
            "honey", "meat", "chicken", "beef", "pork", "fish", "shrimp"
        },
        "gluten_free": {
            "flour", "wheat", "barley", "rye", "bread", "pasta",
            "noodle", "crumbs", "couscous", "semolina"
        },
        "diabetic": {
            "sugar", "honey", "maple", "syrup", "jam", "jelly",
            "brown", "powdered"
        },
    }

    # --------------------------------------------------------------
    def __init__(self,
                 json_path: str  = "cleaned_data/train.json",
                 cache_dir: str  = "tfidf_cache",
                 title_weight: float = 2.0,
                 ingredients_weight: float = 1.5,
                 directions_weight: float  = 0.5,
                 default_mode: str = "tfidf"  # "tfidf" | "semantic" | "hybrid"
                 ):
        self.json_path   = json_path
        self.cache_dir   = cache_dir
        self.title_weight        = title_weight
        self.ingredients_weight  = ingredients_weight
        self.directions_weight   = directions_weight
        self.default_mode        = default_mode

        # typo correction helper
        self.spell = SpellChecker(distance=1)

        # basic stop‑word list
        self.stopwords = {
            'a','an','the','and','or','but','in','on','of','to','from',
            'with','by','for','at','is','are','cup','cups','tablespoon',
            'tablespoons','teaspoon','teaspoons','recipe','about','until',
            'tsp','tsps'
        }

        # load recipes JSON → dict[id → recipe]
        self.recipes = self._load_recipes()

        # TF‑IDF vectors (sparse) ------------------------------------------------
        t0 = time.time()
        self.recipe_tfidf, self.vocabulary = self._load_or_compute_tfidf()
        print(f"[TF‑IDF] ready in {time.time()-t0:.1f}s | vocab={len(self.vocabulary)}")

        # Dense semantic engine (Sentence‑BERT + FAISS) -------------------------
        self.semantic = SemanticSearch(
            index_path ="cleaned_data/ingredient.index",
            names_path ="cleaned_data/ingredient_names.json",
            model_name="all-MiniLM-L6-v2"
        )

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    def _load_recipes(self) -> Dict[str, dict]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {r['id']: r for r in data}

    def _get_cache_path(self) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        base = os.path.basename(self.json_path)
        cfg  = f"_tw{self.title_weight:.1f}_iw{self.ingredients_weight:.1f}_dw{self.directions_weight:.1f}"
        return os.path.join(self.cache_dir, f"{base}{cfg}.pkl")

    def _tokenize(self, text: str) -> List[str]:
        if not text: return []
        cleaned = re.sub(r'[^a-z\s]', ' ', text.lower())
        return [t for t in cleaned.split() if t and t not in self.stopwords]

    def _correct_spelling(self, tokens: List[str]) -> List[str]:
        fixed = []
        for t in tokens:
            corr = self.spell.correction(t)
            if corr and corr != t:
                fixed.append(corr)
            else:
                close = difflib.get_close_matches(t, self.vocabulary, n=1, cutoff=0.7)
                fixed.append(close[0] if close else t)
        return fixed

    def _extract_sections(self, rec: dict) -> Dict[str, str]:
        title = rec.get('title', '')
        ings  = ' '.join(i['name'] if isinstance(i,dict) else str(i)
                         for i in rec.get('ingredients', []))
        dirs  = ' '.join(rec.get('directions', []))
        return {'title': title, 'ingredients': ings, 'directions': dirs}

    # ---------------- TF‑IDF build / load ------------------------------------
    def _load_or_compute_tfidf(self) -> Tuple[List[Tuple[str, Dict[str, float]]], Set[str]]:
        cache = self._get_cache_path()
        if os.path.exists(cache):
            try:
                with open(cache, 'rb') as f: return pickle.load(f)
            except Exception: pass     # fall through if cache corrupted

        tokenized, df = {}, Counter()
        total = len(self.recipes)

        # tokenize & doc‑freq
        for rid, rec in self.recipes.items():
            secs = self._extract_sections(rec)
            toks = {s: self._tokenize(txt) for s, txt in secs.items()}
            tokenized[rid] = toks
            for t in set(t for lst in toks.values() for t in lst):
                df[t] += 1

        vocab = {t for t,f in df.items() if 3 <= f <= total*0.7}
        idf   = {t: math.log(total/df[t]) for t in vocab}

        tfidf = []
        for rid, secs in tokenized.items():
            vec = {}
            for sec, toks in secs.items():
                w = getattr(self, f"{sec}_weight")
                L = max(1, len(toks))
                for t in toks:
                    if t in vocab:
                        vec[t] = vec.get(t,0) + (toks.count(t)/L)*idf[t]*w
            tfidf.append((rid, vec))

        with open(cache, 'wb') as f: pickle.dump((tfidf, vocab), f)
        return tfidf, vocab

    def _cosine(self, a: Dict[str,float], b: Dict[str,float]) -> float:
        common = set(a)&set(b)
        if not common: return 0.0
        dot = sum(a[k]*b[k] for k in common)
        mag = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
        return dot/mag if mag else 0.0

    def _check_diet(self, rec: dict, diet: str) -> bool:
        bad = self.DIETARY_FORBIDDEN.get(diet, set())
        text = ' '.join(i['name'] if isinstance(i,dict) else str(i)
                        for i in rec.get('ingredients', [])).lower()
        return not any(w in text for w in bad)

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def search(self,
               query: str,
               top_n: int = 5,
               debug: bool = False,
               dietary: Optional[str] = None,
               mode:   Optional[str] = None      # "tfidf" | "semantic" | "hybrid"
               ) -> List[dict]:

        mode = mode or self.default_mode

        # ---------- dense semantic only -----------------------------------
        if mode == "semantic":
            dens = self.semantic.query(query, k=top_n)
            return [{"name": n, "score": round(s,4)} for n,s in dens]

        # ---------- dense map (for hybrid) --------------------------------
        dense_map = {}
        if mode == "hybrid":
            dense_map = {n:s for n,s in self.semantic.query(query, k=50)}

        # ---------- sparse branch -----------------------------------------
        toks = self._tokenize(query)
        corr = self._correct_spelling(toks)
        if debug:
            print("tokens:", toks)
            print("corrected:", corr)

        qvec = {}
        for t in corr:
            if t in self.vocabulary:
                qvec[t] = qvec.get(t,0) + 1/len(corr) if corr else 0
        if not qvec: return []

        # diet filter
        cand = ((rid,vec) for rid,vec in self.recipe_tfidf
                if dietary is None or self._check_diet(self.recipes[rid], dietary))

        scored = [(rid, self._cosine(qvec, vec)) for rid,vec in cand if self._cosine(qvec, vec)>0]
        scored.sort(key=lambda x: x[1], reverse=True)

        # ---------- fuse dense score --------------------------------------
        if mode == "hybrid":
            fused=[]
            for rid, s_t in scored:
                ing_names = [i['name'] for i in self.recipes[rid]['ingredients']]
                s_d = max(dense_map.get(n,0) for n in ing_names)
                fused.append((rid, 0.6*s_t + 0.4*s_d))
            fused.sort(key=lambda x:x[1], reverse=True)
            scored = fused

        # ---------- format output -----------------------------------------
        results=[]
        for rid, sc in scored[:top_n]:
            rec = self.recipes[rid]
            ings = [i['name'] for i in rec.get('ingredients', [])][:8]
            results.append({
                "id":   rid,
                "title": rec.get('title',''),
                "score": round(sc,4),
                "ingredients": ings,
                "steps": rec.get('directions', [])
            })
        return results

# ---------------------------------------------------------------------
def main():
    engine = OptimizedRecipeSearchEngine(default_mode="tfidf")
    print("\n=== Interactive Search ===")
    while True:
        q = input("\nSearch (q=quit): ").strip()
        if not q or q.lower()=='q': break
        mode = input("Mode tfidf / semantic / hybrid (enter=default): ").strip() or None
        diet = input("Diet (vegan/gluten_free/diabetic or enter): ").strip() or None
        hits = engine.search(q, top_n=5, debug=False, dietary=diet, mode=mode)
        print(json.dumps(hits, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
