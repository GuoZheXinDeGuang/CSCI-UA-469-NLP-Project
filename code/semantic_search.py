from __future__ import annotations
import argparse, json, os, sys, numpy as np, faiss, torch
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
class _Embedder:
    """Thin wrapper around Sentence‑Transformers (unit‑normalised output)."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device=DEVICE)

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: 128) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,   # crucial → unit vectors
            device=DEVICE,
            show_progress_bar=True
        ).cpu().numpy().astype("float32")          # FAISS likes float32


# ──────────────────────────────────────────────────────────────────────────────
class SemanticSearch:
    """
    Lightweight runtime container:
        • loads FAISS index (+ vectors if needed)
        • embeds queries on the fly (Sentence‑BERT)
        • returns (ingredient_name, cosine_score) tuples
    """
    def __init__(self,
                 index_path: str,
                 names_path: str,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.names: List[str] = json.loads(Path(names_path).read_text())
        self.dim = len(json.loads(Path(names_path).read_text())[0]) \
            if self.names else 384
        self.index = faiss.read_index(index_path)
        self.embedder = _Embedder(model_name)

    # ----------------------------------------------------------------------
    def query(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        vec = self.embedder.encode([text])[0]                    # shape [d]
        # FAISS IndexFlatIP expects inner‑product (=cosine because unit vecs)
        scores, idxs = self.index.search(vec[None, :], k)
        return [(self.names[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]


# ──────────────────────────────────────────────────────────────────────────────
def _build_index(recipes_json: str,
                 emb_path: str,
                 names_path: str,
                 index_path: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 128) -> None:
    """Encode every ingredient once, save vectors + FAISS index."""
    data = json.loads(Path(recipes_json).read_text("utf‑8"))
    names = [ing["name"]
             for rec in data
             for ing in rec["ingredients"]]
    print(f"[build] Found {len(names):,} ingredient strings")

    embedder = _Embedder(model_name)
    vecs = embedder.encode(names, batch_size=batch_size)         # [N,d]

    # save raw vectors + names
    np.save(emb_path, vecs)
    Path(names_path).write_text(json.dumps(names, indent=2))
    print(f"[build] Vectors → {emb_path} | Names → {names_path}")

    # build & save FAISS index
    index = faiss.IndexFlatIP(vecs.shape[1])    # cosine (inner‑product)
    index.add(vecs)
    faiss.write_index(index, index_path)
    print(f"[build] FAISS index ({vecs.shape[0]} vecs) → {index_path}")


# ──────────────────────────────────────────────────────────────────────────────
def _cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    # build sub‑command
    b = sub.add_parser("build", help="one‑time index builder")
    b.add_argument("--recipes",   required=True)
    b.add_argument("--emb-path",  required=True)
    b.add_argument("--names-path",required=True)
    b.add_argument("--index-path",required=True)
    b.add_argument("--model",     default="all-MiniLM-L6-v2")

    # interactive search (quick test)
    q = sub.add_parser("shell", help="interactive query")
    q.add_argument("--index-path",required=True)
    q.add_argument("--names-path",required=True)
    q.add_argument("--model",     default="all-MiniLM-L6-v2")

    args = ap.parse_args()

    if args.mode == "build":
        _build_index(args.recipes, args.emb_path, args.names_path,
                     args.index_path, args.model)
    elif args.mode == "shell":
        ss = SemanticSearch(args.index_path, args.names_path, args.model)
        print("Semantic search console – enter text, q to quit")
        while True:
            txt = input(">>> ").strip()
            if txt.lower() in {"q", "quit"}: break
            for name, score in ss.query(txt, k=10):
                print(f"{score:5.3f}  {name}")
    else:
        ap.print_help(); sys.exit(1)


if __name__ == "__main__":
    _cli()
