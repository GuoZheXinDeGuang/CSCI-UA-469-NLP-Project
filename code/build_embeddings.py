import json, numpy as np, argparse
from pathlib import Path
from embeddings.bert_cls       import BertCLS        # or SentenceBERT
# from embeddings.sentence_bert import SentenceBERT  

def main(path_in, path_out=None, dump_npy=False):
    data = json.loads(Path(path_in).read_text("utf‑8"))
    embedder = BertCLS()            # or SentenceBERT()
    print(f"Using {embedder.__class__.__name__} | dim={embedder.dim}")

    flat_txt, idx_map = [], []      # idx_map: (recipe_i, ingredient_i)
    for r_i, rec in enumerate(data):
        for i_i, ing in enumerate(rec["ingredients"]):
            flat_txt.append(ing["name"])
            idx_map.append((r_i, i_i))

    vecs = embedder.encode(flat_txt)     # [N, dim]

    for (r_i, i_i), v in zip(idx_map, vecs):
        data[r_i]["ingredients"][i_i]["embedding"] = v.tolist()

    dst = path_out or path_in
    Path(dst).write_text(json.dumps(data, indent=2))
    print(f"✔ embeddings written → {dst}")

    if dump_npy:
        np.save(Path(dst).with_suffix(".npy"), vecs)
        print(f"✔ raw vectors → {dst[:-5]}.npy")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  required=True,  dest="inp")
    ap.add_argument("--out", required=False, dest="out")
    ap.add_argument("--npy", action="store_true", help="also dump .npy")
    main(**vars(ap.parse_args()))
