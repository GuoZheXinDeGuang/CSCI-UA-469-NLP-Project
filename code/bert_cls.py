import torch, numpy as np
from transformers import BertTokenizer, BertModel
from torch.nn.functional import normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BertCLS:
    """Vanilla BERT sentence/vector encoder using the [CLS] token."""
    dim = 768       

    def __init__(self, model_name="bert-base-uncased"):
        self.tok   = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(DEVICE).eval()

    @torch.inference_mode()
    def encode(self, texts, batch_size=64):
        outs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(
                texts[i:i+batch_size],
                padding=True, truncation=True, return_tensors="pt"
            ).to(DEVICE)
            # CLS vector = first hidden‑state token
            h_cls = self.model(**enc).last_hidden_state[:, 0, :]   # [B,768]
            h_cls = normalize(h_cls, dim=1)                        # cosine ready
            outs.append(h_cls.cpu())
        return torch.cat(outs).numpy()
