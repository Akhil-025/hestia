import hashlib
class SentenceTransformer:
    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
    def encode(self, batch, show_progress_bar=False, normalize_embeddings=True):
        out = []
        for text in batch:
            h = hashlib.sha256(text.encode()).digest()
            vec = [b / 255.0 for b in h[:8]]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return _FakeArr(out)
class _FakeArr(list):
    def tolist(self): return list(self)
