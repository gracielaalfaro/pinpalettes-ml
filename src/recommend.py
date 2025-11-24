import numpy as np
from sklearn.neighbors import NearestNeighbors
from .palette import extract_palette

class CLIPRecommender:
    def __init__(self, embeddings, image_paths):
        self.embeddings = embeddings
        self.image_paths = np.array(image_paths)
        self.nn = NearestNeighbors(metric="cosine", n_neighbors=20)
        self.nn.fit(embeddings)

    def query(self, query_embedding, top_k=8):
        dists, idxs = self.nn.kneighbors(query_embedding.reshape(1,-1), n_neighbors=top_k)
        return self.image_paths[idxs[0]].tolist(), (1 - dists[0]).tolist()

def rerank_by_palette(query_img, candidate_paths, top_k=8):
    """
    Optional rerank by palette distance.
    """
    q_rgb, _, _ = extract_palette(query_img, k=5)

    def pal_dist(path):
        from PIL import Image
        c_rgb, _, _ = extract_palette(Image.open(path).convert("RGB"), k=5)
        return np.linalg.norm(q_rgb[:5] - c_rgb[:5])

    scored = sorted(candidate_paths, key=pal_dist)
    return scored[:top_k]
