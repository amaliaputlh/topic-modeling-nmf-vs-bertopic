import numpy as np
from umap import UMAP
import pickle

def reduce_dimensions(embeddings):
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    reduced = umap_model.fit_transform(embeddings)
    with open("output/reduced_embeddings.pkl", "wb") as f:
        pickle.dump(reduced, f)
    return reduced
