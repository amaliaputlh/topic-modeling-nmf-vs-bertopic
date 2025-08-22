from sentence_transformers import SentenceTransformer
import numpy as np
import random
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def get_embeddings(docs, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=True)
    np.save("output/embeddings.npy", embeddings)
    return embeddings
