import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def get_top_words_per_topic(topic_source, top_k=10, method="bertopic"):
    if method == "bertopic":
        topics = topic_source.get_topics()
        return [
            [word for word, _ in topic_words[:top_k]]
            for topic_id, topic_words in topics.items()
            if topic_id != -1
        ]
    elif method == "nmf":
        return [topic[:top_k] for topic in topic_source]
    else:
        raise ValueError("Method harus 'bertopic' atau 'nmf'")


def get_topic_embeddings(topic_words, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    topic_embeddings = []
    for topic in topic_words:
        word_embeddings = model.encode(topic)
        topic_vector = np.mean(word_embeddings, axis=0)
        topic_embeddings.append(topic_vector)
    return np.array(topic_embeddings)


def compute_similarity_matrix(topic_embeddings):
    return cosine_similarity(topic_embeddings)


def find_similar_pairs(sim_matrix, threshold=0.85):
    pairs = []
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):
            sim = sim_matrix[i, j]
            if sim > threshold:
                pairs.append((i, j, round(sim, 4)))
    return pairs
