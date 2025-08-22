from sklearn.decomposition import NMF
import numpy as np
import pickle

def run_nmf(X_tfidf, feature_names, docs, num_topics, num_words=10):
    nmf_model = NMF(n_components=num_topics, random_state=42)
    W = nmf_model.fit_transform(X_tfidf)
    H = nmf_model.components_

    topic_keywords = []
    for topic in H:
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topic_keywords.append(top_words)

    with open("output/nmf_topics_ml_nt21.pkl", "wb") as f:
        pickle.dump((topic_keywords, W, docs), f)  

    return nmf_model, topic_keywords, W

