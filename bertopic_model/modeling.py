from bertopic import BERTopic
import pickle

def run_bertopic(docs, clusters, vectorizer_model, ctfidf_model):
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(docs, y=clusters)
    with open("output/bertopic_model.pkl", "wb") as f:
        pickle.dump(topic_model, f)
    return topic_model, topics, probs