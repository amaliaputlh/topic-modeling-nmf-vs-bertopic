import pickle
import numpy as np

def calculate_topic_diversity(topic_words):
    unique_words = set(word for topic in topic_words for word in topic)
    total_words = sum(len(topic) for topic in topic_words)
    return len(unique_words) / total_words if total_words > 0 else 0

def evaluate_bertopic_diversity(model_path):
    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)
    topics = topic_model.get_topics()
    top_words = [ [word for word, _ in topic_words] for topic_id, topic_words in topics.items() if topic_id != -1]
    return calculate_topic_diversity(top_words)

def evaluate_nmf_diversity(topic_keywords):
    return calculate_topic_diversity(topic_keywords)
