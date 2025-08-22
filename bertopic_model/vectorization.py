from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import pickle


def get_vectorizer_and_ctfidf(docs):
    vectorizer = CountVectorizer(min_df=3, max_df=0.8, ngram_range=(1,2), stop_words=None)
    ctfidf = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    
    with open("output/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    return vectorizer, ctfidf
