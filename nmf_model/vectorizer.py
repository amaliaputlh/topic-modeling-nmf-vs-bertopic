from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from scipy import sparse

def generate_tfidf_matrix(docs, min_df=3, max_df=0.8):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1,2), sublinear_tf=True)
    X_tfidf = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    # Simpan matrix dan fitur
    pd.DataFrame(X_tfidf.toarray(), columns=feature_names).to_csv("output/tf_idf_matrix.csv", index=False)
    np.savetxt("output/tf_idf_features.txt", feature_names, fmt="%s")
    sparse.save_npz("output/dtm.npz", X_tfidf)

    return X_tfidf, feature_names

