from hdbscan import HDBSCAN
import pickle

def cluster(reduced_embeddings):
    hdb = HDBSCAN(min_cluster_size=13, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    clusters = hdb.fit_predict(reduced_embeddings)
    with open("output/clusters.pkl", "wb") as f:
        pickle.dump(clusters, f)
    return clusters