import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import math

def generate_bertopic_barchart(topic_model):
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info['Topic'] != -1]
    topics = topic_info['Name'].tolist()
    counts = topic_info['Count'].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(topics))
    ax.barh(y_pos, counts, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics)
    ax.invert_yaxis()
    ax.set_xlabel("Jumlah Dokumen")
    ax.set_title("Distribusi Dokumen per Topik - BERTopic")
    plt.tight_layout()
    return fig

def generate_nmf_barchart(W):
    topic_distribution = pd.Series(W.argmax(axis=1)).value_counts().sort_index()
    topics = [f"Topik {i}" for i in topic_distribution.index]
    counts = topic_distribution.values

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(topics))
    ax.barh(y_pos, counts, color='salmon')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics)
    ax.invert_yaxis()
    ax.set_xlabel("Jumlah Dokumen")
    ax.set_title("Distribusi Dokumen per Topik - NMF")
    plt.tight_layout()
    return fig


def generate_bertopic_table_and_wordcloud(topic_model):
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info['Topic'] != -1]

    df_table = topic_info[["Topic", "Name", "Count"]]
    
    wordclouds = {}
    for topic_id in df_table["Topic"]:
        words = topic_model.get_topic(topic_id)
        if words:
            word_freq = {word: weight for word, weight in words}
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            wordclouds[topic_id] = wordcloud

    return df_table, wordclouds


def generate_nmf_table_and_wordcloud(topic_keywords):
    df_table = pd.DataFrame({
        "Topik": [f"Topik {i}" for i in range(len(topic_keywords))],
        "Kata Kunci": [", ".join(words) for words in topic_keywords]
    })

    wordclouds = {}
    for i, words in enumerate(topic_keywords):
        word_freq = {word: 1 for word in words}
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
        wordclouds[i] = wordcloud

    return df_table, wordclouds


def plot_wordcloud_grid(wordclouds_dict, title_prefix="Topik"):
    num_items = min(16, len(wordclouds_dict))
    n_cols = 4
    n_rows = math.ceil(num_items / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < num_items:
            topic_id = list(wordclouds_dict.keys())[i]
            ax.imshow(wordclouds_dict[topic_id], interpolation='bilinear')
            ax.set_title(f"{title_prefix} {topic_id}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    return fig


from bertopic import BERTopic
import plotly.graph_objects as go

def generate_bertopic_intertopic_map(topic_model: BERTopic) -> go.Figure:
    """
    Menghasilkan intertopic distance map dari model BERTopic.

    Args:
        topic_model (BERTopic): Model BERTopic yang sudah dilatih.

    Returns:
        go.Figure: Visualisasi interaktif jarak antar topik.
    """
    try:
        fig = topic_model.visualize_topics()
        return fig
    except Exception as e:
        print(f"Gagal membuat intertopic map: {e}")
        return None

import pandas as pd

def show_example_docs_per_topic_streamlit(docs, topic_assignments, target_topics=[9, 10], n_examples=5):
    """
    Menampilkan beberapa contoh dokumen dari topik yang ditentukan di Streamlit.
    """
    df = pd.DataFrame({
        "Dokumen": docs,
        "Topik": topic_assignments
    })

    output = ""
    for topic_id in target_topics:
        output += f"### Topik {topic_id} - Contoh Dokumen:\n"
        contoh = df[df["Topik"] == topic_id].sample(
            n=min(n_examples, df["Topik"].value_counts().get(topic_id, 0)), random_state=42)
        for i, row in contoh.iterrows():
            output += f"- {row['Dokumen'][:1000]}...\n"
    return output


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_similarity_between_topics(topic_model, topic_ids=[9, 10]):
    """
    Menghitung cosine similarity antara dua topik berdasarkan representasi vector topiknya.
    """
    topic_representations = topic_model.topic_embeddings_  # Ambil semua embedding topik
    
    if topic_representations is None:
        raise ValueError("Model tidak memiliki topic_embeddings_ (pastikan model dilatih dengan parameter calculate_probabilities=True dan embedding disimpan).")

    vec_1 = topic_representations[topic_ids[0]]
    vec_2 = topic_representations[topic_ids[1]]

    similarity = cosine_similarity([vec_1], [vec_2])[0][0]
    print(f"Cosine similarity antara Topik {topic_ids[0]} dan Topik {topic_ids[1]}: {similarity:.4f}")
    return similarity
