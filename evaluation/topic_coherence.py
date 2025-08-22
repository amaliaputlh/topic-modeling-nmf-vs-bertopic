import os
import pickle
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

from bertopic import BERTopic
import streamlit as st
import ast
import time
import random


def evaluate_bertopic_model(doc_path, sample_size=None):

    start_time = time.time()

    with open(doc_path, "rb") as f:
        documents = pickle.load(f)

    if sample_size is not None and sample_size < len(documents):
        documents = random.sample(documents, sample_size)

    topic_model = st.session_state.get("bertopic_model")

    tokenized_docs = []
    for doc in documents:
        if isinstance(doc, str):
            try:
                tokens = ast.literal_eval(doc) if doc.strip().startswith("[") else doc.split()
                if isinstance(tokens, list) and all(isinstance(tok, str) for tok in tokens):
                    tokenized_docs.append(tokens)
            except Exception:
                tokenized_docs.append(doc.split())

    # Buat dictionary dari dokumen
    dictionary = Dictionary(tokenized_docs)
    vocab = set(dictionary.token2id.keys())  # Ambil seluruh kata unik dari dokumen

    # Ambil kata topik dari BERTopic
    raw_topics = [
        list(set(" ".join([word for word, _ in topic_model.get_topic(topic_id)]).split()))
        for topic_id in topic_model.get_topics().keys() if topic_id != -1
    ]

    # 🔽 FILTERING topik berdasarkan vocab
    filtered_topics = []
    for topic in raw_topics:
        valid_words = [w for w in topic if w in vocab]
        if len(valid_words) >= 2:
            filtered_topics.append(valid_words)

    if not filtered_topics:
        raise ValueError("Tidak ada topik valid dari BERTopic yang lolos filter vocab.")

    # Buat corpus dari dokumen
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Evaluasi coherence
    coherence_model = CoherenceModel(
        topics=filtered_topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    score = coherence_model.get_coherence()

    duration = time.time() - start_time
    st.info(f"Evaluasi coherence terhadap {len(tokenized_docs)} dokumen selesai dalam {duration:.2f} detik.")

    return score

def evaluate_nmf_model(nmf_topic_keywords, docs):
    try:
        if isinstance(nmf_topic_keywords, dict):
            nmf_topic_keywords = list(nmf_topic_keywords.values())

        topic_keywords = [
            [word for word in topic if isinstance(word, str)]
            for topic in nmf_topic_keywords
            if isinstance(topic, list) and len(topic) >= 2
        ]

        if not topic_keywords:
            raise ValueError("Tidak ada topik valid dengan cukup kata untuk dievaluasi.")

        tokenized_docs = [doc.split() for doc in docs if isinstance(doc, str)]
        dictionary = Dictionary(tokenized_docs)
        vocab = set(dictionary.token2id.keys())

        filtered_topic_words = []
        for words in topic_keywords:
            valid_words = [word for word in words if word in vocab]
            if len(valid_words) >= 2:
                filtered_topic_words.append(valid_words)

        if not filtered_topic_words:
            raise ValueError("Tidak ada topik valid dengan cukup kata setelah filter vocab.")

        coherence_model = CoherenceModel(
            topics=filtered_topic_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()

    except Exception as e:
        raise RuntimeError("Gagal evaluasi NMF: " + str(e))


def evaluate_nmf_model(nmf_topic_keywords, docs):
    try:
        if isinstance(nmf_topic_keywords, dict):
            nmf_topic_keywords = list(nmf_topic_keywords.values())

        topic_keywords = [
            [word for word in topic if isinstance(word, str)]
            for topic in nmf_topic_keywords
            if isinstance(topic, list) and len(topic) >= 2
        ]

        if not topic_keywords:
            raise ValueError("Tidak ada topik valid dengan cukup kata untuk dievaluasi.")

        tokenized_docs = [doc.split() for doc in docs if isinstance(doc, str)]
        dictionary = Dictionary(tokenized_docs)
        vocab = set(dictionary.token2id.keys())

        filtered_topic_words = []
        for words in topic_keywords:
            valid_words = [word for word in words if word in vocab]
            if len(valid_words) >= 2:
                filtered_topic_words.append(valid_words)

        if not filtered_topic_words:
            raise ValueError("Tidak ada topik valid dengan cukup kata setelah filter vocab.")

        coherence_model = CoherenceModel(
            topics=filtered_topic_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()

    except Exception as e:
        raise RuntimeError("Gagal evaluasi NMF: " + str(e))