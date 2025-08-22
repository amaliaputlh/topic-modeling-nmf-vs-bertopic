import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pickle
import time

from bertopic_model import embedding, dimensionality_reduction, clustering, vectorization, modeling
from evaluation import topic_coherence, topic_diversity
from evaluation.detect_similarity import get_top_words_per_topic, get_topic_embeddings, compute_similarity_matrix, find_similar_pairs
from utils.save_documents import save_documents_from_csv 
from nmf_model import vectorizer as nmf_vectorizer
from nmf_model import modeling as nmf_modeling

from evaluation.topic_coherence import evaluate_bertopic_model, evaluate_nmf_model
from evaluation.topic_diversity import evaluate_bertopic_diversity, evaluate_nmf_diversity
from visualization import generate_bertopic_barchart, generate_nmf_barchart, generate_bertopic_table_and_wordcloud, generate_nmf_table_and_wordcloud, plot_wordcloud_grid, generate_bertopic_intertopic_map, cosine_similarity_between_topics, show_example_docs_per_topic_streamlit

st.set_page_config(page_title="Sistem Pemodelan Topik", layout="wide")
st.title("SISTEM PEMODELAN TOPIK ULASAN PRODUK SKINCARE")

st.markdown("### Status Model")

# Status BERTopic
if "bertopic_model" in st.session_state:
    topic_model = st.session_state.bertopic_model
    topic_info = topic_model.get_topic_info()
    n_total = topic_info.shape[0] 
    n_outliers = (topic_info["Topic"] == -1).sum()
    n_valid = n_total - n_outliers
    st.success(f"Model **BERTopic** tersedia — {n_valid} topik valid + {n_outliers} outlier (Total: {n_total})")
else:
    st.warning("Model **BERTopic** belum tersedia")

# Status NMF
if "nmf_topic_keywords" in st.session_state:
    num_topics = len(st.session_state.nmf_topic_keywords)
    st.success(f"Model **NMF** tersedia — {num_topics} topik")
else:
    st.warning("Model **NMF** belum tersedia")

# Sidebar 
with st.sidebar:
    uploaded_file = st.file_uploader("Upload File Dataset", type="csv", key="uploader")

    if uploaded_file is not None:
        temp_csv_path = "data/uploaded.csv" 
        with open(temp_csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv(temp_csv_path)
        docs = df["Ulasan"].astype(str).tolist()

        output_bertopic_path = "output/ulasan_documents_bertopic.pkl"
        save_documents_from_csv(temp_csv_path, text_column="Ulasan", output_path=output_bertopic_path)
        st.session_state.bertopic_doc_path = output_bertopic_path

        output_nmf_path = "output/ulasan_documents_nmf.pkl"
        save_documents_from_csv(temp_csv_path, text_column="Ulasan", output_path=output_nmf_path)
        st.session_state.nmf_doc_path = output_nmf_path

        st.session_state.docs = docs

        st.success("Dokumen berhasil diproses dan disimpan untuk kedua metode.")

    # Upload BERTopic model
    bertopic_pkl = st.file_uploader("Upload BERTopic Model (.pkl)", type="pkl", key="bertopic_model_upload")
    if bertopic_pkl is not None:
        with open("uploaded_bertopic.pkl", "wb") as f:
            f.write(bertopic_pkl.getbuffer())
        with open("uploaded_bertopic.pkl", "rb") as f:
            bertopic_model = pickle.load(f)
        st.session_state.bertopic_model = bertopic_model
        st.success("Model BERTopic berhasil dimuat dari file!")

        if os.path.exists("output/ulasan_documents_bertopic.pkl"):
            st.session_state.bertopic_doc_path = "output/ulasan_documents_bertopic.pkl"

    # Upload NMF model
    nmf_pkl = st.file_uploader("Upload NMF Topik (.pkl)", type="pkl", key="nmf_model_upload")
    if nmf_pkl is not None:
        with open("uploaded_nmf.pkl", "wb") as f:
            f.write(nmf_pkl.getbuffer())
        with open("uploaded_nmf.pkl", "rb") as f:
            nmf_data = pickle.load(f)
        st.session_state.nmf_topic_keywords = nmf_data[0]
        st.session_state.nmf_W = nmf_data[1]
        st.session_state.nmf_docs = nmf_data[2]
        with open("output/nmf_docs.pkl", "wb") as f:
            pickle.dump(nmf_data[2], f)
        st.session_state.nmf_doc_path = "output/nmf_docs.pkl"
        st.success("Model NMF berhasil dimuat dari file!")

    method = st.selectbox("Pilih Metode", ["BERTopic", "NMF"], key="select_method")
    if method == "NMF":
        num_topics = st.number_input("Jumlah Topik", min_value=2, step=1, key="num_topics_input")

# Inisialisasi flag stage
if "stage" not in st.session_state:
    st.session_state.stage = "awal"

# Pilihan proses
process = st.selectbox(
    "Pilih Proses",
    ["Modeling", "Visualisasi", "Evaluasi", "Deteksi Topik Mirip 1", "Deteksi Topik Mirip 2"],
    key="select_process"
)

# Tombol Proses → hanya set stage sekali
if st.button("Proses", key="btn_proses") and uploaded_file is not None:
    st.session_state.stage = "proses"
    st.session_state.selected_process = process  # simpan proses yang dipilih

# Eksekusi hanya kalau stage sudah proses
if st.session_state.stage == "proses":

    # Ambil kembali proses yang disimpan
    process = st.session_state.get("selected_process", process)
# Modeling
    if process == "Modeling":
        if method == "BERTopic":
            start_time = time.time()
            embeddings = embedding.get_embeddings(docs)
            reduced_embeddings = dimensionality_reduction.reduce_dimensions(embeddings)
            clusters = clustering.cluster(reduced_embeddings)
            st.write(f"🔹 Jumlah cluster unik (termasuk -1/outlier): {len(set(clusters))}")
            vectorizer_model, ctfidf_model = vectorization.get_vectorizer_and_ctfidf(docs)
            topic_model, topics, probs = modeling.run_bertopic(docs, clusters, vectorizer_model, ctfidf_model)

            duration = time.time() - start_time
            st.info(f"Pemodelan selesai dalam {duration:.2f} detik.")

            st.session_state["bertopic_doc_path"] = output_bertopic_path
            st.session_state.bertopic_model = topic_model
            st.session_state.topics = topics

            st.subheader("Informasi Topik BERTopic")
            st.dataframe(topic_model.get_topic_info())

        elif method == "NMF":
            start_time = time.time()
            X_tfidf, feature_names = nmf_vectorizer.generate_tfidf_matrix(docs)
            nmf_model, topic_keywords, W = nmf_modeling.run_nmf(X_tfidf, feature_names, docs, num_topics=num_topics)

            duration = time.time() - start_time
            st.info(f"Pemodelan selesai dalam {duration:.2f} detik.")

            st.session_state.nmf_topic_keywords = topic_keywords
            st.session_state.nmf_W = W
            st.session_state.nmf_docs = docs

            with open("output/nmf_docs.pkl", "wb") as f:
                pickle.dump(docs, f)
            st.session_state.nmf_doc_path = "output/nmf_docs.pkl"

            doc_topics = W.argmax(axis=1) 
            topic_counts = Counter(doc_topics)

            topic_data = []
            for i, keywords in enumerate(topic_keywords):
                topik = i
                kata_kunci = ', '.join(keywords)
                jumlah_dokumen = topic_counts.get(i, 0)
                topic_data.append((topik, kata_kunci, jumlah_dokumen))

            df_topics = pd.DataFrame(topic_data, columns=["Topik", "Kata Kunci", "Jumlah Dokumen"])

            st.subheader("Informasi Topik NMF")
            st.dataframe(df_topics)

            st.subheader("Contoh Ulasan dari 5 Topik Teratas (NMF)")
            W = st.session_state.nmf_W 
            with open(st.session_state.nmf_doc_path, "rb") as f:
                nmf_docs = pickle.load(f)


            for target_topic in range(5):
                st.markdown(f"**Topik {target_topic}**")
                topic_docs = [doc for doc, topic in zip(nmf_docs, doc_topics) if topic == target_topic]
                for i, doc in enumerate(topic_docs[:3]):
                    st.markdown(f"*Contoh {i+1}:* {doc}")

# Evaluasi
    elif process == "Evaluasi":
        bertopic_coh = bertopic_div = nmf_coh = nmf_div = None
        # BERTopic
        if "bertopic_model" in st.session_state and "bertopic_doc_path" in st.session_state:
            bertopic_model = st.session_state.bertopic_model
            doc_path = st.session_state.bertopic_doc_path
            bertopic_coh = evaluate_bertopic_model(doc_path, sample_size=None)
            with open("temp_bertopic.pkl", "wb") as f:
                pickle.dump(bertopic_model, f)
            bertopic_div = evaluate_bertopic_diversity("temp_bertopic.pkl")

        # NMF
        if "nmf_topic_keywords" in st.session_state and "nmf_doc_path" in st.session_state:
            nmf_keywords = st.session_state.nmf_topic_keywords
            with open(st.session_state.nmf_doc_path, "rb") as f:
                nmf_docs = pickle.load(f)
            nmf_coh = evaluate_nmf_model(nmf_keywords, nmf_docs)
            nmf_div = evaluate_nmf_diversity(nmf_keywords)

        # Tabel perbandingan
        st.subheader("Perbandingan Evaluasi")
        data = pd.DataFrame({
            "Metode": ["BERTopic", "NMF"],
            "Topic Coherence (c_v)": [
                bertopic_coh if bertopic_coh else 0,
                nmf_coh if nmf_coh else 0
            ],
            "Topic Diversity": [
                bertopic_div if bertopic_div else 0,
                nmf_div if nmf_div else 0
            ]
        })
        st.table(data.style.format({"Topic Coherence (c_v)": "{:.4f}", "Topic Diversity": "{:.4f}"}))

        # Bar chart
        st.subheader("Visualisasi Perbandingan (Bar Chart)")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.barplot(x="Metode", y="Topic Coherence (c_v)", data=data, ax=ax[0], palette="Blues_d")
        ax[0].set_title("Topic Coherence (c_v)")
        sns.barplot(x="Metode", y="Topic Diversity", data=data, ax=ax[1], palette="Greens_d")
        ax[1].set_title("Topic Diversity")
        st.pyplot(fig)

        st.subheader("Visualisasi Distribusi Topik dan WordCloud")

        # --- BERTopic ---
        if "bertopic_model" in st.session_state:
            st.markdown("## BERTopic")

            df_topic, wordclouds = generate_bertopic_table_and_wordcloud(st.session_state.bertopic_model)
            st.markdown("### Tabel Topik")
            st.dataframe(df_topic)

            st.markdown("### WordCloud Grid (maks 16 topik)")
            fig_wc_bertopic = plot_wordcloud_grid(wordclouds, title_prefix="Topik")
            st.pyplot(fig_wc_bertopic)
        else:
            st.warning("Model BERTopic belum tersedia.")

        # --- NMF ---
        if "nmf_topic_keywords" in st.session_state:
            st.markdown("## NMF")

            df_topic_nmf, wordclouds_nmf = generate_nmf_table_and_wordcloud(st.session_state.nmf_topic_keywords)
            st.markdown("### Tabel Topik")
            st.dataframe(df_topic_nmf)

            st.markdown("### WordCloud Grid (maks 16 topik)")
            fig_wc_nmf = plot_wordcloud_grid(wordclouds_nmf, title_prefix="Topik")
            st.pyplot(fig_wc_nmf)
        else:
            st.warning("Model NMF belum tersedia.")
       
#Visualisasi
    elif process == "Visualisasi":
        st.subheader("Visualisasi Distribusi Topik dan WordCloud")

        # --- BERTopic ---
        if "bertopic_model" in st.session_state:
            st.markdown("## BERTopic")

            fig_bertopic = generate_bertopic_barchart(st.session_state.bertopic_model)
            st.pyplot(fig_bertopic)

            df_topic, wordclouds = generate_bertopic_table_and_wordcloud(st.session_state.bertopic_model)
            st.markdown("### Tabel Topik")
            st.dataframe(df_topic)

            st.markdown("### WordCloud Grid (maks 16 topik)")
            fig_wc_bertopic = plot_wordcloud_grid(wordclouds, title_prefix="Topik")
            st.pyplot(fig_wc_bertopic)

            st.markdown("### Intertopic Distance Map (BERTopic)")
            fig_intertopic = generate_bertopic_intertopic_map(st.session_state.bertopic_model)
            if fig_intertopic:
                st.plotly_chart(fig_intertopic, use_container_width=True)
            topics = topic_model.get_document_info(docs)["Topic"].tolist()
            show_text = show_example_docs_per_topic_streamlit(docs, topics, target_topics=[9, 10], n_examples=5)
            st.markdown(show_text)

            cosine = cosine_similarity_between_topics(topic_model, topic_ids=[9, 10])
            st.markdown(f"**Cosine similarity antara Topik 9 dan 10:** {cosine:.4f}")


        else:
            st.warning("Model BERTopic belum tersedia.")

        # --- NMF ---
        if "nmf_topic_keywords" in st.session_state:
            st.markdown("## NMF")

            fig_nmf = generate_nmf_barchart(st.session_state.nmf_W)
            st.pyplot(fig_nmf)

            df_topic_nmf, wordclouds_nmf = generate_nmf_table_and_wordcloud(st.session_state.nmf_topic_keywords)
            st.markdown("### Tabel Topik")
            st.dataframe(df_topic_nmf)

            st.markdown("### WordCloud Grid (maks 16 topik)")
            fig_wc_nmf = plot_wordcloud_grid(wordclouds_nmf, title_prefix="Topik")
            st.pyplot(fig_wc_nmf)
        else:
            st.warning("Model NMF belum tersedia.")
#Deteksi
    elif process == "Deteksi Topik Mirip 1":
        st.subheader("Deteksi Topik Mirip BERTopic")
        if "bertopic_model" in st.session_state:
            threshold = st.slider("Threshold cosine similarity", 0.7, 0.99, 0.85, 0.01)
            top_k = st.slider("Jumlah kata per topik", 5, 20, 10)
            model_name =    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            
            if st.button("Deteksi", key="btn_deteksi_1"):
                topic_model = st.session_state.bertopic_model
                top_words = get_top_words_per_topic(topic_model, top_k=top_k)
                topic_embeddings = get_topic_embeddings(top_words, model_name=model_name)
                sim_matrix = compute_similarity_matrix(topic_embeddings)
                pairs = find_similar_pairs(sim_matrix, threshold=threshold)

                if pairs:
                    df_similar = pd.DataFrame(pairs, columns=["Topik 1", "Topik 2", "Similarity"])
                    st.success(f"Ditemukan {len(pairs)} pasang topik mirip.")
                    st.dataframe(df_similar)
                else:
                    st.info("Tidak ada topik yang mirip di atas threshold.")
        else:
            st.warning("Model BERTopic belum tersedia.")

    elif process == "Deteksi Topik Mirip 2":
        st.subheader("Deteksi Topik Mirip NMF")
        if "nmf_topic_keywords" in st.session_state:
            threshold = st.slider("Threshold cosine similarity", 0.7, 0.99, 0.85, 0.01, key="nmf_threshold")
            top_k = st.slider("Jumlah kata per topik", 5, 20, 10, key="nmf_top_k")
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

            if st.button("Deteksi", key="btn_deteksi_2"):
                from evaluation.detect_similarity import (
                    get_top_words_per_topic, get_topic_embeddings,
                    compute_similarity_matrix, find_similar_pairs
                )

                topic_words = get_top_words_per_topic(
                    topic_source=st.session_state.nmf_topic_keywords,
                    top_k=top_k,
                    method="nmf"
                )
                topic_embeddings = get_topic_embeddings(topic_words, model_name=model_name)
                sim_matrix = compute_similarity_matrix(topic_embeddings)
                pairs = find_similar_pairs(sim_matrix, threshold=threshold)

                if pairs:
                    df_similar = pd.DataFrame(pairs, columns=["Topik 1", "Topik 2", "Similarity"])
                    st.success(f"Ditemukan {len(pairs)} pasang topik mirip.")
                    st.dataframe(df_similar)
                else:
                    st.info("Tidak ada topik yang mirip di atas threshold.")
        else:
            st.warning("Topik dari NMF belum tersedia.")

