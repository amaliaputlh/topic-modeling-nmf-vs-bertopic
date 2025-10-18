# 🧴 Skincare Reviews Topic Modeling (NMF vs BERTopic)

## 📘 Background
The rapid advancement of technology and the internet has made online platforms a primary source of public opinion. Many users express their experiences and preferences on digital forums, creating large volumes of unstructured text data.
To extract meaningful insights from these reviews, automated topic modeling techniques in Natural Language Processing (NLP) can be applied to group related discussions and reveal common themes. In the beauty industry, online reviews play a major role in shaping consumer purchasing decisions. Platforms such as Female Daily Talk (FD Talk) serve as the largest online community for skincare and cosmetic discussions in Indonesia, offering valuable user-generated data. However, due to the massive number of reviews, it is difficult for consumers to manually interpret all available opinions. Thus, this research employs unsupervised machine learning to summarize and categorize review content efficiently.Previous studies have shown that both NMF and BERTopic perform well for short-text data like online reviews. NMF relies on TF-IDF matrix decomposition for topic extraction, while BERTopic leverages Sentence-BERT embeddings, HDBSCAN clustering, and c-TF-IDF to produce semantically coherent and flexible topics. This project aims to empirically evaluate both models in the context of skincare product reviews.

---

## 🎯 Research Problem

The main research question is:  

> How do **BERTopic** and **Non-Negative Matrix Factorization (NMF)** compare in identifying and clustering topics from skincare product reviews, based on **topic coherence** and **topic diversity** evaluation metrics?

---

## ⚙️ Scope and Limitations

1. The dataset used in this study consists of **user reviews of skincare products** collected from the **Female Daily Talk (FD Talk)** forum through web scraping.  
2. The specific product analyzed is **Avoskin Miraculous Refining Toner**.  
   *(Note: The BERTopic model is currently optimized for this dataset and may not generalize to other skincare products.)*

---

## 🎯 Objectives
- Apply unsupervised topic modeling on skincare reviews.
- Compare the performance of NMF and BERTopic.
- Evaluate models using coherence and topic diversity.
- Provide interactive visualization through a Streamlit app.

---

## 📊 Dataset
- Source: Reviews of skincare products from Female Daily.
- Size: ~2000 reviews.
- Language: Indonesian.
- Format: Preprocessed CSV

⚠️ Note: Preprocessing (cleaning, tokenizing, stopword removal, stemming, etc.) was conducted separately in Jupyter Notebook. The final preprocessed dataset is saved as CSV and used as input in this project. The preprocessing notebook is included in the preprocessing/ folder for reference.

---

## 📊 Key Insights
1. This study successfully compared **BERTopic** and **NMF** in identifying and grouping topics from skincare product reviews using two evaluation metrics: **topic coherence** and **topic diversity**.  
2. **BERTopic** outperformed NMF on both metrics.  
   - Higher *topic coherence* indicates stronger semantic relationships among keywords.  
   - Slightly higher *topic diversity* combined with cosine similarity analysis shows more semantically distinct topics.  
3. BERTopic’s superiority is attributed to its **Sentence-BERT semantic representation** and **c-TF-IDF topic extraction**, which capture contextual meaning better than NMF’s matrix decomposition approach.  
4. Some overlap between BERTopic topics was observed, but it did not negatively affect coherence or diversity scores — this overlap is natural in embedding-based clustering.  
5. From a computational perspective, **NMF was faster** due to its simpler linear algebraic structure, while **BERTopic required more processing time** because of its transformer-based and clustering components.
