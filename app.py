import os
import json
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
import re
import streamlit as st
from typing import List, Dict, Any, Optional
import requests

# ------------------------- Animation Setup -------------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

# Diverse Lottie animations (varied styles, no ice/snow themes)
WELCOME_ROBOT      = "https://assets5.lottiefiles.com/packages/lf20_kkflmtur.json"  # Friendly robot wave
INDEXING_DOC       = "https://assets2.lottiefiles.com/packages/lf20_6nfjl2av.json"  # Dynamic document upload
SEARCHING_GLASS    = "https://assets4.lottiefiles.com/packages/lf20_Uz4uNe.json"   # Magnifying glass search
FIREWORKS_SUCCESS  = "https://assets10.lottiefiles.com/packages/lf20_towptqfc.json"  # Vibrant fireworks
PARTY_POPPER       = "https://assets1.lottiefiles.com/packages/lf20_yM3Lp0P7.json"   # Party celebration popper
GREEN_CHECKMARK    = "https://assets3.lottiefiles.com/packages/lf20_pBnsC0.json"    # Bold success check

try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

def show_lottie(url: str, height: int = 200, key: str = None):
    anim = load_lottie_url(url)
    if LOTTIE_AVAILABLE and anim:
        st_lottie(anim, height=height, key=key)
    else:
        # Varied fallbacks (confetti, fireworks effect via balloons, etc.)
        if "fireworks" in url or "party" in url:
            st.balloons()
        elif "check" in url:
            st.success("✓ Completed!")
        else:
            st.confetti()  # Modern confetti as default

# ------------------------- Config -------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STORE_DIR = "store"
USE_GPU = True
SUMMARY_SENTENCES = 5
HNSW_M = 32          # Neighbors for build (good default)
EF_SEARCH = 32       # Higher = better recall, tunable for speed/accuracy

# ------------------------- Helpers (unchanged mostly) -------------------------
# ... (keep all previous helpers: ensure_dir, atomic_write_json, etc.)

def build_index(embeddings: np.ndarray) -> faiss.IndexIDMap2:
    dim = embeddings.shape[1]
    hnsw_index = faiss.IndexHNSWFlat(dim, HNSW_M)
    hnsw_index.hnsw.efConstruction = 200  # Good build quality
    return faiss.IndexIDMap2(hnsw_index)

# ------------------------- Analyzer Class (efficiency tweaks) -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, store_dir: str = STORE_DIR):
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.store_dir = store_dir
        ensure_dir(self.store_dir)

    # ... index_pdf unchanged ...

    def search(self, query: str, doc_id: str, top_k: int = 5) -> List[str]:
        # ... load meta/index ...
        index = load_index(idx_path)
        if index is None or not hasattr(index, 'hnsw'):
            raise ValueError("Index invalid")
        index.hnsw.efSearch = EF_SEARCH  # Tune retrieval efficiency/accuracy

        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        _, I = index.search(q_emb, min(top_k * 3, index.ntotal))

        # ... rest unchanged (candidates, rerank) ...

    def extractive_summary(self, doc_id: str, num_sentences: int = SUMMARY_SENTENCES) -> str:
        meta_path, _ = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")

        # Cache embeddings in session_state for efficiency
        cache_key = f"{doc_id}_summary_emb"
        if cache_key not in st.session_state:
            embeddings = batch_encode(self.model, sentences)
            st.session_state[cache_key] = embeddings
        else:
            embeddings = st.session_state[cache_key]

        centroid = np.mean(embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(centroid)

        # Add position bias: favor start/end sentences (common in reports/PDFs)
        positions = np.array([i / len(sentences) for i in range(len(sentences))])
        pos_bias = 1.0 - np.abs(positions - 0.5) * 0.5  # Peak at start/end
        biased_emb = embeddings + (embeddings * pos_bias[:, np.newaxis] * 0.1)  # Light boost

        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        tmp.add(biased_emb)
        _, I = tmp.search(centroid, min(num_sentences * 2, len(sentences)))
        
        top_indices = I[0][:num_sentences]
        top_indices = sorted(top_indices, key=lambda x: x)  # Preserve rough order
        return " ".join(sentences[int(i)] for i in top_indices)

# ------------------------- Streamlit App (varied animations) -------------------------
st.set_page_config(page_title="Advanced PDF Analyzer", layout="centered")
st.title("📄 Advanced PDF Analyzer – Upgraded!")

show_lottie(WELCOME_ROBOT, height=180, key="welcome")

analyzer = EfficientPDFAnalyzer()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing and indexing your PDF..."):
        show_lottie(INDEXING_DOC, height=200, key="indexing_anim")
        try:
            meta = analyzer.index_pdf(uploaded_file)
            st.success(f"Successfully indexed {meta['count']} sentences!")
            st.session_state["doc_id"] = meta["doc_id"]
            show_lottie(FIREWORKS_SUCCESS, height=300, key="index_success")  # Fireworks!
        except Exception as e:
            st.error(f"Error indexing PDF: {e}")

if "doc_id" in st.session_state:
    st.markdown("---")
    query = st.text_input("🔍 Ask a question about the document")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Document", use_container_width=True):
            show_lottie(SEARCHING_GLASS, height=150, key="search_anim")
            try:
                results = analyzer.search(query, st.session_state["doc_id"], top_k=5)
                st.markdown("### Search Results")
                if results:
                    for i, sent in enumerate(results, 1):
                        st.write(f"{i}. {sent}")
                    show_lottie(GREEN_CHECKMARK, height=100, key="search_done")
                else:
                    st.info("No relevant sentences found.")
            except Exception as e:
                st.error(f"Search error: {e}")

    with col2:
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Creating smarter summary..."):
                try:
                    summary = analyzer.extractive_summary(st.session_state["doc_id"])
                    st.markdown("### 📌 Extractive Summary")
                    st.write(summary)
                    show_lottie(PARTY_POPPER, height=300, key="summary_success")  # Party popper!
                except Exception as e:
                    st.error(f"Summary error: {e}")
