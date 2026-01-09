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

# Lottie animation URLs (free from LottieFiles)
SUCCESS_CONFETTI = "https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json"  # Confetti
INDEXING_DOC = "https://assets8.lottiefiles.com/packages/lf20_x62chJ.json"       # Document processing
SEARCHING = "https://assets4.lottiefiles.com/packages/lf20_Uz4uNe.json"          # Magnifying glass
WELCOME = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"            # Hello wave
CHECKMARK = "https://assets3.lottiefiles.com/packages/lf20_pBnsC0.json"         # Success check

# Try to import streamlit-lottie; fallback if not available
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
        # Graceful fallback
        if "confetti" in url or "check" in url:
            st.balloons() if "confetti" in url else st.success("✓ Done!")
        elif "search" in url.lower():
            st.spinner("Searching...")
        else:
            st.snow()  # Fun fallback for any animation

# ------------------------- Config -------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STORE_DIR = "store"
USE_GPU = True
SUMMARY_SENTENCES = 5

# ------------------------- Helpers -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def atomic_write_json(path: str, data: Dict[str, Any]):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

def meta_paths(doc_id: str, store_dir: str):
    return os.path.join(store_dir, f"{doc_id}.meta.json"), os.path.join(store_dir, f"{doc_id}.index")

def read_pdf_text(pdf_source) -> str:
    reader = PdfReader(pdf_source)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def batch_encode(model, sentences: List[str]) -> np.ndarray:
    emb = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, batch_size=32).astype("float32")
    faiss.normalize_L2(emb)
    return emb

def save_meta(meta_path: str, sentences: List[str], ids: List[int], config: Dict[str, Any]):
    atomic_write_json(meta_path, {"sentences": sentences, "ids": ids, "config": config})

def load_meta(meta_path: str) -> Dict[str, Any]:
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def stable_doc_id(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{h}"

def _maybe_to_gpu(index: faiss.Index) -> faiss.Index:
    if USE_GPU:
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            return index
    return index

def build_index(embeddings: np.ndarray) -> faiss.IndexIDMap2:
    dim = embeddings.shape[1]
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)
    return faiss.IndexIDMap2(hnsw_index)

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_index(path: str) -> Optional[faiss.Index]:
    return faiss.read_index(path) if os.path.exists(path) else None

# ------------------------- Analyzer Class -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, store_dir: str = STORE_DIR):
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.store_dir = store_dir
        ensure_dir(self.store_dir)

    def index_pdf(self, pdf_source, doc_id: Optional[str] = None, reindex: bool = False) -> dict:
        inferred_id = stable_doc_id(getattr(pdf_source, "name", "uploaded.pdf"))
        doc_id = doc_id or inferred_id
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)

        if not reindex:
            meta = load_meta(meta_path)
            index = load_index(idx_path)
            if index is not None and meta.get("sentences"):
                return {"status": "loaded", "doc_id": doc_id, "count": len(meta["sentences"]), "index_type": "HNSW"}

        sentences = split_sentences(read_pdf_text(pdf_source))
        if not sentences:
            raise ValueError("No readable sentences extracted")

        embeddings = batch_encode(self.model, sentences)
        base_index = build_index(embeddings)
        index = _maybe_to_gpu(base_index)
        ids = np.arange(len(sentences), dtype="int64")
        index.add_with_ids(embeddings, ids)

        save_index(base_index, idx_path)
        save_meta(meta_path, sentences, ids.tolist(), {"index_type": "HNSW"})
        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences), "index_type": "HNSW"}

    def search(self, query: str, doc_id: str, top_k: int = 5) -> List[str]:
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")
        index = load_index(idx_path)
        if index is None:
            raise ValueError("Index file missing")

        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        _, I = index.search(q_emb, min(top_k * 3, index.ntotal))

        candidates = [sentences[int(idx)] for idx in I[0] if idx != -1]
        if not candidates:
            return []

        pairs = [(query, cand) for cand in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in ranked[:top_k]]

    def extractive_summary(self, doc_id: str, num_sentences: int = SUMMARY_SENTENCES) -> str:
        meta_path, _ = meta_paths(doc_id, self.store_dir)
        sentences = load_meta(meta_path).get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")
        embeddings = batch_encode(self.model, sentences)
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        tmp.add(embeddings)
        _, I = tmp.search(centroid, min(num_sentences, len(sentences)))
        return " ".join(sentences[int(i)] for i in I[0])

# ------------------------- Streamlit App -------------------------
st.set_page_config(page_title="Advanced PDF Analyzer", layout="centered")
st.title("📄 Advanced PDF Analyzer with Animations")

# Welcome animation
show_lottie(WELCOME, height=180, key="welcome")

analyzer = EfficientPDFAnalyzer()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing and indexing your PDF..."):
        show_lottie(INDEXING_DOC, height=200, key="indexing_anim")
        try:
            meta = analyzer.index_pdf(uploaded_file)
            st.success(f"Successfully indexed {meta['count']} sentences!")
            st.session_state["doc_id"] = meta["doc_id"]
            show_lottie(SUCCESS_CONFETTI, height=300, key="index_success")
        except Exception as e:
            st.error(f"Error indexing PDF: {e}")

if "doc_id" in st.session_state:
    st.markdown("---")
    query = st.text_input("🔍 Ask a question about the document")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Document", use_container_width=True):
            show_lottie(SEARCHING, height=150, key="search_anim")
            try:
                results = analyzer.search(query, st.session_state["doc_id"], top_k=5)
                st.markdown("### Search Results")
                if results:
                    for i, sent in enumerate(results, 1):
                        st.write(f"{i}. {sent}")
                    show_lottie(CHECKMARK, height=100, key="search_done")
                else:
                    st.info("No relevant sentences found.")
            except Exception as e:
                st.error(f"Search error: {e}")

    with col2:
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Creating extractive summary..."):
                try:
                    summary = analyzer.extractive_summary(st.session_state["doc_id"])
                    st.markdown("### 📌 Extractive Summary")
                    st.write(summary)
                    show_lottie(SUCCESS_CONFETTI, height=300, key="summary_success")
                except Exception as e:
                    st.error(f"Summary error: {e}")
