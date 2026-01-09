import os
import json
import hashlib
import numpy as np
import faiss
import base64
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
import re
import streamlit as st
from typing import List, Dict, Any, Optional

# -------------------------
# Config
# -------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STORE_DIR = "store"
USE_GPU = True
SUMMARY_SENTENCES = 5
GIF_PATH = "/mnt/data/telugu-pubg.gif"

# -------------------------
# Helpers
# -------------------------
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
    emb = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32
    ).astype("float32")
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
    hnsw = faiss.IndexHNSWFlat(dim, 32)
    return faiss.IndexIDMap2(hnsw)

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_index(path: str) -> Optional[faiss.Index]:
    return faiss.read_index(path) if os.path.exists(path) else None

# -------------------------
# PUBG Loading Animation
# -------------------------
def show_loading_animation(text: str):
    with open(GIF_PATH, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; align-items:center;">
            <img src="data:image/gif;base64,{encoded}" width="220">
            <p style="font-size:16px; font-weight:600;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Analyzer
# -------------------------
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
            meta, index = load_meta(meta_path), load_index(idx_path)
            if index is not None and meta.get("sentences"):
                return {
                    "status": "loaded",
                    "doc_id": doc_id,
                    "count": len(meta["sentences"]),
                    "index_type": "HNSW"
                }

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

        return {
            "status": "indexed",
            "doc_id": doc_id,
            "count": len(sentences),
            "index_type": "HNSW"
        }

    def search(self, query: str, doc_id: str, top_k: int = 3) -> List[str]:
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])

        if not sentences:
            raise ValueError("Document not indexed")

        index = load_index(idx_path)
        if index is None:
            raise ValueError("Index missing")

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        _, I = index.search(q_emb, min(top_k * 3, index.ntotal))
        candidates = [sentences[int(i)] for i in I[0] if i != -1]

        pairs = [(query, c) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        return [s for s, _ in ranked[:top_k]]

# -------------------------
# Streamlit App
# -------------------------
st.title("📄 Advanced PDF Analyzer (PUBG Loader)")

analyzer = EfficientPDFAnalyzer()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    try:
        with st.spinner("Indexing PDF..."):
            show_loading_animation("Indexing PDF...")
            meta = analyzer.index_pdf(uploaded_file)

        st.success(
            f"Indexed {meta['count']} sentences from {uploaded_file.name} "
            f"(Index type: {meta['index_type']})"
        )
        st.session_state["doc_id"] = meta["doc_id"]

    except Exception as e:
        st.error(f"Error indexing PDF: {e}")

if "doc_id" in st.session_state:
    query = st.text_input("Enter search query")

    if st.button("Search"):
        try:
            with st.spinner("Searching..."):
                show_loading_animation("Searching...")
                results = analyzer.search(query, st.session_state["doc_id"], top_k=5)

            st.write("### 🔍 Search Results")
            for r in results:
                st.write(f"- {r}")

        except Exception as e:
            st.error(f"Search error: {e}")
