import os
import json
import hashlib
import time
from math import ceil
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
import re
import streamlit as st

# Optional Lottie support
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

# -------------------------
# Config
# -------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STORE_DIR = "store"
USE_GPU = True
SUMMARY_SENTENCES = 5

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

def save_meta(meta_path: str, sentences: List[str], ids: List[int], config: Dict[str, Any]):
    atomic_write_json(meta_path, {"sentences": sentences, "ids": ids, "config": config})

def load_meta(meta_path: str) -> Dict[str, Any]:
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# UI Animation Helper
# -------------------------
def show_animation(container, lottie_json=None, gif_path=None, height=180):
    if LOTTIE_AVAILABLE and lottie_json is not None:
        st_lottie(lottie_json, height=height, key=f"lottie-{id(container)}")
    elif gif_path:
        container.image(gif_path, use_column_width=False, width=height)

# -------------------------
# Chunked encoder with progress callback
# -------------------------
def batch_encode(model, sentences: List[str], progress_callback=None, chunk_size: int = 128) -> np.ndarray:
    all_embs = []
    total = len(sentences)
    if total == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    chunks = ceil(total / chunk_size)
    for i in range(chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = sentences[start:end]
        emb = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True, batch_size=32).astype("float32")
        faiss.normalize_L2(emb)
        all_embs.append(emb)
        if progress_callback:
            progress_callback(end, total)
        # tiny sleep only for smoother UI updates; remove in production if desired
        time.sleep(0.01)
    return np.vstack(all_embs)

# -------------------------
# Analyzer
# -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, store_dir: str = STORE_DIR):
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.store_dir = store_dir
        ensure_dir(self.store_dir)

    def index_pdf(self, pdf_source, doc_id: Optional[str] = None, reindex: bool = False, ui_container=None) -> dict:
        inferred_id = stable_doc_id(getattr(pdf_source, "name", "uploaded.pdf"))
        doc_id = doc_id or inferred_id
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)

        if not reindex:
            meta, index = load_meta(meta_path), load_index(idx_path)
            if index is not None and meta.get("sentences"):
                return {"status": "loaded", "doc_id": doc_id, "count": len(meta["sentences"]), "index_type": meta.get("config", {}).get("index_type", "HNSW")}

        text = read_pdf_text(pdf_source)
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("No readable sentences extracted")

        progress_bar = None
        anim_slot = None
        if ui_container:
            progress_bar = ui_container.progress(0)
            anim_slot = ui_container.empty()
            # show a small animation while encoding (replace gif_path with your file or lottie_json)
            # show_animation(anim_slot, gif_path="indexing.gif")

        def _progress_callback(done, total):
            if progress_bar:
                progress_bar.progress(min(100, int(done / total * 100)))

        # show global spinner while encoding (spinner is top-level; animation slot shows visuals)
        with st.spinner("Encoding document and building index…"):
            # optional visual snow while indexing
            if ui_container:
                st.snow()
            embeddings = batch_encode(self.model, sentences, progress_callback=_progress_callback, chunk_size=128)
            if ui_container:
                # stop snow by clearing the animation slot (snow is global; we just clear any gif)
                anim_slot.empty()

        base_index = build_index(embeddings)
        index = _maybe_to_gpu(base_index)
        ids = np.arange(len(sentences), dtype="int64")
        index.add_with_ids(embeddings, ids)

        save_index(base_index, idx_path)
        save_meta(meta_path, sentences, ids.tolist(), {"index_type": "HNSW"})

        if progress_bar:
            progress_bar.empty()

        # celebratory animation on success
        st.balloons()

        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences), "index_type": "HNSW"}

    def search(self, query: str, doc_id: str, top_k: int = 3, ui_container=None) -> List[str]:
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")
        index = load_index(idx_path)
        if index is None:
            raise ValueError("Index file missing or unreadable")

        # Use a small animation slot and the top-level spinner
        anim_slot = ui_container.empty() if ui_container else None
        with st.spinner("Searching and reranking results…"):
            if anim_slot:
                # show_animation(anim_slot, gif_path="searching.gif")
                pass
            q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            _, I = index.search(q_emb, min(top_k * 3, max(1, index.ntotal)))
            if anim_slot:
                anim_slot.empty()

        candidates = [sentences[int(idx)] for idx in I[0] if idx != -1]
        if not candidates:
            return []

        pairs = [(query, cand) for cand in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in ranked[:top_k]]

    def extractive_summary(self, doc_id: str, num_sentences: int = SUMMARY_SENTENCES, ui_container=None) -> str:
        meta_path, _ = meta_paths(doc_id, self.store_dir)
        sentences = load_meta(meta_path).get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")

        anim_slot = ui_container.empty() if ui_container else None
        with st.spinner("Generating extractive summary…"):
            if anim_slot:
                # show_animation(anim_slot, gif_path="summarizing.gif")
                pass
            embeddings = batch_encode(self.model, sentences, progress_callback=None, chunk_size=128)
            if anim_slot:
                anim_slot.empty()

        centroid = np.mean(embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        tmp.add(embeddings)
        _, I = tmp.search(centroid, min(num_sentences, len(sentences)))
        return " ".join(sentences[int(i)] for i in I[0])

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Advanced PDF Analyzer", layout="wide")
st.title("📄 Advanced PDF Analyzer (Optimized)")

analyzer = EfficientPDFAnalyzer()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    try:
        ui = st.empty()
        with ui.container():
            st.write("Indexing document. This may take a moment.")
            meta = analyzer.index_pdf(uploaded_file, ui_container=ui)
        st.success(f"Indexed {meta['count']} sentences from {uploaded_file.name} (Index type: {meta['index_type']})")
        st.session_state["doc_id"] = meta["doc_id"]
    except Exception as e:
        st.error(f"Error indexing PDF: {e}")

if "doc_id" in st.session_state:
    query = st.text_input("Enter search query")
    if st.button("Search"):
        try:
            ui = st.empty()
            with ui.container():
                results = analyzer.search(query, st.session_state["doc_id"], top_k=5, ui_container=ui)
            st.write("### 🔍 Search Results")
            if results:
                for sentence in results:
                    st.write(f"- {sentence}")
            else:
                st.write("No relevant results found.")
        except Exception as e:
            st.error(f"Error during search: {e}")

    if st.button("Generate Summary"):
        try:
            ui = st.empty()
            with ui.container():
                summary_text = analyzer.extractive_summary(st.session_state["doc_id"], num_sentences=SUMMARY_SENTENCES, ui_container=ui)
            st.write("### 📌 Extractive Summary")
            st.write(summary_text)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
