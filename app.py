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
import time

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
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
    return faiss.IndexIDMap2(hnsw_index)

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_index(path: str) -> Optional[faiss.Index]:
    return faiss.read_index(path) if os.path.exists(path) else None

# -------------------------
# Animations
# -------------------------
def shimmer_loader(text="Indexing your PDF..."):
    st.markdown("""
    <style>
    .shimmer {
        height: 18px;
        width: 100%;
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 37%, #f0f0f0 63%);
        background-size: 400% 100%;
        animation: shimmer 1.4s ease-in-out infinite;
        border-radius: 6px;
        margin: 8px 0;
    }
    @keyframes shimmer {
        0% { background-position: 100% 0; }
        100% { background-position: -100% 0; }
    }
    </style>
    """, unsafe_allow_html=True)
    st.write(text)
    for _ in range(4):
        st.markdown('<div class="shimmer"></div>', unsafe_allow_html=True)

def staged_progress(stages):
    progress = st.progress(0)
    status = st.empty()
    for i, (label, delay) in enumerate(stages, start=1):
        status.markdown(f"**{label}**")
        progress.progress(int(i / len(stages) * 100))
        time.sleep(delay)
    status.markdown("**Done.**")

def typewriter(text, speed=0.02):
    container = st.empty()
    out = ""
    for ch in text:
        out += ch
        container.markdown(f"### 🔍 {out}")
        time.sleep(speed)

def fade_in_list(items):
    st.markdown("""
    <style>
    .fade-item { opacity: 0; transform: translateY(6px); animation: fadeIn 0.35s forwards; }
    .fade-item:nth-child(1){ animation-delay: 0.05s; }
    .fade-item:nth-child(2){ animation-delay: 0.10s; }
    .fade-item:nth-child(3){ animation-delay: 0.15s; }
    .fade-item:nth-child(4){ animation-delay: 0.20s; }
    .fade-item:nth-child(5){ animation-delay: 0.25s; }
    @keyframes fadeIn { to { opacity: 1; transform: translateY(0); } }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div>", unsafe_allow_html=True)
    for s in items:
        st.markdown(f'<div class="fade-item">- {s}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def pulsing_banner(text="Compiling extractive summary..."):
    st.markdown("""
    <style>
    .pulse {
        padding: 10px 14px;
        border-radius: 8px;
        background: #1f6feb20;
        border: 1px solid #1f6feb55;
        display: inline-block;
        animation: pulse 1.2s ease-in-out infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(31,111,235,0.4); }
        70% { box-shadow: 0 0 0 10px rgba(31,111,235,0); }
        100% { box-shadow: 0 0 0 0 rgba(31,111,235,0); }
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="pulse">{text}</div>', unsafe_allow_html=True)

def animated_header(title="📄 Advanced PDF Analyzer (Optimized)"):
    st.markdown(f"""
    <style>
    .grad {{
        background: linear-gradient(90deg, #0ea5e9, #22c55e, #f59e0b);
        background-size: 200% 200%;
        animation: moveGrad 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700; font-size: 2rem;
        margin-bottom: 8px;
    }}
    @keyframes moveGrad {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    </style>
    <div class="grad">{title}</div>
    """, unsafe_allow_html=True)

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
                    "index_type": meta.get("config", {}).get("index_type", "HNSW")
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
        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences), "index_type": "HNSW"}

    def search(self, query: str, doc_id: str, top_k: int = 3) -> List[str]:
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")
        index = load_index(idx_path)
        if index is None:
            raise ValueError("Index file missing or unreadable")

        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        _, I = index.search(q_emb, min(top_k * 3, index.ntotal))  # get more candidates

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

# -------------------------
# Streamlit App
# -------------------------
animated_header("📄 Advanced PDF Analyzer (Optimized)")

analyzer = EfficientPDFAnalyzer()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    try:
        shimmer_loader("Extracting text and building HNSW index...")
        staged_progress([
            ("Reading PDF pages", 0.5),
            ("Splitting sentences", 0.3),
            ("Encoding embeddings", 0.7),
            ("Building HNSW index", 0.6),
            ("Saving metadata", 0.3),
        ])
        meta = analyzer.index_pdf(uploaded_file)
        st.success(f"Indexed {meta['count']} sentences from {uploaded_file.name} (Index type: {meta['index_type']})")
        st.session_state["doc_id"] = meta["doc_id"]
    except Exception as e:
        st.error(f"Error indexing PDF: {e}")

if "doc_id" in st.session_state:
    query = st.text_input("Enter search query")
    if st.button("Search"):
        try:
            typewriter("Search Results")
            results = analyzer.search(query, st.session_state["doc_id"], top_k=5)
            fade_in_list(results)
        except Exception as e:
            st.error(f"Error during search: {e}")

    if st.button("Generate Summary"):
        try:
            pulsing_banner("Compiling extractive summary...")
            summary_text = analyzer.extractive_summary(st.session_state["doc_id"], num_sentences=SUMMARY_SENTENCES)
            st.write("### 📌 Extractive Summary")
            st.write(summary_text)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
