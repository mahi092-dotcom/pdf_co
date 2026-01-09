import os
import json
import hashlib
import re
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# Config
# =========================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STORE_DIR = "store"
USE_GPU = True
SUMMARY_SENTENCES = 5
CHUNK_SIZE = 4       # sentences per chunk
CHUNK_STRIDE = 2     # overlap

# =========================
# Helpers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def atomic_write_json(path: str, data: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def meta_paths(doc_id: str):
    return (
        os.path.join(STORE_DIR, f"{doc_id}.meta.json"),
        os.path.join(STORE_DIR, f"{doc_id}.index"),
    )


def stable_doc_id(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{h}"


def read_pdf_text(pdf_source) -> str:
    pdf_source.seek(0)
    reader = PdfReader(pdf_source)
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    return re.sub(r"\s+", " ", text)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]


def chunk_sentences(sentences: List[str]) -> List[str]:
    chunks = []
    for i in range(0, len(sentences) - CHUNK_SIZE + 1, CHUNK_STRIDE):
        chunk = " ".join(sentences[i:i + CHUNK_SIZE])
        chunks.append(chunk)
    return chunks


def embed(model, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    return emb


def maybe_gpu(index: faiss.Index) -> faiss.Index:
    if USE_GPU:
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass
    return index


def build_index(dim: int) -> faiss.Index:
    return faiss.IndexIDMap2(faiss.IndexHNSWFlat(dim, 32))


# =========================
# Analyzer
# =========================
class EfficientPDFAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder(RERANK_MODEL, device="cuda" if USE_GPU else "cpu")
        ensure_dir(STORE_DIR)

    def index_pdf(self, pdf_source, reindex: bool = False) -> Dict[str, Any]:
        doc_id = stable_doc_id(getattr(pdf_source, "name", "uploaded.pdf"))
        meta_path, idx_path = meta_paths(doc_id)

        if not reindex and os.path.exists(meta_path) and os.path.exists(idx_path):
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            return {"status": "loaded", "doc_id": doc_id, "count": len(meta["chunks"])}

        with st.spinner("📄 Reading PDF..."):
            text = read_pdf_text(pdf_source)
            sentences = split_sentences(text)
            chunks = chunk_sentences(sentences)

        with st.spinner("🧠 Generating embeddings..."):
            embeddings = embed(self.model, chunks)

        with st.spinner("📦 Building FAISS index..."):
            index = build_index(embeddings.shape[1])
            ids = np.arange(len(chunks)).astype("int64")
            index.add_with_ids(embeddings, ids)
            faiss.write_index(index, idx_path)

        atomic_write_json(meta_path, {
            "chunks": chunks,
            "sentences": sentences,
            "config": {
                "chunk_size": CHUNK_SIZE,
                "stride": CHUNK_STRIDE,
                "model": MODEL_NAME,
            }
        })

        return {"status": "indexed", "doc_id": doc_id, "count": len(chunks)}

    def search(self, query: str, doc_id: str, top_k: int = 5) -> List[str]:
        meta_path, idx_path = meta_paths(doc_id)
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        chunks = meta["chunks"]

        index = maybe_gpu(faiss.read_index(idx_path))
        q_emb = embed(self.model, [query])

        with st.spinner("🔍 Retrieving candidates..."):
            _, I = index.search(q_emb, min(top_k * 4, index.ntotal))
            candidates = [chunks[i] for i in I[0] if i != -1]

        with st.spinner("🎯 Reranking results..."):
            scores = self.reranker.predict([(query, c) for c in candidates])
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        return [c for c, _ in ranked[:top_k]]

    def extractive_summary(self, doc_id: str, k: int) -> str:
        meta_path, _ = meta_paths(doc_id)
        sentences = json.load(open(meta_path, "r", encoding="utf-8"))["sentences"]
        emb = embed(self.model, sentences)

        centroid = emb.mean(axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)
        _, I = idx.search(centroid, min(k, len(sentences)))
        return " ".join(sentences[i] for i in I[0])


# =========================
# Streamlit UI (Animated)
# =========================
st.set_page_config(page_title="Advanced PDF Analyzer", layout="wide")
st.title("📄 Advanced PDF Analyzer")
st.caption("FAISS • Sentence Transformers • Reranking • Animations")

@st.cache_resource

def get_analyzer():
    return EfficientPDFAnalyzer()

analyzer = get_analyzer()

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    progress = st.progress(0)
    progress.progress(20)
    meta = analyzer.index_pdf(uploaded)
    progress.progress(100)
    st.success(f"✅ Indexed {meta['count']} chunks")
    st.session_state["doc_id"] = meta["doc_id"]

if "doc_id" in st.session_state:
    st.divider()
    query = st.text_input("Ask a question about the document")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Search"):
            results = analyzer.search(query, st.session_state["doc_id"])
            for r in results:
                st.markdown(f"- {r}")

    with col2:
        if st.button("📌 Generate Summary"):
            with st.spinner("📝 Summarizing..."):
                summary = analyzer.extractive_summary(st.session_state["doc_id"], SUMMARY_SENTENCES)
            st.info(summary)
