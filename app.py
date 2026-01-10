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
import torch
from transformers import pipeline

# -------------------------
# Config
# -------------------------
MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Top-notch sentence embedding model
RERANK_MODEL = "BAAI/bge-reranker-large"  # State-of-the-art cross-encoder for reranking
SUMMARIZER_MODEL = "facebook/bart-large-cnn"  # Abstractive summarization model
STORE_DIR = "store"
USE_GPU = True
CHUNK_SIZE = 1000  # Chunk size for handling long texts in summarization (characters)
SUMMARY_MAX_LENGTH = 300  # Max length for final summary
SUMMARY_MIN_LENGTH = 100  # Min length for final summary
HNSW_M = 64  # For FAISS HNSW
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stable_doc_id(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{h}"

def read_pdf_text(pdf_source) -> str:
    reader = PdfReader(pdf_source)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def batch_encode(model, sentences: List[str], device: str) -> np.ndarray:
    emb = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, device=device).astype("float32")
    faiss.normalize_L2(emb)
    return emb

def build_index(embeddings: np.ndarray) -> faiss.IndexIDMap2:
    dim = embeddings.shape[1]
    hnsw_index = faiss.IndexHNSWFlat(dim, HNSW_M)
    hnsw_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    return faiss.IndexIDMap2(hnsw_index)

# -------------------------
# Analyzer
# -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, store_dir: str = STORE_DIR):
        self.device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=0 if self.device == 'cuda' else -1)
        self.store_dir = store_dir
        ensure_dir(self.store_dir)

    def load_if_exists(self, doc_id: str) -> bool:
        emb_path = os.path.join(self.store_dir, f"{doc_id}.emb.npy")
        sent_path = os.path.join(self.store_dir, f"{doc_id}.sent.json")
        idx_path = os.path.join(self.store_dir, f"{doc_id}.index")
        if all(os.path.exists(p) for p in [emb_path, sent_path, idx_path]):
            embeddings = np.load(emb_path)
            with open(sent_path, 'r') as f:
                sentences = json.load(f)
            index = faiss.read_index(idx_path)
            st.session_state["meta"] = {
                "doc_id": doc_id,
                "sentences": sentences,
                "index": index,
                "embeddings": embeddings
            }
            return True
        return False

    def save_to_disk(self, doc_id: str, sentences: List[str], embeddings: np.ndarray, index: faiss.IndexIDMap2):
        emb_path = os.path.join(self.store_dir, f"{doc_id}.emb.npy")
        sent_path = os.path.join(self.store_dir, f"{doc_id}.sent.json")
        idx_path = os.path.join(self.store_dir, f"{doc_id}.index")
        np.save(emb_path, embeddings)
        with open(sent_path, 'w') as f:
            json.dump(sentences, f)
        faiss.write_index(index, idx_path)

    def index_pdf(self, pdf_source, doc_id: Optional[str] = None, reindex: bool = False) -> dict:
        inferred_id = stable_doc_id(getattr(pdf_source, "name", "uploaded.pdf"))
        doc_id = doc_id or inferred_id
        if not reindex and self.load_if_exists(doc_id):
            return {"status": "loaded from disk", "doc_id": doc_id, "count": len(st.session_state["meta"]["sentences"])}
        sentences = split_sentences(read_pdf_text(pdf_source))
        if not sentences:
            raise ValueError("No readable sentences extracted")
        embeddings = batch_encode(self.model, sentences, self.device)
        index = build_index(embeddings)
        ids = np.arange(len(sentences), dtype="int64")
        index.add_with_ids(embeddings, ids)
        self.save_to_disk(doc_id, sentences, embeddings, index)
        st.session_state["meta"] = {
            "doc_id": doc_id,
            "sentences": sentences,
            "index": index,
            "embeddings": embeddings
        }
        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences)}

    def search(self, query: str, top_k: int = 5) -> List[str]:
        meta = st.session_state.get("meta", {})
        if not meta:
            raise ValueError("Document not indexed")
        sentences, index = meta["sentences"], meta["index"]
        if hasattr(index.index, 'hnsw'):
            index.index.hnsw.efSearch = HNSW_EF_SEARCH
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, device=self.device).astype("float32")
        _, I = index.search(q_emb, min(top_k * 3, index.ntotal))
        candidates = [sentences[int(idx)] for idx in I[0] if idx != -1]
        pairs = [(query, cand) for cand in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in ranked[:top_k]]

    def summarize_long_text(self, text: str, chunk_size: int = CHUNK_SIZE, max_length: int = SUMMARY_MAX_LENGTH, min_length: int = SUMMARY_MIN_LENGTH) -> str:
        # Split into overlapping chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - 200)]  # 200 char overlap
        summaries = []
        chunk_max = max(130, max_length // len(chunks) + 1)  # Dynamic max_length per chunk
        chunk_min = max(30, min_length // len(chunks) + 1)
        for chunk in chunks:
            if chunk.strip():
                summary = self.summarizer(chunk, max_length=chunk_max, min_length=chunk_min, do_sample=False)[0]['summary_text']
                summaries.append(summary)
        combined = " ".join(summaries)
        # If combined is still too long, recursively summarize
        if len(combined) > chunk_size * 1.5:
            return self.summarize_long_text(combined, chunk_size, max_length, min_length)
        else:
            return self.summarizer(combined, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def abstractive_summary(self, max_length: int = SUMMARY_MAX_LENGTH, min_length: int = SUMMARY_MIN_LENGTH) -> str:
        meta = st.session_state.get("meta", {})
        if not meta:
            raise ValueError("Document not indexed")
        full_text = " ".join(meta["sentences"])
        return self.summarize_long_text(full_text, max_length=max_length, min_length=min_length)

# -------------------------
# Streamlit App with Animations
# -------------------------
st.title("📄 Most Effective PDF Analyzer (with Abstractive Summarization)")

analyzer = EfficientPDFAnalyzer()

# File browsing animation
st.info("📂 Ready to browse files... Upload a PDF to analyze!")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Settings gear spinner during indexing
    with st.spinner("⚙️ Indexing your PDF... Please wait"):
        try:
            meta = analyzer.index_pdf(uploaded_file)
            st.success(f"✅ {meta['status'].capitalize()} {meta['count']} sentences from {uploaded_file.name}")
            st.session_state["doc_id"] = meta["doc_id"]
            st.balloons()  # 🎈 celebration
        except Exception as e:
            st.error(f"Error indexing PDF: {e}")

if "doc_id" in st.session_state:
    query = st.text_input("Enter search query")
    if st.button("Search"):
        # Simple search animation
        with st.spinner("🔍 Searching your document..."):
            time.sleep(1)  # simulate animation
            try:
                results = analyzer.search(query, top_k=5)
                st.success("✨ Results ready!")
                st.write("### 🔍 Search Results")
                for sentence in results:
                    st.markdown(f"- 🚀 {sentence}")
            except Exception as e:
                st.error(f"Error during search: {e}")

    if st.button("Generate Abstractive Summary"):
        with st.spinner("📌 Creating abstractive summary..."):
            time.sleep(1.2)
            try:
                summary_text = analyzer.abstractive_summary()
                st.success("🎉 Abstractive summary generated successfully!")
                st.write("### 📌 Abstractive Summary")
                st.info(summary_text)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
