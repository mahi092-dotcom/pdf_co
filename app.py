import os, json, hashlib, re, time
import numpy as np, faiss, streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Optional

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
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def stable_doc_id(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{h}"

def read_pdf_text(pdf_source) -> str:
    reader = PdfReader(pdf_source)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def split_sentences(text: str): return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def batch_encode(model, sentences): 
    emb = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, batch_size=32).astype("float32")
    faiss.normalize_L2(emb)
    return emb

def build_index(embeddings: np.ndarray) -> faiss.IndexIDMap2:
    dim = embeddings.shape[1]
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)
    return faiss.IndexIDMap2(hnsw_index)

# -------------------------
# Analyzer
# -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, store_dir: str = STORE_DIR):
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.store_dir = store_dir
        ensure_dir(self.store_dir)

    def index_pdf(self, pdf_source, doc_id=None, reindex=False):
        inferred_id = stable_doc_id(getattr(pdf_source, "name", "uploaded.pdf"))
        doc_id = doc_id or inferred_id
        sentences = split_sentences(read_pdf_text(pdf_source))
        if not sentences: raise ValueError("No readable sentences extracted")

        embeddings = batch_encode(self.model, sentences)
        index = build_index(embeddings)
        ids = np.arange(len(sentences), dtype="int64")
        index.add_with_ids(embeddings, ids)

        st.session_state["meta"] = {"doc_id": doc_id, "sentences": sentences, "index": index}
        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences)}

    def search(self, query: str, top_k: int = 3):
        meta = st.session_state.get("meta", {})
        if not meta: raise ValueError("Document not indexed")
        sentences, index = meta["sentences"], meta["index"]

        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        _, I = index.search(q_emb, min(top_k*3, index.ntotal))
        candidates = [sentences[int(idx)] for idx in I[0] if idx != -1]

        pairs = [(query, cand) for cand in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in ranked[:top_k]]

    def extractive_summary(self, num_sentences=SUMMARY_SENTENCES):
        meta = st.session_state.get("meta", {})
        if not meta: raise ValueError("Document not indexed")
        sentences = meta["sentences"]
        embeddings = batch_encode(self.model, sentences)
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        tmp.add(embeddings)
        _, I = tmp.search(centroid, min(num_sentences, len(sentences)))
        return " ".join(sentences[int(i)] for i in I[0])

# -------------------------
# Streamlit App with Animations
# -------------------------
st.title("📄 Advanced PDF Analyzer (Animated)")

analyzer = EfficientPDFAnalyzer()

# File browsing animation
st.info("📂 Ready to browse files... Choose a PDF to analyze!")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("⚙️ Indexing your PDF..."):
        progress = st.progress(0)
        for i in range(1, 101, 25):
            time.sleep(0.2)
            progress.progress(i)
        try:
            meta = analyzer.index_pdf(uploaded_file)
            st.success(f"✅ Indexed {meta['count']} sentences from {uploaded_file.name}")
            st.session_state["doc_id"] = meta["doc_id"]
            st.balloons()
        except Exception as e:
            st.error(f"Error indexing PDF: {e}")

if "doc_id" in st.session_state:
    query = st.text_input("Enter search query")
    if st.button("Search"):
        with st.spinner("🔍 Searching..."):
            time.sleep(1)
            try:
                results = analyzer.search(query, top_k=5)
                st.success("✨ Results ready!")
                st.write("### 🔍 Search Results")
                for sentence in results:
                    st.markdown(f"- 🚀 {sentence}")
            except Exception as e:
                st.error(f"Error during search: {e}")

    if st.button("Generate Summary"):
        with st.spinner("📌 Creating summary..."):
            time.sleep(1.5)
            try:
                summary_text = analyzer.extractive_summary(num_sentences=SUMMARY_SENTENCES)
                # Removed snow animation, replaced with styled message
                st.success("🎉 Summary generated successfully!")
                st.write("### 📌 Extractive Summary")
                st.info(summary_text)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
