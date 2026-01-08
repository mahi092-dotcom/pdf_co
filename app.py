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
CHUNK_SIZE = 300  # tokens per chunk

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

def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def batch_encode(model, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64).astype("float32")
    faiss.normalize_L2(emb)
    return emb

def _maybe_to_gpu(index: faiss.Index) -> faiss.Index:
    if USE_GPU:
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index)
        except Exception: return index
    return index

def build_index(embeddings: np.ndarray) -> faiss.IndexIDMap:
    dim = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dim)   # fast similarity search
    return faiss.IndexIDMap(base_index)   # adds ID support

# -------------------------
# Analyzer
# -------------------------
class UltraEfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(RERANK_MODEL)
        ensure_dir(STORE_DIR)

    def index_pdf(self, pdf_source, doc_id: Optional[str] = None, reindex: bool = False) -> dict:
        inferred_id = stable_doc_id(getattr(pdf_source, "name", "uploaded.pdf"))
        doc_id = doc_id or inferred_id
        text = read_pdf_text(pdf_source)
        chunks = chunk_text(text)
        if not chunks: raise ValueError("No readable text extracted")

        embeddings = batch_encode(self.model, chunks)
        index = build_index(embeddings)
        index = _maybe_to_gpu(index)
        ids = np.arange(len(chunks), dtype="int64")
        index.add_with_ids(embeddings, ids)

        st.session_state["meta"] = {"doc_id": doc_id, "chunks": chunks, "embeddings": embeddings, "index": index}
        return {"status": "indexed", "doc_id": doc_id, "count": len(chunks)}

    def search(self, query: str, top_k: int = 3) -> List[str]:
        meta = st.session_state.get("meta", {})
        if not meta: raise ValueError("Document not indexed")
        chunks, index = meta["chunks"], meta["index"]

        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        _, I = index.search(q_emb, min(top_k*10, index.ntotal))
        candidates = [chunks[int(idx)] for idx in I[0] if idx != -1][:15]

        pairs = [(query, cand) for cand in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in ranked[:top_k]]

    def extractive_summary(self, num_sentences: int = SUMMARY_SENTENCES) -> str:
        meta = st.session_state.get("meta", {})
        if not meta: raise ValueError("Document not indexed")
        chunks, embeddings = meta["chunks"], meta["embeddings"]

        centroid = np.mean(embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        tmp.add(embeddings)
        _, I = tmp.search(centroid, min(num_sentences, len(chunks)))
        return " ".join(chunks[int(i)] for i in I[0])

# -------------------------
# Streamlit App
# -------------------------
st.title("⚡ The Most Efficient PDF Analyzer (After Copilot)")

analyzer = UltraEfficientPDFAnalyzer()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    progress = st.progress(0)
    for i in range(1, 101, 20):
        time.sleep(0.1)
        progress.progress(i)
    try:
        meta = analyzer.index_pdf(uploaded_file)
        st.success(f"✅ Indexed {meta['count']} chunks from {uploaded_file.name}")
        st.session_state["doc_id"] = meta["doc_id"]
        st.balloons()
    except Exception as e:
        st.error(f"Error indexing PDF: {e}")

if "doc_id" in st.session_state:
    query = st.text_input("Enter search query")
    if st.button("Search"):
        with st.spinner("🔍 Retrieving the most relevant answers..."):
            time.sleep(0.5)
            try:
                results = analyzer.search(query, top_k=5)
                st.success("✨ Results ready!")
                st.write("### 🔍 Search Results")
                for sentence in results:
                    st.markdown(f"- 🚀 {sentence}")
            except Exception as e:
                st.error(f"Error during search: {e}")

    if st.button("Generate Summary"):
        with st.spinner("📌 Summarizing document..."):
            time.sleep(1)
            try:
                summary_text = analyzer.extractive_summary(num_sentences=SUMMARY_SENTENCES)
                st.balloons()
                st.write("### 📌 Extractive Summary")
                st.info(summary_text)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
