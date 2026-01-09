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
# Helpers & Logic
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
    emb = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, batch_size=32).astype("float32")
    faiss.normalize_L2(emb)
    return emb

def load_meta(meta_path: str) -> Dict[str, Any]:
    if not os.path.exists(meta_path): return {}
    with open(meta_path, "r", encoding="utf-8") as f: return json.load(f)

def stable_doc_id(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{h}"

def build_index(embeddings: np.ndarray) -> faiss.IndexIDMap2:
    dim = embeddings.shape[1]
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)
    return faiss.IndexIDMap2(hnsw_index)

# -------------------------
# Gemini Style: UI & Animations
# -------------------------
def inject_ui_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at 50% 50%, #121212 0%, #050505 100%);
        color: #e0e0e0;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: #0ea5e9;
        box-shadow: 0 8px 32px 0 rgba(14, 165, 233, 0.2);
    }

    /* Gradient Text */
    .glow-header {
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 20px;
    }

    /* Badges */
    .metric-badge {
        background: rgba(14, 165, 233, 0.15);
        color: #0ea5e9;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }

    /* Pulse & Shimmer */
    .shimmer {
        height: 12px; width: 100%;
        background: linear-gradient(90deg, #1a1a1a 25%, #333 37%, #1a1a1a 63%);
        background-size: 400% 100%;
        animation: shimmer 1.4s ease-in-out infinite;
        border-radius: 4px; margin: 10px 0;
    }
    @keyframes shimmer { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }
    </style>
    """, unsafe_allow_html=True)

def shimmer_loader(text):
    st.write(f"📡 {text}")
    for _ in range(3): st.markdown('<div class="shimmer"></div>', unsafe_allow_html=True)

def typewriter(text, speed=0.01):
    container = st.empty()
    out = ""
    for ch in text:
        out += ch
        container.markdown(f"#### 🔍 {out}")
        time.sleep(speed)

# -------------------------
# Core Analyzer Class
# -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, store_dir: str = STORE_DIR):
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.store_dir = store_dir
        ensure_dir(self.store_dir)

    def index_pdf(self, pdf_source) -> dict:
        doc_id = stable_doc_id(getattr(pdf_source, "name", "uploaded.pdf"))
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)

        # Skip if already exists
        meta = load_meta(meta_path)
        if os.path.exists(idx_path) and meta.get("sentences"):
            return {"status": "loaded", "doc_id": doc_id, "count": len(meta["sentences"])}

        sentences = split_sentences(read_pdf_text(pdf_source))
        embeddings = batch_encode(self.model, sentences)
        base_index = build_index(embeddings)
        ids = np.arange(len(sentences), dtype="int64")
        base_index.add_with_ids(embeddings, ids)

        faiss.write_index(base_index, idx_path)
        atomic_write_json(meta_path, {"sentences": sentences, "ids": ids.tolist(), "config": {"index": "HNSW"}})
        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences)}

    def search(self, query: str, doc_id: str, top_k: int = 5):
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)
        sentences = load_meta(meta_path).get("sentences", [])
        index = faiss.read_index(idx_path)
        
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        _, I = index.search(q_emb, min(top_k * 3, len(sentences)))
        
        candidates = [sentences[int(idx)] for idx in I[0] if idx != -1]
        pairs = [(query, cand) for cand in candidates]
        scores = self.reranker.predict(pairs)
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]

    def extractive_summary(self, doc_id: str, num_sentences: int = 5) -> str:
        meta_path, _ = meta_paths(doc_id, self.store_dir)
        sentences = load_meta(meta_path).get("sentences", [])
        embeddings = batch_encode(self.model, sentences)
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        tmp.add(embeddings)
        _, I = tmp.search(centroid, min(num_sentences, len(sentences)))
        return " ".join(sentences[int(i)] for i in I[0])

# -------------------------
# Streamlit Interface Execution
# -------------------------
inject_ui_style()
st.markdown('<div class="glow-header">Neural PDF Architect</div>', unsafe_allow_html=True)

analyzer = EfficientPDFAnalyzer()

# Sidebar for metadata
with st.sidebar:
    st.markdown("### 🛠️ Neural Engine")
    if "doc_id" in st.session_state:
        st.markdown(f"""
        <div class="glass-card">
            <small>ACTIVE INDEX</small><br>
            <b>{st.session_state['doc_id'][:15]}...</b><br><br>
            <small>NODES LOADED</small><br>
            <b>{st.session_state.get('count', 0)} Vectors</b>
        </div>
        """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file:
    try:
        with st.status("Initializing Knowledge Graph...", expanded=True) as status:
            st.write("Extracting linguistic layers...")
            meta = analyzer.index_pdf(uploaded_file)
            st.session_state["doc_id"] = meta["doc_id"]
            st.session_state["count"] = meta["count"]
            status.update(label="Index Ready", state="complete", expanded=False)
        st.toast("Document Synchronized!")
    except Exception as e:
        st.error(f"Sync Error: {e}")

if "doc_id" in st.session_state:
    tab1, tab2 = st.tabs(["🔍 Semantic Search", "📝 Synthesis"])
    
    with tab1:
        query = st.text_input("Enter your query...", placeholder="e.g., What are the key findings?")
        if query:
            start = time.time()
            results = analyzer.search(query, st.session_state["doc_id"])
            typewriter(f"Analysis complete in {time.time()-start:.3f}s")
            
            for text, score in results:
                st.markdown(f"""
                <div class="glass-card">
                    <span class="metric-badge">Relevance: {score:.2f}</span>
                    <p style='margin-top:10px;'>{text}</p>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        if st.button("Synthesize Executive Summary"):
            shimmer_loader("Recalculating Centroids...")
            summary = analyzer.extractive_summary(st.session_state["doc_id"])
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #0ea5e9;">
                <h4 style="margin-top:0;">📌 Executive Synthesis</h4>
                {summary}
            </div>
            """, unsafe_allow_html=True)
