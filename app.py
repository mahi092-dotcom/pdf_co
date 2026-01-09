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

# More diverse & vibrant Lottie animations (no repetition)
WELCOME_ROBOT       = "https://assets5.lottiefiles.com/packages/lf20_kkflmtur.json"   # Friendly robot
DATA_LOADING        = "https://assets6.lottiefiles.com/packages/lf20_afwjvtdp.json"   # Data processing
ROCKET_INDEX        = "https://assets8.lottiefiles.com/packages/lf20_6nfjl2av.json"   # Rocket upload
SEARCH_EYE          = "https://assets4.lottiefiles.com/packages/lf20_Uz4uNe.json"     # Searching eye
FIREWORKS           = "https://assets10.lottiefiles.com/packages/lf20_towptqfc.json"  # Fireworks
PARTY_POPPER        = "https://assets1.lottiefiles.com/packages/lf20_yM3Lp0P7.json"   # Party popper
CONFETTI_BURST      = "https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json"   # Confetti burst
THUMBS_UP           = "https://assets2.lottiefiles.com/packages/lf20_oxmdw2.json"     # Thumbs up success

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
        # Varied built-in fallbacks
        if "rocket" in url or "fireworks" in url or "party" in url or "confetti" in url:
            st.balloons()
        elif "thumbs" in url:
            st.success("✓ Awesome!")
        else:
            st.snow()  # Light celebration

# ------------------------- Config -------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STORE_DIR = "store"
SUMMARY_SENTENCES = 5
EF_SEARCH = 64  # Good accuracy on CPU

# ------------------------- Helpers (unchanged) -------------------------
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

def build_index(embeddings: np.ndarray) -> faiss.IndexIDMap2:
    dim = embeddings.shape[1]
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)
    hnsw_index.hnsw.efConstruction = 200
    return faiss.IndexIDMap2(hnsw_index)

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_index(path: str) -> Optional[faiss.Index]:
    return faiss.read_index(path) if os.path.exists(path) else None

# ------------------------- Analyzer Class (CPU-only, lazy load) -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, store_dir: str = STORE_DIR):
        self.model_name = model_name
        self.rerank_model = RERANK_MODEL
        self.store_dir = store_dir
        ensure_dir(self.store_dir)
        self.model = None  # Lazy load
        self.reranker = None  # Lazy load

    def _load_models(self):
        if self.model is None:
            with st.spinner("Loading embedding model..."):
                show_lottie(DATA_LOADING, height=150, key="load_embed")
                self.model = SentenceTransformer(self.model_name, device="cpu")
        if self.reranker is None:
            with st.spinner("Loading reranker model..."):
                self.reranker = CrossEncoder(self.rerank_model, device="cpu")

    def index_pdf(self, pdf_source, doc_id: Optional[str] = None, reindex: bool = False) -> dict:
        self._load_models()
        inferred_id = stable_doc_id(getattr(pdf_source, "name", "uploaded.pdf"))
        doc_id = doc_id or inferred_id
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)

        if not reindex:
            meta = load_meta(meta_path)
            index = load_index(idx_path)
            if index is not None and meta.get("sentences"):
                return {"status": "loaded", "doc_id": doc_id, "count": len(meta["sentences"])}

        sentences = split_sentences(read_pdf_text(pdf_source))
        if not sentences:
            raise ValueError("No readable sentences extracted")

        embeddings = batch_encode(self.model, sentences)
        base_index = build_index(embeddings)
        ids = np.arange(len(sentences), dtype="int64")
        base_index.add_with_ids(embeddings, ids)

        save_index(base_index, idx_path)
        save_meta(meta_path, sentences, ids.tolist(), {"index_type": "HNSW"})
        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences)}

    def search(self, query: str, doc_id: str, top_k: int = 5) -> List[str]:
        self._load_models()
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")
        index = load_index(idx_path)
        if index is None:
            raise ValueError("Index file missing")

        if hasattr(index, 'hnsw'):
            index.hnsw.efSearch = EF_SEARCH

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
        self._load_models()
        meta_path, _ = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")

        cache_key = f"{doc_id}_summary_emb"
        if cache_key not in st.session_state:
            embeddings = batch_encode(self.model, sentences)
            st.session_state[cache_key] = embeddings
        else:
            embeddings = st.session_state[cache_key]

        centroid = np.mean(embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(centroid)

        positions = np.arange(len(sentences)) / len(sentences)
        pos_bias = 1.2 - np.abs(positions - 0.1) - np.abs(positions - 0.9)
        pos_bias = np.maximum(pos_bias, 0.8)
        biased_emb = embeddings * (1 + pos_bias[:, np.newaxis] * 0.3)

        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        tmp.add(biased_emb)
        _, I = tmp.search(centroid, min(num_sentences * 3, len(sentences)))
        
        top_indices = sorted(I[0][:num_sentences], key=lambda x: x)
        return " ".join(sentences[int(i)] for i in top_indices)

# ------------------------- Streamlit App -------------------------
st.set_page_config(page_title="Advanced PDF Analyzer", layout="centered")
st.title("📄 Advanced PDF Analyzer – Now with More Animations! 🎉")

show_lottie(WELCOME_ROBOT, height=200, key="welcome")

analyzer = EfficientPDFAnalyzer()  # No heavy load on init

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Indexing your PDF..."):
        show_lottie(ROCKET_INDEX, height=250, key="indexing")
        try:
            meta = analyzer.index_pdf(uploaded_file)
            st.success(f"Indexed {meta['count']} sentences successfully!")
            st.session_state["doc_id"] = meta["doc_id"]
            show_lottie(FIREWORKS, height=300, key="index_fireworks")
            show_lottie(CONFETTI_BURST, height=250, key="index_confetti")  # Extra animation!
        except Exception as e:
            st.error(f"Error: {e}")

if "doc_id" in st.session_state:
    st.markdown("---")
    query = st.text_input("🔍 Ask anything about the document")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Document", use_container_width=True):
            show_lottie(SEARCH_EYE, height=180, key="searching")
            try:
                results = analyzer.search(query, st.session_state["doc_id"])
                st.markdown("### Search Results")
                if results:
                    for i, sent in enumerate(results, 1):
                        st.write(f"{i}. {sent}")
                    show_lottie(THUMBS_UP, height=120, key="search_thumbs")
                else:
                    st.info("No matches found.")
            except Exception as e:
                st.error(f"Search error: {e}")

    with col2:
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                try:
                    summary = analyzer.extractive_summary(st.session_state["doc_id"])
                    st.markdown("### 📌 Extractive Summary")
                    st.write(summary)
                    show_lottie(PARTY_POPPER, height=300, key="summary_party")
                    show_lottie(CONFETTI_BURST, height=250, key="summary_confetti")  # Extra!
                except Exception as e:
                    st.error(f"Summary error: {e}")
