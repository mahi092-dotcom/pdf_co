"""
Advanced PDF Analyzer - Single-file Streamlit app (fixed)
Fixes the AttributeError caused by using Streamlit's experimental decorator
on an instance method. This version uses a simple on-disk embedding cache
to avoid Streamlit decorator issues and to be safe for background threads.

Features:
- Background indexing with progress callbacks and staged Lottie animations
- HNSW index with tuned parameters and saved meta
- On-disk embedding caching (safe for threads)
- Batched reranker scoring and limited candidate reranking
- Extractive summary using centroid + positional bias
- Responsive UI with progress bar, placeholders, and multiple animations

Notes:
- Replace placeholder implementations (read_pdf_text, split_sentences, batch_encode,
  reranker.predict, show_lottie, and Lottie constants) with your real implementations.
- Requires: streamlit, numpy, faiss (faiss-cpu or faiss-gpu), sentence-transformers or similar
"""

import os
import json
import tempfile
import concurrent.futures
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import faiss
import streamlit as st

# -------------------------
# Configuration / Constants
# -------------------------
STORE_DIR = os.environ.get("PDF_STORE_DIR", "./pdf_store")
os.makedirs(STORE_DIR, exist_ok=True)

# HNSW tuning defaults (tune for your dataset)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
EF_SEARCH_DEFAULT = 128

SUMMARY_SENTENCES = 5
BATCH_ENCODE_SIZE = 64
RERANKER_MULTIPLIER = 3  # fetch top_k * multiplier candidates before reranking

# Placeholder Lottie animation keys (replace with actual Lottie objects or loader)
WELCOME_ROBOT = "WELCOME_ROBOT"
ROCKET_INDEX = "ROCKET_INDEX"
ROCKET_WORKING = "ROCKET_WORKING"
FIREWORKS = "FIREWORKS"
CONFETTI_BURST = "CONFETTI_BURST"
SEARCH_EYE = "SEARCH_EYE"
THUMBS_UP = "THUMBS_UP"
PARTY_POPPER = "PARTY_POPPER"

# -------------------------
# Helper utilities
# -------------------------
def meta_paths(doc_id: str, store_dir: str = STORE_DIR) -> Tuple[str, str]:
    meta_path = os.path.join(store_dir, f"{doc_id}.meta.json")
    idx_path = os.path.join(store_dir, f"{doc_id}.index.faiss")
    return meta_path, idx_path

def load_meta(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meta(path: str, sentences: List[str], ids: List[int], extra: Dict[str, Any]):
    meta = {"sentences": sentences, "ids": ids}
    meta.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_index(path: str) -> faiss.Index:
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)

def embeddings_cache_path(doc_id: str) -> str:
    return os.path.join(STORE_DIR, f"{doc_id}.embeddings.npy")

# -------------------------
# Placeholder implementations
# -------------------------
# Replace these with your real implementations.

def read_pdf_text(pdf_file) -> str:
    """
    Read PDF bytes/file-like and return extracted text.
    Replace with pdfminer, PyMuPDF, or other robust extractor.
    """
    try:
        # If streamlit uploaded file, it's a BytesIO-like object
        content = pdf_file.read()
        # Very naive fallback: decode bytes to text
        text = content.decode("utf-8", errors="ignore")
        return text
    except Exception:
        return ""

def split_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter. Replace with a better NLP sentence tokenizer.
    """
    if not text:
        return []
    # Split on newlines and periods; keep short filtering
    raw = [s.strip() for s in text.replace("\r", "\n").split("\n") if s.strip()]
    sentences = []
    for block in raw:
        parts = [p.strip() for p in block.split(".") if p.strip()]
        for p in parts:
            if len(p) > 20:  # filter out very short fragments
                sentences.append(p + ".")
    return sentences

@dataclass
class DummyModel:
    """
    Replace with your real embedding model (e.g., SentenceTransformer).
    Must implement encode(list[str], convert_to_numpy=True, normalize_embeddings=True)
    """
    dim: int = 384

    def encode(self, texts: List[str], convert_to_numpy=True, normalize_embeddings=True) -> np.ndarray:
        # Dummy deterministic embeddings for example purposes
        rng = np.random.RandomState(42)
        emb = rng.randn(len(texts), self.dim).astype("float32")
        # simple normalization
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        return emb

class DummyReranker:
    """
    Replace with your real reranker. Must accept a list of (query, candidate) pairs
    and return a list/np.array of scores (higher = better).
    """
    def predict(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        # Dummy scoring: random but deterministic
        rng = np.random.RandomState(123)
        return rng.rand(len(pairs)).astype("float32")

def show_lottie(lottie_obj, height=200, key=None):
    """
    Placeholder for your Lottie display function.
    Replace with st_lottie or your wrapper that renders Lottie animations.
    For this single-file example, we'll just show an empty placeholder box.
    """
    return st.empty()

# -------------------------
# Embedding cache (on-disk)
# -------------------------
def get_or_compute_embeddings_on_disk(doc_id: str, sentences: List[str], model: DummyModel) -> np.ndarray:
    """
    Thread-safe on-disk cache for embeddings.
    - If cache exists, load and return.
    - Otherwise compute embeddings in batches, normalize, save to .npy, and return.
    """
    cache_path = embeddings_cache_path(doc_id)
    if os.path.exists(cache_path):
        try:
            emb = np.load(cache_path)
            # ensure dtype float32
            if emb.dtype != np.float32:
                emb = emb.astype("float32")
            return emb
        except Exception:
            # corrupted cache: remove and recompute
            try:
                os.remove(cache_path)
            except Exception:
                pass

    # compute embeddings in batches
    all_embs = []
    for i in range(0, len(sentences), BATCH_ENCODE_SIZE):
        batch = sentences[i : i + BATCH_ENCODE_SIZE]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        all_embs.append(emb)
    embeddings = np.vstack(all_embs)
    # normalize to unit length
    faiss.normalize_L2(embeddings)
    # save atomically
    tmp_path = cache_path + ".tmp"
    try:
        np.save(tmp_path, embeddings)
        os.replace(tmp_path, cache_path)
    except Exception:
        # best-effort save; ignore failures
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return embeddings

# -------------------------
# EfficientPDFAnalyzer
# -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, store_dir: str = STORE_DIR):
        self.store_dir = store_dir
        self.model = None
        self.reranker = None
        self._models_loaded = False
        # default HNSW params
        self.hnsw_params = {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION}

    def _load_models(self):
        if self._models_loaded:
            return
        # Replace DummyModel with your real model loader
        self.model = DummyModel(dim=384)
        self.reranker = DummyReranker()
        self._models_loaded = True

    def index_pdf(self, pdf_source, doc_id: str = None, reindex: bool = False, progress_callback=None) -> Dict[str, Any]:
        """
        Index a PDF. If progress_callback provided, call progress_callback(percent:int, message:str).
        This function is safe to run in a background thread.
        """
        self._load_models()
        if doc_id is None:
            doc_id = next(tempfile._get_candidate_names())
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)

        # If not reindex and index exists, load meta and return
        if not reindex:
            meta = load_meta(meta_path)
            index = load_index(idx_path)
            if index is not None and meta.get("sentences"):
                if progress_callback:
                    progress_callback(100, "Already indexed")
                return {"status": "loaded", "doc_id": doc_id, "count": len(meta["sentences"])}

        if progress_callback:
            progress_callback(5, "Reading PDF")
        text = read_pdf_text(pdf_source)
        if progress_callback:
            progress_callback(15, "Splitting sentences")
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("No readable sentences extracted")

        if progress_callback:
            progress_callback(30, "Computing embeddings")
        # Use on-disk cached embeddings (thread-safe)
        embeddings = get_or_compute_embeddings_on_disk(doc_id, sentences, self.model)

        if progress_callback:
            progress_callback(60, "Building HNSW index")
        # Build HNSW index with tuned params
        d = embeddings.shape[1]
        M = self.hnsw_params.get("M", HNSW_M)
        efc = self.hnsw_params.get("efConstruction", HNSW_EF_CONSTRUCTION)
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = efc
        # embeddings are already normalized
        index.add(embeddings.astype("float32"))

        if progress_callback:
            progress_callback(85, "Saving index and metadata")
        ids = list(range(len(sentences)))
        save_index(index, idx_path)
        save_meta(meta_path, sentences, ids, {"index_type": "HNSW", "M": M, "efConstruction": efc})
        if progress_callback:
            progress_callback(100, "Indexing complete")
        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences)}

    def search(self, query: str, doc_id: str, top_k: int = 5) -> List[str]:
        """
        Search the indexed document and return top_k sentences.
        Uses ANN to fetch candidates and a reranker to refine results.
        """
        self._load_models()
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")
        index = load_index(idx_path)
        if index is None:
            raise ValueError("Index file missing")

        # tune efSearch based on top_k
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = min(EF_SEARCH_DEFAULT, max(32, top_k * 20))

        # encode query
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        candidate_count = min(top_k * RERANKER_MULTIPLIER, index.ntotal)
        if candidate_count <= 0:
            return []

        _, I = index.search(q_emb, candidate_count)
        candidate_indices = [int(idx) for idx in I[0] if idx != -1]
        candidates = [sentences[idx] for idx in candidate_indices]
        if not candidates:
            return []

        # Batch reranker scoring
        pairs = [(query, cand) for cand in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in ranked[:top_k]]

    def extractive_summary(self, doc_id: str, num_sentences: int = SUMMARY_SENTENCES) -> str:
        """
        Simple extractive summary: compute centroid of sentence embeddings,
        bias by position, and pick top sentences closest to centroid.
        """
        self._load_models()
        meta_path, _ = meta_paths(doc_id, self.store_dir)
        meta = load_meta(meta_path)
        sentences = meta.get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")

        # get cached embeddings from disk
        embeddings = get_or_compute_embeddings_on_disk(doc_id, sentences, self.model)
        # centroid
        centroid = np.mean(embeddings, axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(centroid)

        # positional bias
        positions = np.arange(len(sentences)) / max(1, len(sentences))
        pos_bias = 1.2 - np.abs(positions - 0.1) - np.abs(positions - 0.9)
        pos_bias = np.maximum(pos_bias, 0.8)
        biased_emb = embeddings * (1 + pos_bias[:, np.newaxis] * 0.3)

        # search top candidates by inner product to centroid
        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(biased_emb)
        tmp.add(biased_emb)
        k = min(num_sentences * 3, len(sentences))
        _, I = tmp.search(centroid, k)
        top_indices = sorted(I[0][:num_sentences], key=lambda x: x)
        return " ".join(sentences[int(i)] for i in top_indices)

# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title="Advanced PDF Analyzer", layout="centered")
st.title("📄 Advanced PDF Analyzer – Now with More Animations! 🎉")

# show welcome animation
show_lottie(WELCOME_ROBOT, height=200, key="welcome")

analyzer = EfficientPDFAnalyzer()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# UI placeholders for progress and animations
progress_bar = st.empty()
status_text = st.empty()
lottie_slot = st.empty()

def ui_progress_callback(pct: int, msg: str):
    """
    UI callback used by background indexing to update progress bar and animations.
    We store progress in session_state and let the main thread render it.
    """
    st.session_state["_index_progress"] = {"pct": int(pct), "msg": msg}

# Background indexing runner
def run_index_background(analyzer_obj, uploaded_file_obj, doc_id=None, reindex=False):
    try:
        result = analyzer_obj.index_pdf(uploaded_file_obj, doc_id=doc_id, reindex=reindex, progress_callback=ui_progress_callback)
        st.session_state["index_result"] = result
    except Exception as e:
        st.session_state["index_error"] = str(e)

if uploaded_file is not None:
    # Reset previous progress state
    st.session_state.pop("_index_progress", None)
    st.session_state.pop("index_result", None)
    st.session_state.pop("index_error", None)

    # Start background indexing when user clicks "Start Indexing" or automatically
    if st.button("Start Indexing", use_container_width=True):
        # show initial UI
        progress_bar.progress(0)
        status_text.text("Queued for indexing...")
        show_lottie(ROCKET_INDEX, height=220, key="index_start")
        # run in background thread
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(run_index_background, analyzer, uploaded_file)
        st.info("Indexing started in background. Progress will appear below.")

    # Poll UI for progress updates
    if "_index_progress" in st.session_state:
        prog = st.session_state["_index_progress"]
        progress_bar.progress(prog["pct"])
        status_text.text(prog["msg"])
        # swap animations at milestones
        if prog["pct"] < 30:
            show_lottie(ROCKET_INDEX, height=200, key="indexing_early")
        elif prog["pct"] < 90:
            show_lottie(ROCKET_WORKING, height=220, key="indexing_mid")
        else:
            show_lottie(FIREWORKS, height=250, key="indexing_done")

    if "index_result" in st.session_state:
        res = st.session_state.pop("index_result")
        st.success(f"Indexed {res['count']} sentences successfully!")
        st.session_state["doc_id"] = res["doc_id"]
        show_lottie(CONFETTI_BURST, height=250, key="index_confetti")
        progress_bar.empty()
        status_text.empty()

    if "index_error" in st.session_state:
        st.error(f"Indexing error: {st.session_state.pop('index_error')}")
        progress_bar.empty()
        status_text.empty()

# If a doc is already indexed in session, show search and summary UI
if "doc_id" in st.session_state:
    st.markdown("---")
    st.markdown("### Document Tools")
    query = st.text_input("🔍 Ask anything about the document")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Document", use_container_width=True):
            if not query:
                st.info("Please enter a query to search.")
            else:
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
                    show_lottie(CONFETTI_BURST, height=250, key="summary_confetti")
                except Exception as e:
                    st.error(f"Summary error: {e}")

# Footer / tips
st.markdown("---")
st.markdown(
    """
**Tips**
- Use the "Start Indexing" button after uploading a PDF to run indexing in the background.
- Tune HNSW parameters in the top of the file (`HNSW_M`, `HNSW_EF_CONSTRUCTION`, `EF_SEARCH_DEFAULT`) for your dataset.
- Replace placeholder functions (PDF reading, sentence splitting, model, reranker, and Lottie rendering) with your production implementations.
- The embedding cache is stored as `.npy` files in the store directory; delete them to force recomputation.
"""
)
