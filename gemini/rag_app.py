# FILE: rag_app.py
import os
import re
import io
import json
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker
from dotenv import load_dotenv
load_dotenv()

# --- Config ---
# MODIFIED: Load our new combined corpus
DEFAULT_MD_PATH = "out/combined_corpus.md"
INDEX_DIR = "index_combined" # Use a new index dir
os.makedirs(INDEX_DIR, exist_ok=True)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 8
DEFAULT_LANG = "en"

# --- Data structures ---
@dataclass
class ProductDoc:
    doc_id: str
    title: str
    url: Optional[str]
    source: Optional[str] # NEW: 'Daraz' or 'Startech'
    category: Optional[str]
    brand: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    raw_md: str

@dataclass
class ChunkRec:
    doc_id: str
    title: str
    url: Optional[str]
    source: Optional[str] # NEW
    category: Optional[str]
    brand: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    text: str

# --- Regex helpers ---
DOC_BLOCK_RE = re.compile(r"(?P<body>.*?)", re.DOTALL|re.IGNORECASE)
TITLE_RE = re.compile(r"^##\s+(.+?)\s*(?:\s{2,}\n|\n|$)", re.MULTILINE)
URL_LINE_RE = re.compile(r"\*\*URL:\*\*\s*(\S+)", re.IGNORECASE)
SOURCE_RE = re.compile(r"\*\*Source:\*\*\s*(.+)", re.IGNORECASE) # NEW
BRAND_RE = re.compile(r"\*\*Brand:\*\*\s*(.+)", re.IGNORECASE)
PRICE_RE = re.compile(r"\*\*Price:\*\*\s*(.+)", re.IGNORECASE)
RATING_RE = re.compile(r"\*\*Rating:\*\*\s*([0-9.]+)\s*/\s*5(?:\s*\((\d+)\s*ratings\))?", re.IGNORECASE)

# (Keep _meta_from_header and _parse_price_value functions as they are)
# ... PASTE _meta_from_header and _parse_price_value here ...

def parse_products_from_md(md_text: str) -> List[ProductDoc]:
    """Pulls each product between DOC markers and extracts light metadata."""
    products: List[ProductDoc] = []
    for m in DOC_BLOCK_RE.finditer(md_text):
        attrs = m.group("attrs") or ""
        body = (m.group("body") or "").strip()
        meta = _meta_from_header(attrs)
        doc_id = meta.get("id") or "doc_" + str(len(products)+1)
        category = meta.get("category")

        title_m = TITLE_RE.search(body)
        title = (title_m.group(1).strip() if title_m else f"Product {doc_id}")

        url_m = URL_LINE_RE.search(body)
        url = url_m.group(1).strip() if url_m else None
        
        # NEW: Parse the Source
        source_m = SOURCE_RE.search(body)
        source = source_m.group(1).strip() if source_m else None

        brand_m = BRAND_RE.search(body)
        brand = brand_m.group(1).strip() if brand_m else None

        price_value = None
        price_m = PRICE_RE.search(body)
        if price_m:
            price_value = _parse_price_value(price_m.group(1))

        rating_avg, rating_cnt = None, None
        rating_m = RATING_RE.search(body)
        if rating_m:
            try: rating_avg = float(rating_m.group(1))
            except Exception: rating_avg = None
            try: rating_cnt = int(rating_m.group(2)) if rating_m.group(2) else None
            except Exception: rating_cnt = None

        products.append(ProductDoc(
            doc_id=doc_id, title=title, url=url,
            source=source, # NEW
            category=category, brand=brand, price_value=price_value,
            rating_avg=rating_avg, rating_cnt=rating_cnt, raw_md=body
        ))
    return products

# --- Chunking (Chonkie) ---
# (Keep build_chunker function as-is)
# ... PASTE build_chunker here ...

def product_to_chunks(product: ProductDoc, chunker: RecursiveChunker) -> List[ChunkRec]:
    chunks = []
    try:
        chonks = chunker(product.raw_md)
    except Exception:
        split = [s.strip() for s in re.split(r"\n{2,}", product.raw_md) if s.strip()]
        chonks = [{"text": s} for s in split]

    for c in chonks:
        text = (getattr(c, "text", None) or (c["text"] if isinstance(c, dict) else "")).strip()
        if not text: continue
        indexed_text = _clean_for_bm25(text) # _clean_for_bm25 is defined below
        if not indexed_text: continue

        chunks.append(ChunkRec(
            doc_id=product.doc_id, title=product.title, url=product.url,
            source=product.source, # NEW
            category=product.category, brand=product.brand,
            price_value=product.price_value, rating_avg=product.rating_avg, rating_cnt=product.rating_cnt,
            text=indexed_text
        ))
    return chunks

# --- BM25 indexing ---
# (Keep STOPWORDS, _clean_for_bm25, _tokenize, _sha1, _index_paths functions as-is)
# ... PASTE STOPWORDS, _clean_for_bm25, _tokenize, _sha1, _index_paths here ...

def build_or_load_bm25(products: List[ProductDoc], lang: str) -> Tuple[BM25Okapi, List[ChunkRec], List[List[str]]]:
    # (This function remains identical to your original)
    # ... PASTE build_or_load_bm25 function here ...

def _passes_filters(chunk: ChunkRec,
                    allowed_sources: Optional[set], # NEW
                    allowed_categories: Optional[set],
                    brand_filter: Optional[str],
                    price_min: Optional[float],
                    price_max: Optional[float],
                    rating_min: Optional[float]) -> bool:
    """Apply facet filters to a chunk."""
    # NEW: Filter by source (Daraz/Startech)
    if allowed_sources and (chunk.source not in allowed_sources):
        return False
    if allowed_categories and (chunk.category not in allowed_categories):
        return False
    if brand_filter:
        b = (chunk.brand or "").lower()
        if brand_filter.lower() not in b:
            return False
    # (Rest of the filter logic is identical)
    if price_min is not None and (chunk.price_value is not None) and (chunk.price_value < price_min): return False
    if price_max is not None and (chunk.price_value is not None) and (chunk.price_value > price_max): return False
    if rating_min is not None and (chunk.rating_avg is not None) and (chunk.rating_avg < rating_min): return False
    return True

# (Keep _parse_query_constraints function as-is)
# ... PASTE _parse_query_constraints here ...

def bm25_search(bm25: BM25Okapi,
                chunks: List[ChunkRec],
                tokenized_corpus: List[List[str]],
                query: str,
                top_k: int,
                allowed_sources: Optional[set] = None, # NEW
                allowed_categories: Optional[set] = None,
                brand_filter: Optional[str] = None,
                price_min: Optional[float] = None,
                price_max: Optional[float] = None,
                rating_min: Optional[float] = None,
                diversify: bool = True) -> List[Tuple[ChunkRec, float]]:
    
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)

    pairs: List[Tuple[int, float]] = []
    for i, sc in enumerate(scores):
        c = chunks[i]
        # MODIFIED: Pass the new filter
        if _passes_filters(c, allowed_sources, allowed_categories, brand_filter, price_min, price_max, rating_min):
            pairs.append((i, float(sc)))

    # (Rest of the search/boost/diversify logic is identical)
    # ... PASTE the rest of the bm25_search function here ...

# --- OpenAI helpers (streaming) ---
# (Keep _ensure_client function as-is)
# ... PASTE _ensure_client here ...

def _build_messages(query: str, results: List[Tuple[ChunkRec, float]]) -> List[Dict[str, str]]:
    """Build a compact prompt with clearly numbered context blocks for citations."""
    ctx_blocks = []
    for i, (c, s) in enumerate(results, 1):
        head = f"[{i}] {c.title} — DocID: {c.doc_id}" + (f" — {c.url}" if c.url else "")
        fields = []
        if c.source: fields.append(f"Source: {c.source}") # NEW
        if c.brand: fields.append(f"Brand: {c.brand}")
        if c.category: fields.append(f"Category: {c.category}")
        if c.price_value is not None: fields.append(f"PriceValue: {int(c.price_value)}")
        if c.rating_avg is not None: fields.append(f"Rating: {c.rating_avg}/5")
        meta_line = " | ".join(fields)
        ctx_blocks.append(f"{head}\n{meta_line}\n---\n{c.text}\n")
    
    # (System prompt and message creation is identical)
    system = (
        "You are a precise product assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say you don't know. Include concise bullets. "
        "Cite as [#] with DocID and include URL when available."
    )
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# (Keep stream_answer function as-is)
# ... PASTE stream_answer here ...


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="RAG: Daraz + Startech", layout="wide")
st.title("🛍️ Combined RAG — Daraz & Startech")

# (Sidebar code is identical)
with st.sidebar:
    # ... (paste your sidebar settings code here) ...
    st.markdown("---")
    st.caption("Provide your corpus:")
    uploaded = st.file_uploader("Upload *combined* corpus (.md)", type=["md"])
    default_path = st.text_input("Or path to corpus file", value=DEFAULT_MD_PATH)
    st.markdown("---")
    st.info("To update data, run `python run_scraping.py` and then `python build_corpus.py` from your terminal.")


# (Corpus loading code is identical)
md_text = None
if uploaded is not None:
    md_text = uploaded.read().decode("utf-8", errors="ignore")
elif default_path and os.path.exists(default_path):
    with open(default_path, "r", encoding="utf-8") as f:
        md_text = f.read()

if not md_text:
    st.info("Corpus file not found (`out/combined_corpus.md`).\n\nPlease run `python run_scraping.py` and `python build_corpus.py` first.")
    st.stop()

with st.spinner("Parsing products..."):
    products = parse_products_from_md(md_text)
if not products:
    st.error("No products detected in corpus.")
    st.stop()

with st.spinner("Chunking (Chonkie) & building BM25 index..."):
    bm25, chunk_table, tokenized_corpus = build_or_load_bm25(products, lang)

# Derive facets
all_sources = sorted({p.source for p in products if p.source})
all_categories = sorted({p.category for p in products if p.category}) # Requirement 5!
all_brands = sorted({(p.brand or "").strip() for p in products if p.brand})

st.success(f"Parsed **{len(products):,}** products ({len(all_sources)} sources) → **{len(chunk_table):,}** chunks. Index ready.")

# --- Facets UI ---
st.markdown("#### Filters")
ncols = st.columns([1.5, 1.5, 1.2, 1.2, 1.0])
c1, c2, c3, c4, c5 = ncols
with c1:
    # NEW: Filter by source
    sel_sources = st.multiselect("Source", options=all_sources, default=all_sources)
with c2:
    # This is Requirement 5: "show which category available"
    sel_categories = st.multiselect("Category", options=all_categories, default=[])
with c3:
    brand_filter = st.text_input("Brand contains", "")
with c4:
    price_max_ui = st.text_input("Max price (BDT)", "")
with c5:
    rating_min_ui = st.text_input("Min rating (0-5)", "")

# (Keep _to_float function as-is)
# ... PASTE _to_float here ...

# (Chat/Query box code is identical)
st.markdown("---")
query = st.text_input("Ask about products (e.g., 'laptop under 50000' or 'best mini camera')", "")
go = st.button("Search")
# ... (expander for brands) ...

if go and query.strip():
    constraints = _parse_query_constraints(query)
    
    # Merge filters
    allowed_sources = set(sel_sources) if sel_sources else None # NEW
    allowed_categories = set(sel_categories) if sel_categories else None
    brand_q = brand_filter if brand_filter.strip() else None
    price_min = constraints["price_min"]
    price_max = price_max_filter if price_max_filter is not None else constraints["price_max"]
    rating_min = rating_min_filter if rating_min_filter is not None else constraints["rating_min"]

    with st.spinner("Retrieving with BM25..."):
        results = bm25_search(
            bm25, chunk_table, tokenized_corpus, query,
            top_k=top_k,
            allowed_sources=allowed_sources, # NEW
            allowed_categories=allowed_categories,
            brand_filter=brand_q,
            price_min=price_min,
            price_max=price_max,
            rating_min=rating_min,
            diversify=diversify,
        )

    if not results:
        st.warning("No results matched your query/filters.")
        st.stop()

    colL, colR = st.columns([0.55, 0.45], gap="large")

    with colL:
        st.subheader("Top matches")
        for i, (chunk, score) in enumerate(results, 1):
            meta_bits = []
            if chunk.source: meta_bits.append(f"**{chunk.source}**") # NEW
            if chunk.brand: meta_bits.append(f"**Brand:** {chunk.brand}")
            if chunk.category: meta_bits.append(f"**Category:** {chunk.category}")
            if chunk.price_value is not None: meta_bits.append(f"**Price:** ~৳{int(chunk.price_value)}")
            if chunk.rating_avg is not None:
                rc = f" ({chunk.rating_cnt} ratings)" if chunk.rating_cnt is not None else ""
                meta_bits.append(f"**Rating:** {chunk.rating_avg}/5{rc}")
            
            st.markdown(
                f"**[{i}] {chunk.title}** \n"
                f"DocID: `{chunk.doc_id}` • Score: `{score:.3f}`  \n"
                f"{'URL: ' + chunk.url if chunk.url else ''}  \n"
                + ("  \n".join(meta_bits) if meta_bits else "")
            )
            with st.expander("View chunk"):
                st.write(chunk.text)
    
    # (Right column for LLM answer and Download button are identical)
    with colR:
        # ... (paste your right column code here) ...
    
    # ... (paste your download button code here) ...