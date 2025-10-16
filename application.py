"""
Daraz Products RAG — Chonkie + BM25 (Streamlit App)
===================================================

What this file does (high level):
- Reads a markdown corpus you scraped/generated for Daraz products. Each product is wrapped by
  HTML-style comment markers like <!--DOC:START ...--> ... <!--DOC:END-->, containing a block of
  markdown (title, URL, brand, price, rating, and free text).
- Parses each product block into a light-weight ProductDoc object, extracting convenient metadata
  fields (brand, price_value, rating_avg, rating_cnt, etc.). The raw markdown of each doc is kept
  intact in `raw_md` for showing to the user.
- Uses **Chonkie** (RecursiveChunker) to split each product’s markdown into search-friendly chunks
  that respect markdown structure (headings, lists) where possible. If Chonkie fails (e.g. API
  changes), we fall back to a very simple chunker based on blank lines.
- Builds a **BM25** index over the cleaned chunk texts so we can do fast lexical retrieval. The
  index is cached to disk under `index/` using a content signature, so repeated runs are instant
  until the corpus changes.
- Lets the user search with natural language and basic constraints (brand, category, price caps,
  minimum rating). We parse some of those constraints from the text query (e.g. “under 2000” or
  “rating 4.5+”). UI filters in the sidebar take precedence if the same constraint is specified in
  multiple places.
- Streams a compact LLM answer (using OpenAI Chat Completions) that is strictly grounded on the
  retrieved chunks. The system prompt tells the model to cite as [#] with DocID and include URLs
  when available.

Notes:
- This is a minimal, dependency-light RAG prototype for product search. BM25 gives solid lexical
  recall, while Chonkie helps avoid mid-sentence splitting.
- You can swap the LLM to anything Chat Completions-compatible; the grounding is driven by the
  retrieved context blocks built in `_build_messages`.
- The code tries to fail safely: if the chunker or regexes change upstream, you still get usable
  behavior.
"""

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


# ----------------------------
# App Config
# ----------------------------
# A small set of knobs you’re likely to change while experimenting.
DEFAULT_MD_PATH = "corpuses/daraz_products_corpus.md"   # where we look by default if no file is uploaded
INDEX_DIR = "index"                                 # local cache folder for BM25 + metadata
os.makedirs(INDEX_DIR, exist_ok=True)

DEFAULT_MODEL = "gpt-4o-mini"                       # cheap + reasonably capable
DEFAULT_TOPK = 8                                     # how many chunks to retrieve for LLM
DEFAULT_LANG = "en"                                  # chunking recipe language (Chonkie)

# ----------------------------
# Data structures
# ----------------------------
# We keep the parsed product at two granularity levels: the original doc (ProductDoc) and the
# derived chunk (ChunkRec). Having both makes it easy to show pretty metadata in the UI, and also
# perform metadata-aware filtering during retrieval.

@dataclass
class ProductDoc:
    """A single product as parsed from the markdown corpus.

    Attributes:
        doc_id: Stable identifier. Comes from the DOC header if present; otherwise we auto-generate.
        title: Human-friendly title, grabbed from an H2 (##) line if available.
        url: The canonical product URL (if present). Useful for clicks and citations.
        category: Optional category label from DOC header attrs.
        brand: Parsed from a "**Brand:** ..." line in the body, if present.
        price_value: A numeric approximation of the price (lowest if a range). Used for filtering.
        rating_avg: Average star rating as a float, when present.
        rating_cnt: Count of ratings (int), when present.
        raw_md: The exact markdown body inside DOC markers — we keep this for display and chunking.
    """
    doc_id: str
    title: str
    url: Optional[str]
    category: Optional[str]
    brand: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    raw_md: str  # the body inside DOC markers (for display)


@dataclass
class ChunkRec:
    """A single search unit that BM25 scores. A product can yield multiple chunks.

    We copy over convenient metadata so we can:
      - Show nice cards in the UI (brand/category/price/rating)
      - Apply filters BEFORE scoring/diversification (saves time and avoids irrelevant hits)
    """
    doc_id: str
    title: str
    url: Optional[str]
    category: Optional[str]
    brand: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    text: str


# ----------------------------
# Regex helpers for parsing corpus
# ----------------------------
# We’re fairly liberal in the regexes to accommodate variations in scraped/hand-made markdown.
# If you change your corpus format later, tweak these patterns.
DOC_BLOCK_RE = re.compile(r"<!--DOC:START(?P<attrs>[^>]*)-->(?P<body>.*?)<!--DOC:END-->", re.DOTALL|re.IGNORECASE)
TITLE_RE = re.compile(r"^##\s+(.+?)\s*(?:\s{2,}\n|\n|$)", re.MULTILINE)
URL_LINE_RE = re.compile(r"\*\*URL:\*\*\s*(\S+)", re.IGNORECASE)
BRAND_RE = re.compile(r"\*\*Brand:\*\*\s*(.+)", re.IGNORECASE)
PRICE_RE = re.compile(r"\*\*Price:\*\*\s*(.+)", re.IGNORECASE)
RATING_RE = re.compile(r"\*\*Rating:\*\*\s*([0-9.]+)\s*/\s*5(?:\s*\((\d+)\s*ratings\))?", re.IGNORECASE)

def _meta_from_header(attrs: str) -> Dict[str,str]:
    """Parses the attribute blob that appears right after DOC:START.

    Example: <!--DOC:START id=123 category=HomeTextile -->
    We split on spaces and then split each key=value. No quotes supported here on purpose — keep it
    simple and predictable in the corpus.
    """
    out = {}
    for kv in attrs.strip().split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_price_value(s: str) -> Optional[float]:
    """
    Extract a numeric price from a display string.

    Why this exists: scraped prices often include currency symbols, grouping commas, or even ranges.
    We try to be forgiving:
      - "৳ 1,999" → 1999.0
      - "1,999 BDT" → 1999.0
      - "৳1,500 - ৳2,100" → take the **minimum** (1500.0) so "under X" filters work as expected.

    If we can’t find any numbers, return None.
    """
    s = s.replace(",", "")
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums:
        return None
    try:
        vals = [float(x) for x in nums]
        return min(vals) if vals else None
    except Exception:
        # If something weird sneaks through (locale, non-ASCII digits), fail soft.
        return None


def parse_products_from_md(md_text: str) -> List[ProductDoc]:
    """Pulls each product between DOC markers and extracts light metadata.

    This is the single place that converts your corpus into structured records. Any improvements you
    make here (e.g., richer metadata) will immediately improve filtering and display downstream.
    """
    products: List[ProductDoc] = []
    for m in DOC_BLOCK_RE.finditer(md_text):
        attrs = m.group("attrs") or ""
        body = (m.group("body") or "").strip()

        # Parse key-value attributes from the DOC header
        meta = _meta_from_header(attrs)
        doc_id = meta.get("id") or "doc_" + str(len(products)+1)  # stable-but-simple fallback
        category = meta.get("category")

        # Prefer an H2 title (## Some Product Title)
        title_m = TITLE_RE.search(body)
        title = (title_m.group(1).strip() if title_m else f"Product {doc_id}")

        # URL line looks like: **URL:** https://...
        url_m = URL_LINE_RE.search(body)
        url = url_m.group(1).strip() if url_m else None

        # Optional fields in the body. We keep failures non-fatal.
        brand = None
        brand_m = BRAND_RE.search(body)
        if brand_m:
            brand = brand_m.group(1).strip()

        price_value = None
        price_m = PRICE_RE.search(body)
        if price_m:
            price_value = _parse_price_value(price_m.group(1))

        rating_avg, rating_cnt = None, None
        rating_m = RATING_RE.search(body)
        if rating_m:
            try:
                rating_avg = float(rating_m.group(1))
            except Exception:
                rating_avg = None
            try:
                rating_cnt = int(rating_m.group(2)) if rating_m.group(2) else None
            except Exception:
                rating_cnt = None

        products.append(ProductDoc(
            doc_id=doc_id, title=title, url=url,
            category=category, brand=brand, price_value=price_value,
            rating_avg=rating_avg, rating_cnt=rating_cnt, raw_md=body
        ))
    return products


# ----------------------------
# Chunking (Chonkie)
# ----------------------------
# We wrap the chunker creation so you can easily swap recipes or languages from the UI.

def build_chunker(lang: str = DEFAULT_LANG) -> RecursiveChunker:
    """Construct a markdown-aware chunker.

    The "markdown" recipe in Chonkie tries to split at natural boundaries such as headings and
    lists. This tends to produce chunks that preserve semantic coherence (which helps both BM25 and
    downstream LLMs).
    """
    return RecursiveChunker.from_recipe("markdown", lang=lang)


def product_to_chunks(product: ProductDoc, chunker: RecursiveChunker) -> List[ChunkRec]:
    """Turn a single ProductDoc into multiple ChunkRec entries.

    - We prefer Chonkie’s output. If it errors (API drift, missing recipe), we fall back to a very
      coarse blank-line splitter so the app keeps working.
    - We also lightly clean each chunk for BM25 by dropping bare URLs and image markers. The raw
      markdown is still available for display in the left panel.
    """
    chunks = []
    try:
        chonks = chunker(product.raw_md)
    except Exception:
        # Fail-safe: if Chonkie recipe signature changes, fall back to coarse split on blank lines.
        split = [s.strip() for s in re.split(r"\n{2,}", product.raw_md) if s.strip()]
        chonks = [{"text": s} for s in split]

    for c in chonks:
        # Chonkie returns objects with a .text. Our fallback returns dicts with a "text" key.
        text = (getattr(c, "text", None) or (c["text"] if isinstance(c, dict) else "")).strip()
        if not text:
            continue

        # Remove long/bare URL-only lines from the **indexed** text.
        # We keep raw_md untouched so the left panel can show the original content.
        indexed_text = _clean_for_bm25(text)
        if not indexed_text:
            continue

        chunks.append(ChunkRec(
            doc_id=product.doc_id, title=product.title, url=product.url,
            category=product.category, brand=product.brand,
            price_value=product.price_value, rating_avg=product.rating_avg, rating_cnt=product.rating_cnt,
            text=indexed_text
        ))
    return chunks


# ----------------------------
# BM25 indexing
# ----------------------------
# We keep the pre-processing intentionally simple. For product text, heavy NLP often hurts more than
# it helps, and BM25 is robust to small variations.
STOPWORDS = set([
    "the","a","an","and","or","of","for","on","in","to","from","with","by","at","is","are","was","were",
    "this","that","these","those","it","its","as","be","can","will","has","have"
])


def _clean_for_bm25(text: str) -> str:
    """Strip noisy lines from a chunk before indexing.

    Rationale:
    - Keeping bare URLs hurts lexical matching (tons of tokens with little meaning).
    - We also skip image sections (often start with "**Images" in our corpus).
    - Everything else is left intact to keep recall high.
    """
    clean_lines = []
    for line in text.splitlines():
        ll = line.strip()
        if not ll:
            continue
        if ll.lower().startswith("**images"):
            continue
        if "http://" in ll or "https://" in ll:
            # Keep only the human text before URLs, drop the URL parts themselves.
            pieces = re.split(r"\s+https?://\S+", ll)
            ll = " ".join([p for p in pieces if p.strip()])
            if not ll:
                continue
        clean_lines.append(ll)
    return "\n".join(clean_lines)


def _tokenize(text: str) -> List[str]:
    """A minimal tokenizer for BM25: lowercase, keep alphanumerics/underscores, drop stopwords."""
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in toks if t not in STOPWORDS]


def _sha1(s: str) -> str:
    """Tiny helper for stable content signatures (used in caching)."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _index_paths(sig: str) -> Tuple[str, str]:
    """Given a signature, return the two cache file paths (bm25.pkl, meta.pkl)."""
    return (
        os.path.join(INDEX_DIR, f"bm25_{sig}.pkl"),
        os.path.join(INDEX_DIR, f"meta_{sig}.pkl"),
    )


def build_or_load_bm25(products: List[ProductDoc], lang: str) -> Tuple[BM25Okapi, List[ChunkRec], List[List[str]]]:
    """Create (or load) a BM25 index for the current corpus.

    Steps:
      1) Chunk every product (markdown-aware)
      2) Create a content signature (so we can reuse indexes across runs when nothing changed)
      3) If cached index exists → load & return
      4) Otherwise, tokenize and train a fresh BM25, then persist it to disk

    Returns:
      - bm25: the trained BM25Okapi instance
      - chunks: the flattened list of ChunkRec (order matches the BM25 corpus order)
      - tokenized_corpus: the pre-tokenized version of every chunk.text (BM25 expects this)
    """
    # Build chunks
    chunker = build_chunker(lang=lang)
    all_chunks: List[ChunkRec] = []
    for p in products:
        all_chunks.extend(product_to_chunks(p, chunker))

    # Signature for disk caching: content hash + lang + cleaning version.
    # If you change tokenization or cleaning logic in the future, bump the leading version string.
    content_sig = _sha1("\n".join([c.doc_id + "\t" + c.text for c in all_chunks]))
    sig = _sha1(f"v1|lang={lang}|{content_sig}")
    bm25_pkl, meta_pkl = _index_paths(sig)

    if os.path.exists(bm25_pkl) and os.path.exists(meta_pkl):
        # Fast path: reuse a previous build. Great for iterative UI tweaks.
        with open(bm25_pkl, "rb") as f:
            bm25 = pickle.load(f)
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        tokenized_corpus = meta["tokenized_corpus"]
        chunks = meta["chunks"]
        return bm25, chunks, tokenized_corpus

    # Build BM25 fresh
    tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Persist for next run
    with open(bm25_pkl, "wb") as f:
        pickle.dump(bm25, f)
    with open(meta_pkl, "wb") as f:
        pickle.dump({"tokenized_corpus": tokenized_corpus, "chunks": all_chunks}, f)

    return bm25, all_chunks, tokenized_corpus


def _passes_filters(chunk: ChunkRec,
                    allowed_categories: Optional[set],
                    brand_filter: Optional[str],
                    price_min: Optional[float],
                    price_max: Optional[float],
                    rating_min: Optional[float]) -> bool:
    """Apply facet filters to a chunk BEFORE we bother with extra work.

    Each filter is soft/optional: if None/empty, it’s ignored.
    We also treat missing metadata (e.g., price_value=None) as *unknown* rather than auto-fail.
    This way, a product without price won’t be excluded by "under 2000" unless it has a price that
    violates the bound. That keeps recall decent in sparse datasets.
    """
    if allowed_categories and (chunk.category not in allowed_categories):
        return False
    if brand_filter:
        b = (chunk.brand or "").lower()
        if brand_filter.lower() not in b:
            return False
    if price_min is not None and (chunk.price_value is not None) and (chunk.price_value < price_min):
        return False
    if price_max is not None and (chunk.price_value is not None) and (chunk.price_value > price_max):
        return False
    if rating_min is not None and (chunk.rating_avg is not None) and (chunk.rating_avg < rating_min):
        return False
    return True


def _parse_query_constraints(q: str) -> Dict[str, Optional[float]]:
    """Extract lightweight price/rating constraints from a natural language query.

    Supported patterns (case-insensitive; commas forgiven):
      - "under 2000", "below 2k", "<= 1500", "less than 2500" → sets price_max
      - "between 1500 and 3000" → sets price_min and price_max
      - "rating >= 4.5", "4.5+ rating", "at least 4 stars" → sets rating_min

    We purposefully **don’t** try to be perfect; it’s better to capture 80% of common phrasing than
    to mis-parse rare patterns. UI filters are always available and take precedence.
    """
    qn = q.lower().replace(",", "")
    price_min = None
    price_max = None
    rating_min = None

    # between X and Y
    m = re.search(r"between\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)", qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        price_min, price_max = (min(a,b), max(a,b))

    # under/below/<=
    m = re.search(r"(?:under|below|<=|less than)\s*(\d+(?:\.\d+)?)", qn)
    if m:
        price_max = float(m.group(1))

    # >= (supports optional currency tokens after the number)
    m = re.search(r"(?:>=|at least)\s*(\d+(?:\.\d+)?)\s*(?:bdt|৳|tk|taka)?", qn)
    if m:
        price_min = max(price_min or 0.0, float(m.group(1)))

    # rating patterns — a few variants we see a lot
    m = re.search(r"rating\s*(?:>=|at least|of at least)?\s*([0-5](?:\.\d+)?)", qn)
    if m:
        rating_min = float(m.group(1))
    else:
        m = re.search(r"([0-5](?:\.\d+)?)\s*\+\s*rating", qn)
        if m:
            rating_min = float(m.group(1))
        else:
            m = re.search(r"(?:at least|minimum|min)\s*([0-5](?:\.\d+)?)\s*(?:stars|rating)", qn)
            if m:
                rating_min = float(m.group(1))

    return {"price_min": price_min, "price_max": price_max, "rating_min": rating_min}


def bm25_search(bm25: BM25Okapi,
                chunks: List[ChunkRec],
                tokenized_corpus: List[List[str]],
                query: str,
                top_k: int,
                allowed_categories: Optional[set] = None,
                brand_filter: Optional[str] = None,
                price_min: Optional[float] = None,
                price_max: Optional[float] = None,
                rating_min: Optional[float] = None,
                diversify: bool = True) -> List[Tuple[ChunkRec, float]]:
    """Score all chunks with BM25, apply filters, and optionally diversify across products.

    Diversification rationale: When a product has multiple strong chunks, the top-k can get flooded
    by the same product. We first take the best chunk per unique product (doc_id), then if we still
    need more to reach top_k, we allow repeats. This usually makes the left panel feel more useful.
    """
    # 1) Score against the tokenized query
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)

    # 2) Pair (index, score) and filter out chunks that violate user constraints
    pairs: List[Tuple[int, float]] = []
    for i, sc in enumerate(scores):
        c = chunks[i]
        if _passes_filters(c, allowed_categories, brand_filter, price_min, price_max, rating_min):
            pairs.append((i, float(sc)))

    # 3) Light field-aware boosting: if the query words appear in the title or brand, give a small nudge
    q_words = set(q_tokens)
    def _boost(idx: int, s: float) -> float:
        c = chunks[idx]
        boost = 0.0
        title_words = set(_tokenize(c.title))
        brand_words = set(_tokenize(c.brand or ""))
        if q_words & title_words:
            boost += 0.10 * s  # 10% bump for title matches
        if q_words & brand_words:
            boost += 0.05 * s  # 5% bump for brand matches
        return s + boost

    pairs = [(i, _boost(i, s)) for (i, s) in pairs]
    pairs.sort(key=lambda x: x[1], reverse=True)

    if not diversify:
        # Classic top-k without diversification
        return [(chunks[i], s) for i, s in pairs[:top_k]]

    # 4) Diversify: pick top-scoring unique products first
    seen_docs = set()
    diversified: List[Tuple[ChunkRec, float]] = []

    # First pass: 1 per product
    for i, s in pairs:
        c = chunks[i]
        if c.doc_id in seen_docs:
            continue
        diversified.append((c, s))
        seen_docs.add(c.doc_id)
        if len(diversified) >= top_k:
            return diversified

    # Second pass (if we still need slots): allow repeats from already-selected products
    if len(diversified) < top_k:
        for i, s in pairs:
            c = chunks[i]
            diversified.append((c, s))
            if len(diversified) >= top_k:
                break
    return diversified


# ----------------------------
# OpenAI helpers (streaming)
# ----------------------------
# These helpers keep all OpenAI interaction in one place, so the rest of the app stays framework-
# agnostic. If you want to swap providers, duplicate the interface here.

def _ensure_client() -> OpenAI:
    """Create an OpenAI client, loudly failing if the API key isn’t set.

    This avoids a confusing "authentication error during stream" later on.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI()


def _build_messages(query: str, results: List[Tuple[ChunkRec, float]]) -> List[Dict[str, str]]:
    """Build a compact prompt with clearly numbered context blocks for citations.

    We format every chunk as:
      [#] Title — DocID: xyz — optional URL
      Brand | Category | PriceValue | Rating
      ---
      chunk text...

    The system prompt instructs the model to:
      - answer **only** from context (don’t hallucinate)
      - be concise with bullets
      - cite like [#] with DocID and include URLs when available
    """
    ctx_blocks = []
    for i, (c, s) in enumerate(results, 1):
        head = f"[{i}] {c.title} — DocID: {c.doc_id}" + (f" — {c.url}" if c.url else "")
        fields = []
        if c.brand: fields.append(f"Brand: {c.brand}")
        if c.category: fields.append(f"Category: {c.category}")
        if c.price_value is not None: fields.append(f"PriceValue: {int(c.price_value)}")
        if c.rating_avg is not None: fields.append(f"Rating: {c.rating_avg}/5")
        meta_line = " | ".join(fields)
        ctx_blocks.append(f"{head}\n{meta_line}\n---\n{c.text}\n")
    system = (
        "You are a precise product assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say you don't know. Include concise bullets. "
        "Cite as [#] with DocID and include URL when available."
    )
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def stream_answer(model: str, messages: List[Dict[str, str]], temperature: float = 0.2):
    """Stream tokens to Streamlit as they arrive, so the UI feels responsive.

    If anything goes wrong (rate limits, bad key), the caller catches and surfaces the error.
    """
    client = _ensure_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


# ----------------------------
# Streamlit UI
# ----------------------------
# The UI is split into: sidebar (settings & corpus input), main area (stats, filters, search box),
# and two-column result layout (left: matches; right: LLM answer).

st.set_page_config(page_title="RAG: Recursive Chunking + BM25 (Daraz)", layout="wide")
st.title("Daraz Products RAG — Recursive Chunking + BM25")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    # Keep model list short and cheap by default. You can add bigger models if you like.
    model = st.selectbox("OpenAI model (cheap)", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", ], index=0)
    # Chonkie supports multiple languages; add more if your corpus isn’t English.
    lang = st.selectbox("Chunk recipe language", ["en",], index=0)
    top_k = st.slider("Top-K chunks", 1, 20, DEFAULT_TOPK)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    diversify = st.checkbox("Diversify: limit one chunk per product first", value=True)

    st.markdown("---")
    st.caption("Provide your corpus:")
    uploaded = st.file_uploader("Upload Markdown corpus (.md)", type=["md"], accept_multiple_files=False)
    default_path = st.text_input("Or path to Markdown file", value=DEFAULT_MD_PATH)

# Load corpus text: prefer the uploaded file; otherwise use the default path if it exists.
md_text = None
if uploaded is not None:
    md_text = uploaded.read().decode("utf-8", errors="ignore")
elif default_path and os.path.exists(default_path):
    with open(default_path, "r", encoding="utf-8") as f:
        md_text = f.read()


if not md_text:
    # Early exit to keep the UI clean if no data is available.
    st.info("Upload your `daraz_products_corpus.md` or set its path in the sidebar.")
    st.stop()

# Parse products
with st.spinner("Parsing products…"):
    products = parse_products_from_md(md_text)

if not products:
    st.error("No products detected. Ensure the file contains <!--DOC:START ...--> and <!--DOC:END--> blocks.")
    st.stop()

# Build or load BM25
with st.spinner("Chunking (Chonkie) & building BM25 index…"):
    bm25, chunk_table, tokenized_corpus = build_or_load_bm25(products, lang=lang)

# Derive facet values for UI dropdowns. We guard against Nones.
all_categories = sorted({p.category for p in products if p.category})
all_brands = sorted({(p.brand or "").strip() for p in products if p.brand})

st.success(f"Parsed **{len(products):,}** products → **{len(chunk_table):,}** chunks. BM25 index ready.")

# Facets UI
st.markdown("#### Filters")
# Column widths tuned to fit common inputs nicely; adjust to taste.
ncols = st.columns([1.7, 1.3, 1.2, 1.2])
c1, c2, c3, c4 = ncols
with c1:
    sel_categories = st.multiselect("Category", options=all_categories, default=[])
with c2:
    brand_filter = st.text_input("Brand contains", "")
with c3:
    price_max_ui = st.text_input("Max price (BDT)", "")
with c4:
    rating_min_ui = st.text_input("Min rating (0-5)", "")


def _to_float(x: str) -> Optional[float]:
    """Parse a numeric text field safely.

    We accept simple "1234" or "1234.56". If the field is empty or malformed, return None so the
    caller can treat it as "no filter" rather than erroring. Commas are removed for convenience.
    """
    x = x.strip().replace(",", "")
    if not x:
        return None
    m = re.match(r"^\d+(?:\.\d+)?$", x)
    return float(x) if m else None


# Convert UI inputs to typed filters
price_max_filter = _to_float(price_max_ui)
rating_min_filter = _to_float(rating_min_ui)

# Chat / Query box
st.markdown("---")
query = st.text_input("Ask about products (e.g., 'best bedding set under 2000 with rating 4.5+')", "")
go = st.button("Search")

# Small helper: show a quick glance of parsed brands so users know what to type
with st.expander("Show some brands detected", expanded=False):
    st.write(", ".join(all_brands[:60]) + (" ..." if len(all_brands) > 60 else ""))

if go and query.strip():
    # 1) Parse constraints from the natural language query
    constraints = _parse_query_constraints(query)

    # 2) Merge with UI filters (UI takes precedence if provided)
    allowed_categories = set(sel_categories) if sel_categories else None
    brand_q = brand_filter if brand_filter.strip() else None
    price_min = constraints["price_min"]
    price_max = price_max_filter if price_max_filter is not None else constraints["price_max"]
    rating_min = rating_min_filter if rating_min_filter is not None else constraints["rating_min"]

    # 3) Retrieve with BM25
    with st.spinner("Retrieving with BM25…"):
        results = bm25_search(
            bm25, chunk_table, tokenized_corpus, query,
            top_k=top_k,
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

    # 4) Two-column layout: left = matches; right = streamed answer
    colL, colR = st.columns([0.55, 0.45], gap="large")

    with colL:
        st.subheader("Top matches")
        for i, (chunk, score) in enumerate(results, 1):
            # Build a compact metadata line. We avoid None checks in the UI by constructing a list
            # and joining only what’s present.
            meta_bits = []
            if chunk.brand: meta_bits.append(f"**Brand:** {chunk.brand}")
            if chunk.category: meta_bits.append(f"**Category:** {chunk.category}")
            if chunk.price_value is not None: meta_bits.append(f"**Price:** ~৳{int(chunk.price_value)}")
            if chunk.rating_avg is not None:
                rc = f" ({chunk.rating_cnt} ratings)" if chunk.rating_cnt is not None else ""
                meta_bits.append(f"**Rating:** {chunk.rating_avg}/5{rc}")

            # We purposely render score (useful for debugging and for power users)
            st.markdown(
                f"**[{i}] {chunk.title}**  \n"
                f"DocID: `{chunk.doc_id}` • Score: `{score:.3f}`  \n"
                f"{'URL: ' + chunk.url if chunk.url else ''}  \n"
                + ("  \n".join(meta_bits) if meta_bits else "")
            )
            with st.expander("View chunk"):
                st.write(chunk.text)

    with colR:
        st.subheader("Answer")
        messages = _build_messages(query, results)
        try:
            st.write_stream(stream_answer(model, messages, temperature=temperature))
        except Exception as e:
            # Most errors here are authentication or rate limit related. Surface the raw message to
            # keep troubleshooting simple.
            st.error(f"OpenAI error: {e}")

    # 5) Download button: let users export the matched items as JSON for analysis/auditing.
    export_rows = []
    for i, (c, s) in enumerate(results, 1):
        export_rows.append({
            "rank": i,
            "score": s,
            "doc_id": c.doc_id,
            "title": c.title,
            "url": c.url or "",
            "category": c.category or "",
            "brand": c.brand or "",
            "price_value": c.price_value if c.price_value is not None else "",
            "rating_avg": c.rating_avg if c.rating_avg is not None else "",
            "rating_cnt": c.rating_cnt if c.rating_cnt is not None else "",
            "chunk_text": c.text[:2000],  # keep the export lightweight
        })
    export_bytes = io.BytesIO()
    export_bytes.write(json.dumps(export_rows, ensure_ascii=False, indent=2).encode("utf-8"))
    export_bytes.seek(0)
    st.download_button("Download results (JSON)", data=export_bytes, file_name="results.json", mime="application/json")
