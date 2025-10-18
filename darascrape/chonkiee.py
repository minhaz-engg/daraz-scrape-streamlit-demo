#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG over Daraz product dumps (products.json per category).

- Uses Chonkie (RecursiveChunker) to chunk robust product "documents".
- Uses OpenAI Embeddings for retrieval and OpenAI Chat for answering.
- Flat, dependency-light, fast; no vector DB (NumPy cosine + small cache).
- Just set USER_QUERY and run:  python rag_products.py

Folder layout expected:
result/
  ├── www_daraz_com_bd_kitchen_fixtures/
  │     └── products.json
  ├── www_daraz_com_bd_shop_bedding_sets/
  │     └── products.json
  └── ... (many categories)

Notes:
- JSON can vary a lot; normalization is defensive.
- Only products.json is read in each category folder.

Author: you + hippo power 🦛
"""

import os, json, glob, re, hashlib, pickle, math, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
import numpy as np

# ---- OpenAI SDK (v1) ----
from openai import OpenAI

# ---- Chonkie (chunking) ----
from chonkie import RecursiveChunker

# ---- Optional: token-accurate chunk sizing; falls back to char-count ----
try:
    import tiktoken
    _enc = None
    for name in ("o200k_base", "cl100k_base"):
        try:
            _enc = tiktoken.get_encoding(name)
            break
        except Exception:
            continue
    if _enc is None:
        raise RuntimeError("No tiktoken encoding found.")

    def count_tokens(s: str) -> int:
        return len(_enc.encode(s))
except Exception:
    def count_tokens(s: str) -> int:
        # rough heuristic if tiktoken not available
        return max(1, len(s) // 4)


# =================== CONFIG ===================

ROOT_DIR = "result"  # your scraped data root

# Embedding model (small = cheap/fast; change if you want)
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Chat model for the final answer (change via env if you prefer)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")

# Chunking: target token length and minimum chars
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "320"))
MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "24"))

# Retrieval
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "12"))      # retrieve top-k chunks
TOP_PRODUCTS = int(os.getenv("TOP_PRODUCTS", "6"))       # cap unique products in final context

# Embedding batching + cache
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", ".emb_cache.pkl")

# Put your question here (or set via env)
USER_QUERY = os.getenv("RAG_QUERY", "show me some traditional laptop")

# ==============================================


@dataclass
class ProductDoc:
    doc_id: str
    category: str
    title: Optional[str]
    url: Optional[str]
    price_value: Optional[float]
    price_display: Optional[str]
    rating_avg: Optional[float]
    rating_count: Optional[int]
    sold_count: Optional[int]
    brand: Optional[str]
    seller_name: Optional[str]
    colors: Optional[List[str]]
    description: Optional[str]
    source_path: str
    raw: Dict[str, Any]


def _first(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v not in (None, "", [], {}):
            return v
    return None


def _to_https(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("http"):
        return u
    if u.startswith("/"):
        # site-relative; assume daraz main host
        return "https://www.daraz.com.bd" + u
    # already looks like full path? try to patch
    if u.startswith("www."):
        return "https://" + u
    return u


_number_re = re.compile(r"(\d[\d,\.]*)")
def _parse_number(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = _number_re.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None

def _parse_int(s: Optional[str]) -> Optional[int]:
    v = _parse_number(s)
    return int(v) if v is not None else None

def _parse_sold(location_val: Optional[str]) -> Optional[int]:
    if not location_val:
        return None
    m = re.search(r"(\d[\d,]*)\s*sold", location_val, flags=re.I)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            return None
    return _parse_int(location_val)  # fallback

def _get_nested(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def normalize_product(prod: Dict[str, Any], category: str, source_path: str) -> ProductDoc:
    # Robust ID (prefer clear item ID; fallback to hash)
    raw_id = _first(prod.get("data_item_id"), prod.get("data_sku_simple"))
    if not raw_id:
        # stable hash from title+url
        base = (_first(prod.get("product_title"), _get_nested(prod, ["detail", "name"], "")) or "") \
               + "|" + (_first(prod.get("product_detail_url"), prod.get("detail_url"), "") or "")
        raw_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

    title = _first(prod.get("product_title"), _get_nested(prod, ["detail", "name"]))
    url = _to_https(_first(prod.get("detail_url"), prod.get("product_detail_url"), _get_nested(prod, ["detail", "url"])))
    brand = _first(_get_nested(prod, ["detail", "brand"]), _get_nested(prod, ["brand"]))

    # Price (value + display)
    price_display = _first(_get_nested(prod, ["detail", "price", "display"]), prod.get("product_price"))
    price_value = _first(_get_nested(prod, ["detail", "price", "value"]), _parse_number(price_display))

    rating_avg = _get_nested(prod, ["detail", "rating", "average"])
    rating_count = _get_nested(prod, ["detail", "rating", "count"])
    if isinstance(rating_count, str):
        rating_count = _parse_int(rating_count)

    sold_count = _parse_sold(prod.get("location"))
    seller_name = _first(_get_nested(prod, ["detail", "seller", "name"]))
    colors = _get_nested(prod, ["detail", "colors"], [])
    if not isinstance(colors, list):
        colors = [str(colors)] if colors is not None else []

    # Description/highlights/specs
    desc = _first(
        _get_nested(prod, ["detail", "details", "description_text"]),
        _get_nested(prod, ["detail", "details", "raw_text"]),
    )

    return ProductDoc(
        doc_id=str(raw_id),
        category=category,
        title=str(title) if title else None,
        url=url,
        price_value=float(price_value) if price_value is not None else None,
        price_display=str(price_display) if price_display else None,
        rating_avg=float(rating_avg) if rating_avg is not None else None,
        rating_count=int(rating_count) if rating_count is not None else None,
        sold_count=int(sold_count) if sold_count is not None else None,
        brand=str(brand) if brand else None,
        seller_name=str(seller_name) if seller_name else None,
        colors=[str(c) for c in colors] if colors else [],
        description=str(desc) if desc else None,
        source_path=source_path,
        raw=prod,
    )

def product_text(doc: ProductDoc) -> str:
    # Build a terse, retrieval-friendly representation (no images)
    parts = []
    parts.append(f"PRODUCT_ID: {doc.doc_id}")
    parts.append(f"CATEGORY: {doc.category}")
    if doc.title: parts.append(f"TITLE: {doc.title}")
    if doc.brand: parts.append(f"BRAND: {doc.brand}")
    if doc.price_display: parts.append(f"PRICE: {doc.price_display}")
    elif doc.price_value is not None: parts.append(f"PRICE_VALUE: {doc.price_value}")
    if doc.rating_avg is not None: parts.append(f"RATING_AVG: {doc.rating_avg:.2f}")
    if doc.rating_count is not None: parts.append(f"RATING_COUNT: {doc.rating_count}")
    if doc.sold_count is not None: parts.append(f"SOLD: {doc.sold_count}")
    if doc.seller_name: parts.append(f"SELLER: {doc.seller_name}")
    if doc.colors: parts.append("COLORS: " + ", ".join(doc.colors))
    if doc.description: parts.append(f"DESCRIPTION: {doc.description}")
    if doc.url: parts.append(f"URL: {doc.url}")
    return "\n".join(parts)


def iter_product_docs(root_dir: str) -> List[ProductDoc]:
    docs: List[ProductDoc] = []
    for products_json in glob.glob(os.path.join(root_dir, "*", "products.json")):
        category = os.path.basename(os.path.dirname(products_json))
        try:
            with open(products_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
        except Exception as e:
            print(f"[WARN] Skipping {products_json}: {e}")
            continue

        for prod in data:
            if not isinstance(prod, dict):
                continue
            doc = normalize_product(prod, category=category, source_path=products_json)
            docs.append(doc)
    return docs


# --------- Chunking with Chonkie (fast & simple) ---------
def chunk_docs(docs: List[ProductDoc]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns:
      chunks_texts: List[str]  (to embed)
      chunks_meta:  List[dict] (metadata per chunk with backrefs)
    """
    chunker = RecursiveChunker(
        tokenizer_or_token_counter=count_tokens,
        chunk_size=CHUNK_SIZE_TOKENS,
        # default rules are fine; min chars prevents tiny slivers
        min_characters_per_chunk=MIN_CHARS_PER_CHUNK,
    )

    chunks_texts: List[str] = []
    chunks_meta: List[Dict[str, Any]] = []

    for doc in docs:
        text = product_text(doc)
        chunks = chunker(text)  # callable interface
        for i, ch in enumerate(chunks):
            clean = ch.text.replace("\u0000", " ").strip()
            if not clean:
                continue
            chunks_texts.append(clean)
            chunks_meta.append({
                "doc_id": doc.doc_id,
                "chunk_idx": i,
                "category": doc.category,
                "title": doc.title,
                "url": doc.url,
                "price_display": doc.price_display,
                "price_value": doc.price_value,
                "rating_avg": doc.rating_avg,
                "rating_count": doc.rating_count,
                "sold_count": doc.sold_count,
                "seller_name": doc.seller_name,
                "source_path": doc.source_path,
                "token_count": getattr(ch, "token_count", None),
            })
    return chunks_texts, chunks_meta


# --------- Embedding (OpenAI) + tiny persistent cache ---------
class EmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        try:
            with open(self.path, "rb") as f:
                self._store = pickle.load(f)
        except Exception:
            self._store = {}  # sha1 -> list[float]

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, vec: List[float]):
        self._store[key] = vec

    def save(self):
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self._store, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[WARN] Could not save cache: {e}")

def _sha1_for_text_model(text: str, model: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()

def embed_texts(client: OpenAI, texts: List[str], model: str, batch_size: int, cache: Optional[EmbeddingCache] = None) -> np.ndarray:
    """
    Returns normalized embeddings (L2 unit vectors) as ndarray of shape (N, D)
    """
    all_vecs: List[List[float]] = []
    to_embed_idx: List[int] = []
    keys: List[Optional[str]] = []

    # Cache lookup
    if cache:
        for i, t in enumerate(texts):
            key = _sha1_for_text_model(t.replace("\n", " "), model)
            keys.append(key)
            vec = cache.get(key)
            if vec is None:
                to_embed_idx.append(i)
                all_vecs.append(None)  # placeholder
            else:
                all_vecs.append(vec)
    else:
        keys = [None] * len(texts)
        to_embed_idx = list(range(len(texts)))
        all_vecs = [None] * len(texts)

    # Batch embed missing
    for start in range(0, len(to_embed_idx), batch_size):
        batch_ids = to_embed_idx[start:start + batch_size]
        batch = [texts[i].replace("\n", " ") for i in batch_ids]
        resp = client.embeddings.create(model=model, input=batch, encoding_format="float")
        for local_j, i in enumerate(batch_ids):
            vec = resp.data[local_j].embedding
            all_vecs[i] = vec
            if cache and keys[i]:
                cache.set(keys[i], vec)

    if cache:
        cache.save()

    # Convert to numpy and L2-normalize
    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


# --------- Retrieval + prompt assembly ---------
def cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> List[int]:
    # mat: (N, D), query: (D,)
    scores = mat @ query_vec  # cosine since both sides are normalized
    # Argpartition for speed; then sort the small slice
    if k >= len(scores):
        idx = np.argsort(-scores)
        return idx.tolist()
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx.tolist()

def build_context(retrieved_idx: List[int], chunks_texts: List[str], chunks_meta: List[Dict[str, Any]], limit_products: int) -> Tuple[str, List[Dict[str, Any]]]:
    seen = set()
    lines = []
    included_meta: List[Dict[str, Any]] = []
    for j in retrieved_idx:
        meta = chunks_meta[j]
        doc_id = meta["doc_id"]
        if doc_id in seen:
            # allow multiple chunks per product, but cap unique products
            pass
        else:
            if len(seen) >= limit_products:
                break
            seen.add(doc_id)
        # Add a compact header for each chunk so the model can ground answers
        header = [
            f"[DOC {doc_id} | {meta.get('title') or 'Untitled'}]",
            f"Category: {meta.get('category')}",
            f"URL: {meta.get('url') or 'N/A'}",
            f"Price: {meta.get('price_display') or meta.get('price_value') or 'N/A'}",
            f"Rating: {meta.get('rating_avg')} ({meta.get('rating_count')} ratings) | Sold: {meta.get('sold_count')}",
            f"Seller: {meta.get('seller_name') or 'N/A'}",
        ]
        lines.append("\n".join(header))
        lines.append(chunks_texts[j])
        lines.append("-" * 60)
        included_meta.append(meta)
    return "\n".join(lines), included_meta


SYSTEM_PROMPT = """You are a precise product QA assistant. Answer ONLY using the provided product context.
- Prefer items with higher rating and more ratings if the user asks for "best".
- If the user mentions budget, filter/compare prices accordingly.
- When specific attributes are requested (brand, colors, size), extract from context exactly.
- If you don't find an answer in context, say you don't know.
- Return concise bullet points with: Title, Brand, Price, Rating (avg, count), Sold, Seller, and a URL.
- Do not hallucinate missing values; mark them as N/A.
"""

def answer_with_llm(client: OpenAI, model: str, user_query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User query:\n{user_query}\n\nContext (multiple products & chunks):\n{context}"},
    ]
    # Chat Completions are widely supported; simple + reliable.
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def main():
    t0 = time.time()
    client = OpenAI()

    print(f"[INFO] Loading products from: {ROOT_DIR}")
    docs = iter_product_docs(ROOT_DIR)
    if not docs:
        print("[ERROR] No products found. Is the 'result/' folder present?")
        return
    print(f"[OK] Loaded {len(docs)} products from {ROOT_DIR}")

    print("[INFO] Chunking with Chonkie...")
    chunks_texts, chunks_meta = chunk_docs(docs)
    print(f"[OK] Produced {len(chunks_texts)} chunks (avg per product: {len(chunks_texts)/max(1,len(docs)):.2f})")

    print("[INFO] Embedding chunks (with small local cache for speed)...")
    cache = EmbeddingCache(EMBED_CACHE_PATH)
    chunk_vecs = embed_texts(client, chunks_texts, model=EMBED_MODEL, batch_size=EMBED_BATCH_SIZE, cache=cache)
    print(f"[OK] Embedded chunks => shape {chunk_vecs.shape}")

    print(f"[INFO] Query: {USER_QUERY}")
    query_vec = embed_texts(client, [USER_QUERY], model=EMBED_MODEL, batch_size=1, cache=cache)[0]

    idx = cosine_topk(query_vec, chunk_vecs, k=TOP_K_CHUNKS)
    context, included_meta = build_context(idx, chunks_texts, chunks_meta, limit_products=TOP_PRODUCTS)

    print("[INFO] Asking LLM for the final grounded answer...")
    answer = answer_with_llm(client, CHAT_MODEL, USER_QUERY, context)

    print("\n================= ANSWER =================")
    print(answer)
    print("=========================================\n")

    elapsed = time.time() - t0
    print(f"[DONE] Total time: {elapsed:.2f}s | Products: {len(docs)} | Chunks: {len(chunks_texts)}")


if __name__ == "__main__":
    main()
