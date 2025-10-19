import os
import re
import io
import json
import time
import math
import asyncio
import hashlib
import pickle
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin, urlencode, parse_qsl, urlunparse

import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from openai import OpenAI
import nest_asyncio

# --- init env / loop ---
load_dotenv()
nest_asyncio.apply()

# ---------- Config ----------
APP_TITLE = "Daraz + Star Tech: 5‑minute Scraper → RAG Search"
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_BUDGET_SEC = 5 * 60   # 5 minutes
MAX_CONCURRENCY_PER_SITE = int(os.getenv("MAX_CONCURRENCY_PER_SITE", "3"))
DARAZ_PRODUCTS_PER_PAGE = 40  # heuristic only, we autodetect end by empty page
STARTECH_PRODUCTS_PER_PAGE = 40
MAX_PAGES_PER_CATEGORY = 50   # safety cap; time budget usually stops earlier

BM25_CACHE_DIR = os.path.join(OUT_DIR, "index")
os.makedirs(BM25_CACHE_DIR, exist_ok=True)

# ---------------- crawl4ai ----------------
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

# ---------------- Utilities ----------------
def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def set_query_param(url: str, key: str, value: str) -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q[key] = value
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), p.fragment))

def category_from_url(url: str) -> str:
    p = urlparse(url)
    segs = [s for s in p.path.split("/") if s]
    return segs[-1] if segs else p.netloc

def parse_price_value(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace(",", "")
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums:
        return None
    try:
        vals = [float(x) for x in nums]
        return min(vals) if vals else None
    except Exception:
        return None

def unique_id_from_url(url: str) -> str:
    # stable-ish id from URL path
    p = urlparse(url)
    return hashlib.sha1((p.netloc + p.path).encode("utf-8")).hexdigest()[:16]

def atomic_write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------- Product schema ----------------
@dataclass
class Product:
    source: str               # "daraz" | "startech"
    category: str
    title: str
    url: str
    price_display: Optional[str] = None
    price_value: Optional[float] = None
    rating_avg: Optional[float] = None
    rating_cnt: Optional[int] = None
    availability: Optional[str] = None
    image_url: Optional[str] = None
    id: Optional[str] = None  # stable id, dedupe key


# ---------------- HTML fetching ----------------
async def fetch_html(url: str, wait_css: Optional[str] = None) -> Optional[str]:
    """
    Render page using crawl4ai/Playwright; returns full HTML or None on failure.
    """
    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.DISABLED,
        prettiify=False,
        wait_for_images=False,
        delay_before_return_html=True if wait_css else False,
        mean_delay=0.2,
        scroll_delay=0.3,
        verbose=False,
    )
    js_scroll = """
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    for (let i=0;i<4;i++){ window.scrollBy(0, document.body.scrollHeight/3); await sleep(400); }
    """

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=cfg, js_code=js_scroll, wait_for=(f"css:{wait_css}" if wait_css else None))
            # crawl4ai returns a single CrawlResult for a single URL
            if result and hasattr(result, "html") and result.html:
                return result.html
            # older versions might return list
            if isinstance(result, list):
                for r in result:
                    if getattr(r, "html", None):
                        return r.html
    except Exception:
        return None
    return None


# ---------------- Parsers ----------------
def parse_daraz_list(html: str, base_url: str, category: str) -> List[Product]:
    soup = BeautifulSoup(html, "lxml")

    # Cards on Daraz often have these markers (they do change; we try several)
    cards = soup.select("[data-qa-locator='product-item'], .Bm3ON, .gridItem")
    out: List[Product] = []

    for c in cards:
        # URL
        a = c.select_one("a[href]")
        if not a:
            continue
        href = a.get("href")
        if not href:
            continue
        url = urljoin(base_url, href)

        # Title
        title_el = c.select_one(".RfADt a, a[title]")
        title = (title_el.get("title") or title_el.get_text(" ", strip=True)) if title_el else a.get_text(" ", strip=True)
        if not title:
            # sometimes the first <img alt> holds name
            img = c.select_one("img[alt]")
            title = img.get("alt").strip() if img and img.get("alt") else None
        if not title:
            continue

        # Price
        price_el = c.select_one(".aBrP0 .ooOxS, .aBrP0 span, .price, [class*=price]")
        price_display = price_el.get_text(" ", strip=True) if price_el else None
        price_value = parse_price_value(price_display)

        # Rating (rarely on SRP; keep best-effort)
        rating_avg = None
        rating_cnt = None
        rnode = c.select_one("[aria-label*='out of 5'], .rating, .rating__score")
        if rnode:
            m = re.search(r"([0-5](?:\.\d+)?)", rnode.get_text(" ", strip=True))
            if m:
                try:
                    rating_avg = float(m.group(1))
                except Exception:
                    pass

        # Image
        img = c.select_one("img[type='product'], img")
        image_url = img.get("src") if img and img.get("src") else None
        if image_url and image_url.startswith("//"):
            image_url = "https:" + image_url

        # ID: try data attributes or derive from URL
        pid = c.get("data-item-id") or c.get("data_item_id") or c.get("data-sku-simple")
        if not pid:
            # try PDP pattern ...-i<id>-s<shop>.html
            m = re.search(r"-i(\d+)-s\d+\.html", url)
            pid = m.group(1) if m else unique_id_from_url(url)

        out.append(Product(
            source="daraz",
            category=category,
            title=title.strip(),
            url=url,
            price_display=price_display,
            price_value=price_value,
            rating_avg=rating_avg,
            rating_cnt=rating_cnt,
            availability=None,
            image_url=image_url,
            id=str(pid),
        ))
    return out


def parse_startech_list(html: str, base_url: str, category: str) -> List[Product]:
    soup = BeautifulSoup(html, "lxml")
    cards = soup.select(".p-item, .product-layout")
    out: List[Product] = []

    for c in cards:
        title_el = c.select_one(".p-item-name a, .product-name a, h4 a")
        if not title_el:
            continue
        url = urljoin(base_url, title_el.get("href"))
        title = title_el.get_text(" ", strip=True)

        price_el = c.select_one(".p-item-price, .price-new, .price")
        price_display = price_el.get_text(" ", strip=True) if price_el else None
        price_value = parse_price_value(price_display)

        status_el = c.select_one(".p-item-stock, .stock-status, .status")
        status = status_el.get_text(" ", strip=True) if status_el else None

        img = c.select_one("img")
        image_url = urljoin(base_url, img.get("data-src") or img.get("src")) if img else None

        out.append(Product(
            source="startech",
            category=category,
            title=title,
            url=url,
            price_display=price_display,
            price_value=price_value,
            rating_avg=None,
            rating_cnt=None,
            availability=status,
            image_url=image_url,
            id=unique_id_from_url(url),
        ))
    return out


# ---------------- Site scrapers (time‑boxed) ----------------
async def crawl_daraz_category(cat_url: str, deadline: float, max_pages: int) -> List[Product]:
    base = cat_url
    category = category_from_url(cat_url)
    seen: set = set()
    items: List[Product] = []

    page = 1
    while time.monotonic() < deadline and page <= max_pages:
        url = set_query_param(base, "page", str(page))
        # simple cache-buster to avoid stale content
        url = set_query_param(url, "_v", str(int(time.time()*1000)))
        html = await fetch_html(url, wait_css="[data-qa-locator='product-item'], .Bm3ON, .gridItem")
        if not html:
            break
        products = parse_daraz_list(html, base, category)
        new_added = 0
        for p in products:
            if p.id in seen:
                continue
            seen.add(p.id)
            items.append(p)
            new_added += 1
        if new_added == 0:
            break
        page += 1
    return items


async def crawl_startech_category(cat_url: str, deadline: float, max_pages: int) -> List[Product]:
    base = cat_url
    category = category_from_url(cat_url)
    seen: set = set()
    items: List[Product] = []

    page = 1
    while time.monotonic() < deadline and page <= max_pages:
        url = set_query_param(base, "page", str(page))
        html = await fetch_html(url, wait_css="body")
        if not html:
            break
        products = parse_startech_list(html, base, category)
        new_added = 0
        for p in products:
            if p.id in seen:
                continue
            seen.add(p.id)
            items.append(p)
            new_added += 1
        if new_added == 0:
            break
        page += 1
    return items


async def timeboxed_scrape(daraz_links: List[str], startech_links: List[str]) -> List[Product]:
    """
    Scrape both sites concurrently, but never exceed TIME_BUDGET_SEC.
    """
    deadline = time.monotonic() + TIME_BUDGET_SEC

    sem_d = asyncio.Semaphore(MAX_CONCURRENCY_PER_SITE)
    sem_s = asyncio.Semaphore(MAX_CONCURRENCY_PER_SITE)

    async def run_d(url):
        async with sem_d:
            return await crawl_daraz_category(url, deadline, MAX_PAGES_PER_CATEGORY)

    async def run_s(url):
        async with sem_s:
            return await crawl_startech_category(url, deadline, MAX_PAGES_PER_CATEGORY)

    tasks = [run_d(u) for u in daraz_links] + [run_s(u) for u in startech_links]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_products: List[Product] = []
    for r in results:
        if isinstance(r, Exception):
            continue
        all_products.extend(r)

    # de-duplicate across categories by global id + source
    seen = set()
    unique: List[Product] = []
    for p in all_products:
        key = (p.source, p.id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


# ---------------- Corpus builders ----------------
def product_to_md_block(p: Product) -> str:
    lines = []
    lines.append(f"<!--DOC:START id={p.id} source={p.source} category={p.category}-->")
    lines.append(f"## {p.title}  \n**DocID:** {p.id}")
    meta = []
    meta.append(f"**Source:** {p.source}")
    meta.append(f"**Category:** {p.category}")
    if p.url: meta.append(f"**URL:** {p.url}")
    if p.price_display: meta.append(f"**Price:** {p.price_display}")
    if p.rating_avg is not None:
        rc = f" ({p.rating_cnt} ratings)" if p.rating_cnt is not None else ""
        meta.append(f"**Rating:** {p.rating_avg}/5{rc}")
    if p.availability: meta.append(f"**Status:** {p.availability}")
    lines.append("  \n".join(meta))
    if p.image_url:
        lines.append("")
        lines.append("**Images (sample):**")
        lines.append(f"- {p.image_url}")
    lines.append("")
    lines.append(f"_Source: {p.source} — scraped {now_ts()}_")
    lines.append("")
    lines.append("---")
    lines.append("<!--DOC:END-->")
    lines.append("")
    return "\n".join(lines)


def build_corpus(products: List[Product]) -> Tuple[str, str]:
    """
    Returns (markdown_corpus, jsonl_bytes as str)
    """
    md_parts = ["# Combined Product Corpus (Daraz + Star Tech)\n"]
    jsonl_lines = []
    for p in products:
        md_parts.append(product_to_md_block(p))
        meta = {
            "id": p.id, "title": p.title, "source": p.source, "category": p.category,
            "url": p.url, "price_display": p.price_display, "price_value": p.price_value,
            "rating_avg": p.rating_avg, "rating_cnt": p.rating_cnt
        }
        text = f"{p.title}\nSource: {p.source}\nCategory: {p.category}\nPrice: {p.price_display or ''}\nURL: {p.url or ''}"
        jsonl_lines.append(json.dumps({"id": p.id, "text": text, "metadata": meta}, ensure_ascii=False))
    return "\n".join(md_parts), "\n".join(jsonl_lines)


# ---------------- Parsing corpus → ProductDoc/Chunk ----------------
DOC_BLOCK_RE = re.compile(r"<!--DOC:START(?P<attrs>[^>]*)-->(?P<body>.*?)<!--DOC:END-->", re.DOTALL|re.IGNORECASE)
TITLE_RE = re.compile(r"^##\s+(.+?)\s*(?:\s{2,}\n|\n|$)", re.MULTILINE)
URL_LINE_RE = re.compile(r"\*\*URL:\*\*\s*(\S+)", re.IGNORECASE)
SOURCE_RE = re.compile(r"\*\*Source:\*\*\s*(.+)", re.IGNORECASE)
CATEGORY_RE = re.compile(r"\*\*Category:\*\*\s*(.+)", re.IGNORECASE)
PRICE_RE = re.compile(r"\*\*Price:\*\*\s*(.+)", re.IGNORECASE)
RATING_RE = re.compile(r"\*\*Rating:\*\*\s*([0-5](?:\.\d+)?)", re.IGNORECASE)

def _meta_from_header(attrs: str) -> Dict[str,str]:
    out = {}
    for kv in attrs.strip().split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

@dataclass
class ProductDoc:
    doc_id: str
    title: str
    url: Optional[str]
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    raw_md: str

@dataclass
class ChunkRec:
    doc_id: str
    title: str
    url: Optional[str]
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    text: str

def parse_products_from_md(md_text: str) -> List[ProductDoc]:
    prods: List[ProductDoc] = []
    for m in DOC_BLOCK_RE.finditer(md_text):
        attrs = _meta_from_header(m.group("attrs") or "")
        body = (m.group("body") or "").strip()
        doc_id = attrs.get("id") or f"doc_{len(prods)+1}"
        title_m = TITLE_RE.search(body)
        title = title_m.group(1).strip() if title_m else f"Product {doc_id}"
        url_m = URL_LINE_RE.search(body)
        url = url_m.group(1).strip() if url_m else None
        src_m = SOURCE_RE.search(body)
        src = src_m.group(1).strip().lower() if src_m else attrs.get("source")
        cat_m = CATEGORY_RE.search(body)
        cat = cat_m.group(1).strip() if cat_m else attrs.get("category")
        price_m = PRICE_RE.search(body)
        price_value = parse_price_value(price_m.group(1)) if price_m else None
        rating_m = RATING_RE.search(body)
        rating_avg = float(rating_m.group(1)) if rating_m else None

        prods.append(ProductDoc(
            doc_id=doc_id, title=title, url=url,
            source=src, category=cat, price_value=price_value,
            rating_avg=rating_avg, rating_cnt=None, raw_md=body
        ))
    return prods


# ---------------- Chunking + BM25 ----------------
STOPWORDS = set([
    "the","a","an","and","or","of","for","on","in","to","from","with","by","at","is","are","was","were",
    "this","that","these","those","it","its","as","be","can","will","has","have"
])

def simple_markdown_chunker(text: str, max_chars: int = 1200) -> List[str]:
    """
    Lightweight chunker:
      - split on top-level headings / blank lines / list groups
      - merge up to ~max_chars
    """
    raw = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]
    chunks: List[str] = []
    cur = ""
    for block in raw:
        if len(cur) + len(block) + 2 <= max_chars:
            cur = (cur + "\n\n" + block).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = block
    if cur:
        chunks.append(cur)
    return chunks

def clean_for_bm25(text: str) -> str:
    out = []
    for line in text.splitlines():
        ll = line.strip()
        if not ll: continue
        if ll.lower().startswith("**images"): continue
        # strip URLs from index text
        ll = re.sub(r"\s+https?://\S+", "", ll).strip()
        if ll:
            out.append(ll)
    return "\n".join(out)

def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in toks if t not in STOPWORDS]

def build_bm25(products: List[ProductDoc]) -> Tuple[BM25Okapi, List[ChunkRec], List[List[str]], str]:
    # Build chunks
    chunks: List[ChunkRec] = []
    for p in products:
        for text in simple_markdown_chunker(p.raw_md):
            idx_text = clean_for_bm25(text)
            if not idx_text:
                continue
            chunks.append(ChunkRec(
                doc_id=p.doc_id, title=p.title, url=p.url, source=p.source,
                category=p.category, price_value=p.price_value,
                rating_avg=p.rating_avg, rating_cnt=p.rating_cnt, text=idx_text
            ))

    # Cache signature
    sig = hashlib.sha1("\n".join([c.doc_id + "\t" + c.text for c in chunks]).encode("utf-8")).hexdigest()
    bm25_pkl = os.path.join(BM25_CACHE_DIR, f"bm25_{sig}.pkl")
    meta_pkl = os.path.join(BM25_CACHE_DIR, f"meta_{sig}.pkl")

    if os.path.exists(bm25_pkl) and os.path.exists(meta_pkl):
        with open(bm25_pkl, "rb") as f:
            bm25 = pickle.load(f)
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        return bm25, meta["chunks"], meta["tokenized_corpus"], sig

    tokenized = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(bm25_pkl, "wb") as f:
        pickle.dump(bm25, f)
    with open(meta_pkl, "wb") as f:
        pickle.dump({"chunks": chunks, "tokenized_corpus": tokenized}, f)
    return bm25, chunks, tokenized, sig


# ---------------- Retrieval ----------------
def passes_filters(c: ChunkRec,
                   sources: Optional[set], categories: Optional[set],
                   price_max: Optional[float], rating_min: Optional[float]) -> bool:
    if sources and (c.source not in sources): return False
    if categories and (c.category not in categories): return False
    if price_max is not None and (c.price_value is not None) and (c.price_value > price_max): return False
    if rating_min is not None and (c.rating_avg is not None) and (c.rating_avg < rating_min): return False
    return True

def bm25_search(bm25: BM25Okapi, chunks: List[ChunkRec], tok: List[List[str]],
                query: str, k: int,
                sources: Optional[set], categories: Optional[set],
                price_max: Optional[float], rating_min: Optional[float]) -> List[Tuple[ChunkRec, float]]:
    qtok = tokenize(query)
    scores = bm25.get_scores(qtok)
    pairs = []
    for i, sc in enumerate(scores):
        c = chunks[i]
        if passes_filters(c, sources, categories, price_max, rating_min):
            pairs.append((i, float(sc)))
    pairs.sort(key=lambda x: x[1], reverse=True)

    # Diversify: first unique doc_ids
    seen = set()
    results: List[Tuple[ChunkRec, float]] = []
    for i, s in pairs:
        if chunks[i].doc_id in seen: continue
        results.append((chunks[i], s))
        seen.add(chunks[i].doc_id)
        if len(results) >= k:
            break
    return results


# ---------------- OpenAI helpers ----------------
def ensure_openai() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return OpenAI()

def build_messages(query: str, results: List[Tuple[ChunkRec, float]]) -> List[Dict[str, str]]:
    ctx = []
    for i, (c, s) in enumerate(results, 1):
        head = f"[{i}] {c.title} — DocID: {c.doc_id}" + (f" — {c.url}" if c.url else "")
        fields = []
        if c.source: fields.append(f"Source: {c.source}")
        if c.category: fields.append(f"Category: {c.category}")
        if c.price_value is not None: fields.append(f"PriceValue: {int(c.price_value)}")
        if c.rating_avg is not None: fields.append(f"Rating: {c.rating_avg}/5")
        meta = " | ".join(fields)
        ctx.append(f"{head}\n{meta}\n---\n{c.text}\n")
    system = (
        "You are a precise shopping assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say you don't know. Use bullets. "
        "Cite as [#] with DocID and include URL when available."
    )
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def stream_answer(model: str, messages: List[Dict[str, str]], temperature: float = 0.2):
    client = ensure_openai()
    resp = client.chat.completions.create(model=model, temperature=temperature, messages=messages, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.markdown("### Settings")
    st.caption("Enter category URLs (one per line). Defaults provided:")
    daraz_default = "\n".join([
        "https://www.daraz.com.bd/mini-cameras/",
        "https://www.daraz.com.bd/shop-bedding-sets/",
    ])
    startech_default = "\n".join([
        "https://www.startech.com.bd/laptop-notebook/laptop",
        "https://www.startech.com.bd/component/processor",
    ])
    daraz_text = st.text_area("Daraz categories", value=daraz_default, height=110)
    startech_text = st.text_area("Star Tech categories", value=startech_default, height=110)

    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    top_k = st.slider("Top‑K chunks", 1, 20, 8)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.caption("Scraping runtime is hard-capped at 5 minutes.")
    go = st.button("▶️ Run 5‑minute scrape + Build corpus + Index")

daraz_links = [u.strip() for u in daraz_text.splitlines() if u.strip()]
startech_links = [u.strip() for u in startech_text.splitlines() if u.strip()]

# Session store
if "products" not in st.session_state:
    st.session_state.products = []
if "md_corpus" not in st.session_state:
    st.session_state.md_corpus = ""
if "jsonl_corpus" not in st.session_state:
    st.session_state.jsonl_corpus = ""
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "tok" not in st.session_state:
    st.session_state.tok = None
if "sig" not in st.session_state:
    st.session_state.sig = None

if go:
    st.info("Starting time‑boxed scraping…")
    with st.spinner("Scraping Daraz + Star Tech (5 minutes max)…"):
        products = asyncio.run(timeboxed_scrape(daraz_links, startech_links))
    st.success(f"Scraped {len(products)} unique products.")

    # Save raw scrape
    raw_path = os.path.join(OUT_DIR, f"scraped_{now_ts()}.json")
    atomic_write_json(raw_path, [p.__dict__ for p in products])
    st.caption(f"Saved raw products → `{raw_path}`")

    # Build corpus
    md, jsonl = build_corpus(products)
    st.session_state.products = products
    st.session_state.md_corpus = md
    st.session_state.jsonl_corpus = jsonl

    md_path = os.path.join(OUT_DIR, "products_corpus.md")
    jsonl_path = os.path.join(OUT_DIR, "products_corpus.jsonl")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(jsonl)
    st.caption(f"Corpus written: `{md_path}`, `{jsonl_path}`")

    # Parse + index
    with st.spinner("Parsing corpus & building BM25…"):
        docs = parse_products_from_md(md)
        bm25, chunks, tok, sig = build_bm25(docs)
        st.session_state.bm25 = bm25
        st.session_state.chunks = chunks
        st.session_state.tok = tok
        st.session_state.sig = sig
    st.success(f"BM25 index ready — {len(docs)} docs → {len(chunks)} chunks. (sig={sig[:8]}…)")

# If we have an index, show search UI
if st.session_state.bm25 is not None:
    # Facets
    all_sources = sorted({c.source for c in st.session_state.chunks if c.source})
    all_categories = sorted({c.category for c in st.session_state.chunks if c.category})

    st.markdown("### Filters")
    col1, col2, col3, col4 = st.columns([1.2, 2, 1.2, 1.2])
    with col1:
        sel_sources = st.multiselect("Source", options=all_sources, default=[])
    with col2:
        sel_categories = st.multiselect("Category", options=all_categories, default=[])
    with col3:
        price_max_ui = st.text_input("Max price (BDT)", "")
    with col4:
        rating_min_ui = st.text_input("Min rating (0-5)", "")

    def _num(x: str) -> Optional[float]:
        x = x.strip().replace(",", "")
        if not x: return None
        return float(x) if re.match(r"^\d+(\.\d+)?$", x) else None

    price_max = _num(price_max_ui)
    rating_min = _num(rating_min_ui)
    sources_set = set(sel_sources) if sel_sources else None
    cats_set = set(sel_categories) if sel_categories else None

    st.markdown("---")
    query = st.text_input("Ask about products (e.g., 'budget laptop under 60000, rating 4+ from Star Tech')", "")
    goq = st.button("🔎 Search")

    if goq and query.strip():
        with st.spinner("Retrieving…"):
            results = bm25_search(
                st.session_state.bm25,
                st.session_state.chunks,
                st.session_state.tok,
                query,
                k=top_k,
                sources=sources_set,
                categories=cats_set,
                price_max=price_max,
                rating_min=rating_min,
            )
        if not results:
            st.warning("No results matched your query/filters.")
        else:
            colL, colR = st.columns([0.55, 0.45], gap="large")
            with colL:
                st.subheader("Top matches")
                for i, (c, s) in enumerate(results, 1):
                    meta = []
                    if c.source: meta.append(f"**Source:** {c.source}")
                    if c.category: meta.append(f"**Category:** {c.category}")
                    if c.price_value is not None: meta.append(f"**Price:** ~৳{int(c.price_value)}")
                    if c.rating_avg is not None: meta.append(f"**Rating:** {c.rating_avg}/5")
                    st.markdown(
                        f"**[{i}] {c.title}**  \n"
                        f"DocID: `{c.doc_id}` • Score: `{s:.3f}`  \n"
                        f"{'URL: ' + c.url if c.url else ''}  \n"
                        + ("  \n".join(meta) if meta else "")
                    )
                    with st.expander("View chunk"):
                        st.write(c.text)

            with colR:
                st.subheader("Answer")
                msgs = build_messages(query, results)
                try:
                    st.write_stream(stream_answer(model, msgs, temperature=temperature))
                except Exception as e:
                    st.error(f"OpenAI error: {e}")

            # Export button
            export = []
            for i, (c, s) in enumerate(results, 1):
                export.append({
                    "rank": i, "score": s, "doc_id": c.doc_id, "title": c.title,
                    "source": c.source, "category": c.category, "url": c.url or "",
                    "price_value": c.price_value if c.price_value is not None else "",
                    "rating_avg": c.rating_avg if c.rating_avg is not None else "",
                    "chunk_text": c.text[:2000],
                })
            b = io.BytesIO(json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"))
            b.seek(0)
            st.download_button("Download results (JSON)", data=b, file_name="results.json", mime="application/json")

else:
    st.info("Enter category links and click **Run 5‑minute scrape + Build corpus + Index** to begin.")

st.markdown("---")
st.caption("Please respect each website’s Terms of Service and robots.txt. This demo performs polite, time‑boxed collection for research/prototyping.")
