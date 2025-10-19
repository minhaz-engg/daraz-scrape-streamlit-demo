# streamlit_daraz_timeboxed_rag.py
# -------------------------------------------------------
# Streamlit app: time-boxed Daraz scraping → corpus build → BM25 RAG UI (OpenAI for answers)
# -------------------------------------------------------

import os
import re
import io
import json
import time
import math
import glob
import pickle
import random
import hashlib
import asyncio
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse

import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker

# Crawl4AI / Playwright
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    GeolocationConfig,
    CrawlResult,
)

load_dotenv()

# ----------------------------
# Tunables
# ----------------------------
TIME_BUDGET_SECONDS_DEFAULT = 5 * 60  # 5 minutes
PAGE_PARAM = "page"
PRODUCTS_PER_PAGE_GUESS = 40  # used only for rough progress text
SCRAPE_SLEEP_BETWEEN_PAGES = (0.75, 1.75)

# Paths
APP_ROOT = Path(__file__).parent
RESULT_DIR = (APP_ROOT / "result").resolve()
OUT_DIR = (APP_ROOT / "out").resolve()
INDEX_DIR = (APP_ROOT / "index").resolve()
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# RAG defaults
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 8
DEFAULT_LANG = "en"

# ----------------------------
# Helpers (paths, URLs, JSON)
# ----------------------------
def folder_name_from_url(url: str, max_len: int = 96) -> str:
    p = urlparse(url)
    host = (p.netloc or "site").lower()
    path = (p.path or "/")
    segs = [host] + [s for s in path.split("/") if s]
    raw = "_".join(segs)
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    if not safe:
        safe = "site"
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_")
    return safe

def ensure_category_dir(link: str) -> Dict[str, Path]:
    folder = folder_name_from_url(link)
    category_dir = RESULT_DIR / folder
    category_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": category_dir,
        "schema": category_dir / "schema.json",            # (kept for parity)
        "products": category_dir / "products.json",        # merged/dedup
        "pages_index": category_dir / "pages_index.json",  # page->ids
    }

def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def load_json_if_exists(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def set_query_param(url: str, key: str, value: str) -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q[key] = value
    new_query = urlencode(q, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))

def build_page_url(base_url: str, page: int) -> str:
    return set_query_param(base_url, PAGE_PARAM, str(page))

# ----------------------------
# Product ID & dedupe
# ----------------------------
def get_product_id(card: Dict[str, Any]) -> str:
    pid = (card.get("data_item_id") or "").strip()
    if pid:
        return pid
    sku = (card.get("data_sku_simple") or "").strip()
    if sku:
        return sku.split("_", 1)[0].strip()
    url_guess = card.get("product_detail_url") or card.get("detail_url") or ""
    m = re.search(r"-i(\d+)-s(\d+)\.html", url_guess)
    if m:
        return f"{m.group(1)}_BD-{m.group(2)}"
    # Fallback: title+image (very weak)
    return (card.get("product_title") or "unknown") + "||" + (card.get("image_url") or "unknown")

def dedupe_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in cards:
        pid = get_product_id(c)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(c)
    return out

# ----------------------------
# HTML parsing for Daraz listing cards
# ----------------------------
PRICE_RE = re.compile(r"(?:৳|tk|taka)\s*([0-9][0-9,\.]*)", re.IGNORECASE)
SOLD_RE  = re.compile(r"([0-9]+(?:\.[0-9]+)?[kK]?)\s*sold", re.IGNORECASE)

def _text(el) -> str:
    return (el.get_text(" ", strip=True) if el else "").strip()

def normalize_url(u: Optional[str], base_url: str) -> Optional[str]:
    if not u:
        return None
    s = u.strip()
    if s.startswith("//"):
        return "https:" + s
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return urljoin(base_url, s)

def parse_listing_products(html: str, base_url: str) -> List[Dict[str, Any]]:
    """
    Robust-enough parser for Daraz listing pages.
    Looks for known container selectors and pulls a minimal product card.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    cards = soup.select("[data-qa-locator='product-item'], .Bm3ON, .gridItem, .product-item")
    out: List[Dict[str, Any]] = []

    for div in cards:
        try:
            data_item_id = div.get("data-item-id") or ""
            data_sku_simple = div.get("data-sku-simple") or ""

            # main link (PDP)
            a = div.select_one("a[href*='-i'][href$='.html']") or div.select_one("a[href*='-i']")
            detail_url = normalize_url(a.get("href") if a else None, base_url)

            # title
            title = (a.get("title").strip() if a and a.get("title") else "").strip()
            if not title:
                # common container for title on Daraz
                title = _text(div.select_one(".RfADt a, .RfADt, a"))

            # image
            im = (div.select_one("img[type='product']") or
                  div.select_one("img[data-src]") or
                  div.select_one("img[src]"))
            image_url = None
            if im:
                image_url = im.get("src") or im.get("data-src") or None
                image_url = normalize_url(image_url, base_url)

            # price (grab first money-like chunk near price container)
            price_container = div.select_one(".aBrP0") or div
            price_text = ""
            if price_container:
                # prefer explicit price element
                price_text = _text(price_container.select_one(".ooOxS")) or _text(price_container)
            if not PRICE_RE.search(price_text):
                # scan broader area
                m2 = PRICE_RE.search(_text(div))
                price_text = m2.group(0) if m2 else price_text

            # sold text (optional)
            sold_match = SOLD_RE.search(_text(div))
            sold_text = sold_match.group(0) if sold_match else None

            card = {
                "data_item_id": data_item_id,
                "data_sku_simple": data_sku_simple,
                "product_detail_url": detail_url,
                "product_title": title or None,
                "image_url": image_url,
                "product_price": price_text or None,
                "location": sold_text,  # in your schema this was often "5.3K sold"
            }
            # keep only cards that really look like products
            if card["product_detail_url"] and (card["product_title"] or card["image_url"]):
                out.append(card)
        except Exception:
            continue

    return dedupe_cards(out)

# ----------------------------
# Crawl primitives (time-boxed)
# ----------------------------
def _progress_sleep():
    time.sleep(random.uniform(*SCRAPE_SLEEP_BETWEEN_PAGES))

def _progress_msg(writer, msg: str):
    if writer:
        writer.write(msg)

async def _fetch_html(crawler: AsyncWebCrawler, url: str) -> str:
    """
    Use Crawl4AI/Playwright to fetch rendered HTML. Scrolls a bit to load lazy content.
    """
    js_seq = """
        const sleep = ms => new Promise(r => setTimeout(r, ms));
        await sleep(500);
        for (let y=0; y<3; y++){
            window.scrollBy(0, document.body.scrollHeight/3);
            await sleep(450 + Math.floor(Math.random()*250));
        }
        await sleep(400);
    """
    config = CrawlerRunConfig(
        cache_mode=CacheMode.DISABLED,
        geolocation=GeolocationConfig(latitude=23.8103, longitude=90.4125),
        prettiify=False,
        wait_for_images=False,
        delay_before_return_html=False,
        mean_delay=0.1,
        scroll_delay=0.2,
        verbose=False,
    )
    results: List[CrawlResult] = await crawler.arun(
        url=url, config=config, js_code=js_seq,
        wait_for="css:[data-qa-locator='product-item'], body"
    )
    for res in results or []:
        if res and res.success and res.html:
            return res.html
    return ""

async def scrape_categories_timeboxed(
    category_links: List[str],
    time_budget_seconds: int,
    status_writer=None,
    progress_cb=None
) -> Dict[str, Dict[str, Any]]:
    """
    Time-box scraping across multiple categories.
    Returns { category_folder: { 'products': [..], 'pages_index': {page: [ids]} } }
    """
    start = time.time()
    deadline = start + time_budget_seconds
    out: Dict[str, Dict[str, Any]] = {}

    browser_config = BrowserConfig(
        # enable_stealth=True,
        # headless=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Round-robin through categories, 1 page per turn (good for breadth under time pressure)
        page_cursors = {link: 1 for link in category_links}
        per_cat_storage: Dict[str, Dict[str, Any]] = {}
        for link in category_links:
            paths = ensure_category_dir(link)
            per_cat_storage[link] = {
                "paths": paths,
                "all_by_id": {get_product_id(p): p for p in load_json_if_exists(paths["products"], default=[])},
                "pages_index": load_json_if_exists(paths["pages_index"], default={}),
            }

        while time.time() < deadline:
            any_progress = False

            for link in category_links:
                now = time.time()
                if now >= deadline:
                    break

                cursor = page_cursors[link]
                base_url = build_page_url(link, cursor)

                _progress_msg(status_writer, f"Fetching {base_url}\n")
                try:
                    html = await _fetch_html(crawler, base_url)
                except Exception as e:
                    _progress_msg(status_writer, f"  ❌ fetch failed: {e}\n")
                    continue

                if not html:
                    _progress_msg(status_writer, "  ⚠️ empty HTML; skipping.\n")
                    continue

                cards = parse_listing_products(html, base_url)
                cards = dedupe_cards(cards)
                if not cards:
                    _progress_msg(status_writer, "  ⚠️ no products found on this page.\n")
                    # Try next category / page anyway
                    page_cursors[link] = cursor + 1
                    continue

                paths = per_cat_storage[link]["paths"]
                all_by_id: Dict[str, Any] = per_cat_storage[link]["all_by_id"]
                pages_index: Dict[str, List[str]] = per_cat_storage[link]["pages_index"]

                ids_this_page = []
                new_count = 0
                for c in cards:
                    pid = get_product_id(c)
                    if not pid:
                        continue
                    ids_this_page.append(pid)
                    if pid not in all_by_id:
                        all_by_id[pid] = c
                        new_count += 1

                # Save per-page
                page_file = paths["dir"] / f"page_{cursor}.json"
                atomic_write_json(page_file, cards)
                # Update merged store
                atomic_write_json(paths["products"], list(all_by_id.values()))
                pages_index[str(cursor)] = ids_this_page
                atomic_write_json(paths["pages_index"], pages_index)

                _progress_msg(status_writer, f"  ✅ page {cursor}: {len(cards)} cards ({new_count} new)\n")

                # Next page
                page_cursors[link] = cursor + 1
                any_progress = True

                # pacing
                _progress_sleep()

                # progress bar update
                if progress_cb:
                    progress_cb(min(0.99, (time.time() - start) / (time_budget_seconds + 1e-9)))

            if not any_progress:
                # nothing to do
                break

    # Build return object
    for link in category_links:
        folder = folder_name_from_url(link)
        paths = ensure_category_dir(link)
        out[folder] = {
            "products": load_json_if_exists(paths["products"], default=[]),
            "pages_index": load_json_if_exists(paths["pages_index"], default={})
        }
    return out

# ----------------------------
# Build RAG corpus (Markdown/TXT/JSONL) in memory + write to disk
# ----------------------------
MAX_IMAGES = 8
MAX_VARIANTS = 20
MAX_DESC_CHARS = 2500

def category_readable_name(folder: str) -> str:
    prefix = "www_daraz_com_bd_"
    if folder.startswith(prefix):
        folder = folder[len(prefix):]
    return folder.replace("_", " ").strip()

def clean_text(t: Optional[str]) -> Optional[str]:
    if not t:
        return t
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = t.strip()
    if len(t) > MAX_DESC_CHARS:
        t = t[:MAX_DESC_CHARS].rstrip() + " …"
    return t

def unify_product(prod: Dict[str, Any], category_folder: str, products_path: Path) -> Dict[str, Any]:
    # We only have listing fields; keep structure compatible with your corpus
    title = prod.get("product_title") or "Unknown Product"
    url = prod.get("product_detail_url")
    image_url = prod.get("image_url")
    cat_slug = category_readable_name(Path(category_folder).name)
    pid = prod.get("data_item_id") or get_product_id(prod)

    unified = {
        "id": str(pid),
        "title": title,
        "brand": None,
        "category": cat_slug,
        "category_dir": Path(category_folder).name,
        "source_file": str(products_path),
        "url": url,
        "sku": prod.get("data_sku_simple"),
        "image_url": image_url if (isinstance(image_url, str) and image_url.startswith(("http://","https://","//"))) else None,
        "images": [image_url] if image_url else [],
        "listing_price_display": prod.get("product_price"),
        "price_display": prod.get("product_price"),
        "price_value": None,  # unknown w/o PDP
        "original_price_display": None,
        "discount_display": None,
        "discount_percent": None,
        "rating_average": None,
        "rating_count": None,
        "sold_text": prod.get("location"),
        "colors": [],
        "sizes": [],
        "variants": [],
        "delivery_options": [],
        "return_and_warranty": [],
        "seller_name": None,
        "seller_link": None,
        "seller_metrics": {},
        "description": None,
    }
    return unified

def product_to_markdown_block(p: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"<!--DOC:START id={p['id']} category={p.get('category','')} -->")
    lines.append(f"## {p['title']}  \n**DocID:** {p['id']}")
    lines.append("")
    meta = []
    if p.get("category"): meta.append(f"**Category:** {p['category']}")
    if p.get("brand"): meta.append(f"**Brand:** {p['brand']}")
    if p.get("sku"): meta.append(f"**SKU:** {p['sku']}")
    if p.get("url"): meta.append(f"**URL:** {p['url']}")
    if p.get("sold_text"): meta.append(f"**Sales:** {p['sold_text']}")
    if meta:
        lines.append("  \n".join(meta))
        lines.append("")
    # price/rating
    pd = p.get("price_display") or p.get("listing_price_display")
    if pd: lines.append(f"**Price:** {pd}")
    if p.get("discount_display"): lines.append(f"**Discount:** {p['discount_display']}")
    if p.get("rating_average") is not None:
        rc = p.get("rating_count")
        lines.append(f"**Rating:** {p['rating_average']}/5" + (f" ({rc} ratings)" if rc is not None else ""))
    if p.get("price_display") or p.get("rating_average") is not None:
        lines.append("")
    # description
    if p.get("description"):
        lines.append("**Description:**")
        lines.append(clean_text(p["description"]) or "")
        lines.append("")
    # images
    if p.get("images"):
        lines.append("**Images (sample):**")
        for im in p["images"][:MAX_IMAGES]:
            lines.append(f"- {im}")
        lines.append("")
    src = f"{p.get('category_dir')} / products.json"
    lines.append(f"_Source: {src}_")
    lines.append("")
    lines.append("---")
    lines.append("<!--DOC:END-->")
    lines.append("")
    return "\n".join(lines)

def build_corpus_from_result(result_root: Path) -> Tuple[str, str, List[Dict[str, Any]], List[str]]:
    """
    Scans result/*/products.json → returns:
      md_text, txt_text, unified_products, categories
    Also writes out to ./out/.
    """
    products_files = sorted(result_root.glob("*/products.json"))
    if not products_files:
        return "", "", [], []

    unified_all: List[Dict[str, Any]] = []
    md_parts: List[str] = ["# Daraz Product Corpus\n\n> Built by time-boxed scrape.\n\n"]
    txt_parts: List[str] = ["DARAZ PRODUCT CORPUS\n"]

    seen_ids = set()
    categories = set()

    for pfile in products_files:
        cat_dir = pfile.parent
        categories.add(cat_dir.name)
        try:
            data = json.loads(pfile.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
        except Exception:
            continue

        for raw in data:
            uid = get_product_id(raw)
            if uid in seen_ids:
                continue
            seen_ids.add(uid)

            unified = unify_product(raw, str(cat_dir), pfile)
            unified_all.append(unified)

            md_parts.append(product_to_markdown_block(unified))
            # plain text (compact)
            txt_parts.append(f"### DOC START | id={unified['id']} | category={unified.get('category','')}")
            txt_parts.append(f"PRODUCT: {unified['title']}")
            if unified.get("url"): txt_parts.append(f"URL: {unified['url']}")
            if unified.get("category"): txt_parts.append(f"CATEGORY: {unified['category']}")
            if unified.get("brand"): txt_parts.append(f"BRAND: {unified['brand']}")
            if unified.get("listing_price_display"): txt_parts.append(f"PRICE: {unified['listing_price_display']}")
            if unified.get("sold_text"): txt_parts.append(f"SALES: {unified['sold_text']}")
            txt_parts.append("### DOC END\n")

    md = "\n".join(md_parts) + f"\n> Summary: {len(unified_all)} products from {len(categories)} categories.\n"
    txt = "\n".join(txt_parts) + f"\nSUMMARY: {len(unified_all)} products from {len(categories)} categories.\n"

    # persist (optional)
    (OUT_DIR / "daraz_products_corpus.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "daraz_products_corpus.txt").write_text(txt, encoding="utf-8")
    with (OUT_DIR / "daraz_products_corpus.jsonl").open("w", encoding="utf-8") as fjsonl:
        for p in unified_all:
            # compact record for embeddings/search audit
            rec = {
                "id": p["id"],
                "text": f"{p['title']} — Price: {p.get('price_display') or p.get('listing_price_display') or ''} — URL: {p.get('url') or ''}",
                "metadata": {
                    "title": p["title"],
                    "category": p.get("category"),
                    "url": p.get("url"),
                    "price_display": p.get("price_display") or p.get("listing_price_display"),
                    "sold_text": p.get("sold_text"),
                }
            }
            fjsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return md, txt, unified_all, sorted(categories)

# ----------------------------
# RAG (parse corpus → chunks → BM25 → OpenAI)
# ----------------------------
@dataclass
class ProductDoc:
    doc_id: str
    title: str
    url: Optional[str]
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
    category: Optional[str]
    brand: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    text: str

DOC_BLOCK_RE = re.compile(r"<!--DOC:START(?P<attrs>[^>]*)-->(?P<body>.*?)<!--DOC:END-->", re.DOTALL|re.IGNORECASE)
TITLE_RE = re.compile(r"^##\s+(.+?)\s*(?:\s{2,}\n|\n|$)", re.MULTILINE)
URL_LINE_RE = re.compile(r"\*\*URL:\*\*\s*(\S+)", re.IGNORECASE)
BRAND_RE = re.compile(r"\*\*Brand:\*\*\s*(.+)", re.IGNORECASE)
PRICE_RE_LINE = re.compile(r"\*\*Price:\*\*\s*(.+)", re.IGNORECASE)
RATING_RE_LINE = re.compile(r"\*\*Rating:\*\*\s*([0-9.]+)\s*/\s*5(?:\s*\((\d+)\s*ratings\))?", re.IGNORECASE)

def _meta_from_header(attrs: str) -> Dict[str,str]:
    out = {}
    for kv in attrs.strip().split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def _parse_price_value(s: str) -> Optional[float]:
    s = s.replace(",", "")
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums:
        return None
    try:
        vals = [float(x) for x in nums]
        return min(vals) if vals else None
    except Exception:
        return None

def parse_products_from_md(md_text: str) -> List[ProductDoc]:
    products: List[ProductDoc] = []
    for m in DOC_BLOCK_RE.finditer(md_text):
        attrs = m.group("attrs") or ""
        body = (m.group("body") or "").strip()
        meta = _meta_from_header(attrs)
        doc_id = meta.get("id") or f"doc_{len(products)+1}"
        category = meta.get("category")

        tit_m = TITLE_RE.search(body)
        title = (tit_m.group(1).strip() if tit_m else f"Product {doc_id}")

        url_m = URL_LINE_RE.search(body)
        url = url_m.group(1).strip() if url_m else None

        brand_m = BRAND_RE.search(body)
        brand = brand_m.group(1).strip() if brand_m else None

        price_value = None
        pm = PRICE_RE_LINE.search(body)
        if pm: price_value = _parse_price_value(pm.group(1))

        rating_avg, rating_cnt = None, None
        rm = RATING_RE_LINE.search(body)
        if rm:
            try:    rating_avg = float(rm.group(1))
            except: rating_avg = None
            try:    rating_cnt = int(rm.group(2)) if rm.group(2) else None
            except: rating_cnt = None

        products.append(ProductDoc(
            doc_id=doc_id, title=title, url=url,
            category=category, brand=brand,
            price_value=price_value, rating_avg=rating_avg, rating_cnt=rating_cnt,
            raw_md=body
        ))
    return products

def build_chunker(lang: str = DEFAULT_LANG) -> RecursiveChunker:
    return RecursiveChunker.from_recipe("markdown", lang=lang)

def _clean_for_bm25(text: str) -> str:
    clean = []
    for line in text.splitlines():
        ll = line.strip()
        if not ll: continue
        if ll.lower().startswith("**images"): continue
        if "http://" in ll or "https://" in ll:
            pieces = re.split(r"\s+https?://\S+", ll)
            ll = " ".join([p for p in pieces if p.strip()])
            if not ll: continue
        clean.append(ll)
    return "\n".join(clean)

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
        indexed = _clean_for_bm25(text)
        if not indexed: continue
        chunks.append(ChunkRec(
            doc_id=product.doc_id, title=product.title, url=product.url,
            category=product.category, brand=product.brand,
            price_value=product.price_value, rating_avg=product.rating_avg, rating_cnt=product.rating_cnt,
            text=indexed
        ))
    return chunks

STOPWORDS = set("""
the a an and or of for on in to from with by at is are was were this that these those it its as be can will has have
this that these those it its as be can will has have
""".split())

def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in toks if t not in STOPWORDS]

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _index_paths(sig: str) -> Tuple[str, str]:
    return (
        str(INDEX_DIR / f"bm25_{sig}.pkl"),
        str(INDEX_DIR / f"meta_{sig}.pkl"),
    )

def build_or_load_bm25(products: List[ProductDoc], lang: str) -> Tuple[BM25Okapi, List[ChunkRec], List[List[str]]]:
    chunker = build_chunker(lang)
    all_chunks: List[ChunkRec] = []
    for p in products:
        all_chunks.extend(product_to_chunks(p, chunker))
    content_sig = _sha1("\n".join([c.doc_id + "\t" + c.text for c in all_chunks]))
    sig = _sha1(f"v1|lang={lang}|{content_sig}")
    bm25_pkl, meta_pkl = _index_paths(sig)

    if os.path.exists(bm25_pkl) and os.path.exists(meta_pkl):
        with open(bm25_pkl, "rb") as f: bm25 = pickle.load(f)
        with open(meta_pkl, "rb") as f: meta = pickle.load(f)
        return bm25, meta["chunks"], meta["tokenized_corpus"]

    tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(bm25_pkl, "wb") as f: pickle.dump(bm25, f)
    with open(meta_pkl, "wb") as f: pickle.dump({"tokenized_corpus": tokenized_corpus, "chunks": all_chunks}, f)
    return bm25, all_chunks, tokenized_corpus

def _passes_filters(chunk: ChunkRec,
                    allowed_categories: Optional[set],
                    brand_filter: Optional[str],
                    price_min: Optional[float],
                    price_max: Optional[float],
                    rating_min: Optional[float]) -> bool:
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
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    pairs: List[Tuple[int, float]] = []
    for i, sc in enumerate(scores):
        c = chunks[i]
        if _passes_filters(c, allowed_categories, brand_filter, price_min, price_max, rating_min):
            pairs.append((i, float(sc)))

    # light boost if title/brand match
    q_words = set(q_tokens)
    def _boost(idx: int, s: float) -> float:
        c = chunks[idx]
        boost = 0.0
        title_words = set(_tokenize(c.title))
        brand_words = set(_tokenize(c.brand or ""))
        if q_words & title_words: boost += 0.10 * s
        if q_words & brand_words: boost += 0.05 * s
        return s + boost

    pairs = [(i, _boost(i, s)) for (i, s) in pairs]
    pairs.sort(key=lambda x: x[1], reverse=True)

    if not diversify:
        return [(chunks[i], s) for i, s in pairs[:top_k]]

    seen_docs = set()
    diversified: List[Tuple[ChunkRec, float]] = []
    for i, s in pairs:
        c = chunks[i]
        if c.doc_id in seen_docs: continue
        diversified.append((c, s))
        seen_docs.add(c.doc_id)
        if len(diversified) >= top_k:
            return diversified
    if len(diversified) < top_k:
        for i, s in pairs:
            c = chunks[i]
            diversified.append((c, s))
            if len(diversified) >= top_k: break
    return diversified

def _ensure_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return OpenAI()

def _build_messages(query: str, results: List[Tuple[ChunkRec, float]]) -> List[Dict[str, str]]:
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
    system = ("You are a precise product assistant. Answer ONLY from the provided context. "
              "If the answer isn't present, say you don't know. Keep it concise with bullets. "
              "Cite as [#] with DocID and include URL when available.")
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def stream_answer(model: str, messages: List[Dict[str, str]], temperature: float = 0.2):
    client = _ensure_client()
    resp = client.chat.completions.create(
        model=model, temperature=temperature, messages=messages, stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Daraz RAG (time‑boxed scrape)", layout="wide")
st.title("Daraz Products — Time‑boxed Scrape → RAG (BM25 + OpenAI)")

with st.sidebar:
    st.markdown("### ⚙️ Scrape settings")
    categories_text = st.text_area(
        "Category links (one per line)",
        value="\n".join([
            "https://www.daraz.com.bd/mini-cameras/",
            "https://www.daraz.com.bd/wire-racks/",
        ]),
        height=120
    )
    minutes = st.slider("Time budget (minutes)", 1, 10, 5)
    diversify = st.checkbox("Diversify results (1 chunk per product first)", value=True)
    model = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4.1-mini","gpt-4o"], index=0)
    top_k = st.slider("Top-K chunks", 1, 20, DEFAULT_TOPK)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.caption("Optional: use existing `./result/*/products.json` (skip scraping).")
    use_existing = st.checkbox("Skip scraping; just build corpus from ./result/", value=False)

go = st.button("🚀 Run: Scrape (time‑boxed), Build Corpus & Index")

# Placeholders for progress / status
status_area = st.empty()
log_area = st.empty()
progress_bar = st.progress(0.0)

md_text = ""
products_unified: List[Dict[str, Any]] = []
available_categories: List[str] = []

if go:
    try:
        cat_links = [ln.strip() for ln in categories_text.splitlines() if ln.strip()]
        if not use_existing and not cat_links:
            st.error("Please provide at least one category link or check 'Skip scraping'.")
            st.stop()

        with st.status("Starting pipeline…", expanded=True) as status:
            log_box = log_area.container()
            def _log(msg): log_box.write(msg)
            def _progress(p): progress_bar.progress(p)

            # 1) Scrape time‑boxed
            if not use_existing:
                status.update(label="Scraping category listings (time‑boxed)…")
                _log(f"Budget: {minutes} minutes across {len(cat_links)} categories.")
                asyncio.run(scrape_categories_timeboxed(
                    cat_links, minutes * 60,
                    status_writer=log_box, progress_cb=_progress
                ))
                _log("Scraping finished.")

            progress_bar.progress(1.0)

            # 2) Build corpus
            status.update(label="Building RAG corpus…")
            md_text, txt_text, products_unified, available_categories = build_corpus_from_result(RESULT_DIR)
            if not md_text:
                st.error("No products found under ./result/.")
                st.stop()
            _log(f"Corpus contains {len(products_unified)} products across {len(available_categories)} categories.")
            progress_bar.progress(1.0)

            # 3) Parse docs & build BM25
            status.update(label="Parsing corpus & building BM25 index…")
            docs = parse_products_from_md(md_text)
            bm25, chunk_table, tokenized_corpus = build_or_load_bm25(docs, lang=DEFAULT_LANG)
            status.update(label="Index ready.", state="complete")
            st.success(f"Parsed **{len(docs):,}** docs → **{len(chunk_table):,}** chunks. Categories: {', '.join(available_categories[:10])}{' …' if len(available_categories)>10 else ''}")

        # --- Search UI ---
        st.markdown("### Filters")
        c1, c2, c3, c4 = st.columns([1.7, 1.3, 1.2, 1.2])
        with c1:
            sel_categories = st.multiselect("Category", options=available_categories, default=[])
        with c2:
            brand_filter = st.text_input("Brand contains", "")
        with c3:
            price_max_ui = st.text_input("Max price (BDT)", "")
        with c4:
            rating_min_ui = st.text_input("Min rating (0-5)", "")

        def _to_float(x: str) -> Optional[float]:
            x = (x or "").strip().replace(",", "")
            if not x: return None
            m = re.match(r"^\d+(?:\.\d+)?$", x)
            return float(x) if m else None

        price_max_filter = _to_float(price_max_ui)
        rating_min_filter = _to_float(rating_min_ui)

        st.markdown("---")
        query = st.text_input("Ask about products (e.g., 'best wire rack under 2000')", "")
        goq = st.button("🔎 Search")

        if goq and query.strip():
            allowed_categories = set(sel_categories) if sel_categories else None
            with st.spinner("Retrieving with BM25…"):
                results = bm25_search(
                    bm25, chunk_table, tokenized_corpus, query,
                    top_k=top_k,
                    allowed_categories=allowed_categories,
                    brand_filter=brand_filter if brand_filter.strip() else None,
                    price_min=None,
                    price_max=price_max_filter,
                    rating_min=rating_min_filter,
                    diversify=diversify,
                )
            if not results:
                st.warning("No results matched your query/filters.")
            else:
                colL, colR = st.columns([0.55, 0.45], gap="large")
                with colL:
                    st.subheader("Top matches")
                    for i, (chunk, score) in enumerate(results, 1):
                        meta_bits = []
                        if chunk.brand: meta_bits.append(f"**Brand:** {chunk.brand}")
                        if chunk.category: meta_bits.append(f"**Category:** {chunk.category}")
                        if chunk.price_value is not None: meta_bits.append(f"**Price:** ~৳{int(chunk.price_value)}")
                        if chunk.rating_avg is not None:
                            rc = f" ({chunk.rating_cnt} ratings)" if chunk.rating_cnt is not None else ""
                            meta_bits.append(f"**Rating:** {chunk.rating_avg}/5{rc}")
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
                        st.error(f"OpenAI error: {e}")

                # Export results (JSON)
                export_rows = []
                for i, (c, s) in enumerate(results, 1):
                    export_rows.append({
                        "rank": i, "score": s, "doc_id": c.doc_id, "title": c.title, "url": c.url or "",
                        "category": c.category or "", "brand": c.brand or "",
                        "price_value": c.price_value if c.price_value is not None else "",
                        "rating_avg": c.rating_avg if c.rating_avg is not None else "",
                        "rating_cnt": c.rating_cnt if c.rating_cnt is not None else "",
                        "chunk_text": c.text[:2000],
                    })
                export_bytes = io.BytesIO()
                export_bytes.write(json.dumps(export_rows, ensure_ascii=False, indent=2).encode("utf-8"))
                export_bytes.seek(0)
                st.download_button("Download results (JSON)", data=export_bytes, file_name="results.json", mime="application/json")

    except Exception as e:
        st.error("Pipeline failed:\n\n" + "".join(traceback.format_exception(e)))
        st.stop()

# ----------------------------
# Integration hooks (optional)
# ----------------------------
# 🔁 If you want to reuse your existing advanced crawler & PDP enrichment:
#   - Replace `parse_listing_products(...)` with your `JsonCssExtractionStrategy` results (cards)
#   - Call your `enrich_products_with_details(...)` before saving per-page or final products.json
#   - Keep file outputs under ./result/<category> so the corpus builder continues to work
