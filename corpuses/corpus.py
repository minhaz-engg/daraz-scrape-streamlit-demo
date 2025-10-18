#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a RAG-friendly corpus from Daraz category `products.json` files.

- Input   : result/<category>/products.json
- Outputs : out/daraz_products_corpus.md
            out/daraz_products_corpus.txt
            out/daraz_products_corpus.jsonl

Design for scale:
- Streams writes (MD/TXT/JSONL) to avoid holding full corpus in memory.
- Deduplicates across categories using a set of stable product IDs/keys.
- Caps long fields (description/images/variants) to keep chunks efficient.

Just run:  python build_rag_corpus_all.py
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------
# Config (edit if needed)
# ----------------------------
INPUT_ROOT = Path("./layer_23/result")
OUT_DIR = Path("out")

# Limits to keep chunks clean and embeddings efficient
MAX_IMAGES = 8
MAX_VARIANTS = 20
MAX_DESC_CHARS = 2500  # cap very long descriptions
PRINT_EVERY = 2000     # progress log cadence

# Output file names
MD_PATH = OUT_DIR / "daraz_products_corpus.md"
TXT_PATH = OUT_DIR / "daraz_products_corpus.txt"
JSONL_PATH = OUT_DIR / "daraz_products_corpus.jsonl"

# ----------------------------
# Utilities
# ----------------------------
def normalize_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return u

def category_readable_name(folder: str) -> str:
    # "www_daraz_com_bd_shop_bedding_sets" -> "shop bedding sets"
    prefix = "www_daraz_com_bd_"
    if folder.startswith(prefix):
        folder = folder[len(prefix):]
    return folder.replace("_", " ").strip()

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

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

def list_preview(items: List[str], max_n: int) -> List[str]:
    out = []
    for s in items[:max_n]:
        if isinstance(s, str):
            s = s.strip()
            if s:
                out.append(s)
    return out

def unique_stable_key(prod: Dict[str, Any]) -> str:
    # priority: data_item_id -> detail.url -> detail_url -> product_detail_url -> data_sku_simple -> title+image
    candidates = [
        prod.get("data_item_id"),
        safe_get(prod, "detail", "url"),
        prod.get("detail_url"),
        prod.get("product_detail_url"),
        prod.get("data_sku_simple"),
    ]
    for c in candidates:
        if c:
            return str(c)
    return (prod.get("product_title") or "unknown") + "||" + (prod.get("image_url") or "unknown")

def unify_product(prod: Dict[str, Any], category_folder: str, products_path: Path) -> Dict[str, Any]:
    detail = prod.get("detail") or {}

    title = detail.get("name") or prod.get("product_title") or "Unknown Product"
    brand = detail.get("brand") or None

    # Prices
    price = detail.get("price") or {}
    price_disp = price.get("display") or prod.get("product_price") or None
    price_val = price.get("value")
    orig_disp = price.get("original_display")
    discount_disp = price.get("discount_display")
    discount_pct = price.get("discount_percent")

    # Ratings
    rating = detail.get("rating") or {}
    rating_avg = rating.get("average")
    rating_count = rating.get("count")

    # URLs / images
    url = normalize_url(detail.get("url") or prod.get("detail_url") or prod.get("product_detail_url"))
    image_url = prod.get("image_url")
    images = detail.get("images") or ([image_url] if image_url else [])
    images = [normalize_url(i) for i in images if isinstance(i, str) and i and i.startswith(("http://", "https://", "//"))]
    images = list_preview([i for i in images if i], max_n=MAX_IMAGES)

    # Options / variants
    colors = [c for c in (detail.get("colors") or []) if isinstance(c, str)]
    sizes = [s for s in (detail.get("sizes") or []) if isinstance(s, str)]

    variants = detail.get("variants") or []
    variant_summaries = []
    for v in variants[:MAX_VARIANTS]:
        v_color = v.get("color")
        v_price = safe_get(v, "price", "display") or safe_get(v, "price", "value")
        vsizes = v.get("sizes")
        bits = []
        if v_color:
            bits.append(str(v_color))
        if v_price:
            bits.append(f"price {v_price}")
        if vsizes and isinstance(vsizes, list):
            bits.append("sizes: " + ", ".join([str(x) for x in vsizes[:5]]))
        if bits:
            variant_summaries.append(" — ".join(bits))

    # Delivery / returns / seller
    delivery_options = detail.get("delivery_options") or []
    ret_warranty = detail.get("return_and_warranty") or []

    seller_name = safe_get(detail, "seller", "name")
    seller_link = normalize_url(safe_get(detail, "seller", "link"))
    seller_metrics = safe_get(detail, "seller", "metrics") or {}

    description_text = clean_text(
        safe_get(detail, "details", "description_text") or
        safe_get(detail, "details", "raw_text")
    ) or None

    # Top-level bits
    sku = prod.get("data_sku_simple")
    item_id = prod.get("data_item_id")
    sold_text = prod.get("location")  # e.g., "5.3K sold"
    category_slug = category_readable_name(Path(category_folder).name)

    unified = {
        "id": str(item_id) if item_id else unique_stable_key(prod),
        "title": str(title),
        "brand": brand,
        "category": category_slug,
        "category_dir": str(Path(category_folder).name),
        "source_file": str(products_path),
        "url": url,
        "sku": sku,
        "image_url": image_url if (isinstance(image_url, str) and image_url.startswith(("http://", "https://", "//"))) else None,
        "images": images,

        "listing_price_display": prod.get("product_price"),
        "price_display": price_disp,
        "price_value": price_val,
        "original_price_display": orig_disp,
        "discount_display": discount_disp,
        "discount_percent": discount_pct,

        "rating_average": rating_avg,
        "rating_count": rating_count,
        "sold_text": sold_text,

        "colors": colors,
        "sizes": sizes,
        "variants": variant_summaries,

        "delivery_options": delivery_options,
        "return_and_warranty": ret_warranty,

        "seller_name": seller_name,
        "seller_link": seller_link,
        "seller_metrics": seller_metrics,

        "description": description_text,
    }
    return unified

# ----------------------------
# Writers (streaming)
# ----------------------------
def write_markdown_header(fmd):
    fmd.write("# Daraz Product Corpus\n\n")
    fmd.write("> Document markers: Each product is wrapped by `<!--DOC:START ...-->` and `<!--DOC:END-->` for reliable chunking.\n\n")

def product_to_markdown_block(p: Dict[str, Any]) -> str:
    lines = []
    # explicit start marker for chunkers
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
    # pricing & rating
    price_bits = []
    pd = p.get("price_display") or p.get("listing_price_display")
    if pd: price_bits.append(f"**Price:** {pd}")
    if p.get("original_price_display"): price_bits.append(f"**Original:** {p['original_price_display']}")
    if p.get("discount_display"): price_bits.append(f"**Discount:** {p['discount_display']}")
    rating_bits = []
    if p.get("rating_average") is not None:
        rc = p.get("rating_count")
        rating_bits.append(f"**Rating:** {p['rating_average']}/5" + (f" ({rc} ratings)" if rc is not None else ""))
    if price_bits or rating_bits:
        lines.append("  \n".join(price_bits + rating_bits))
        lines.append("")
    # options
    if p.get("colors"): lines.append("**Colors:** " + ", ".join(p["colors"]))
    if p.get("sizes"): lines.append("**Sizes:** " + ", ".join(p["sizes"]))
    if p.get("variants"):
        lines.append("**Variants (sample):**")
        for v in p["variants"]:
            lines.append(f"- {v}")
    if p.get("colors") or p.get("sizes") or p.get("variants"):
        lines.append("")
    # delivery / warranty
    if p.get("delivery_options"):
        lines.append("**Delivery Options:**")
        for d in p["delivery_options"]:
            if isinstance(d, dict):
                bits = []
                if d.get("title"): bits.append(d["title"])
                if d.get("time"): bits.append(d["time"])
                if d.get("fee"): bits.append(d["fee"])
                if bits: lines.append("- " + " — ".join(bits))
        lines.append("")
    if p.get("return_and_warranty"):
        lines.append("**Return & Warranty:**")
        for rw in p["return_and_warranty"]:
            if rw: lines.append(f"- {rw}")
        lines.append("")
    # seller
    if p.get("seller_name") or p.get("seller_metrics"):
        seller_line = []
        if p.get("seller_name"): seller_line.append(f"**Seller:** {p['seller_name']}")
        if p.get("seller_link"): seller_line.append(f"[Store Link]({p['seller_link']})")
        if seller_line:
            lines.append(" ".join(seller_line))
        metrics = p.get("seller_metrics") or {}
        if metrics:
            mparts = []
            for k, v in metrics.items():
                if v: mparts.append(f"{k}: {v}")
            if mparts:
                lines.append("- " + " | ".join(mparts))
        lines.append("")
    # description
    if p.get("description"):
        lines.append("**Description:**")
        lines.append(p["description"])
        lines.append("")
    # images
    if p.get("images"):
        lines.append("**Images (sample):**")
        for im in p["images"]:
            lines.append(f"- {im}")
        lines.append("")
    # traceability
    src = f"{p.get('category_dir')} / products.json"
    lines.append(f"_Source: {src}_")
    lines.append("")
    lines.append("---")
    # explicit end marker
    lines.append("<!--DOC:END-->")
    lines.append("")  # final newline
    return "\n".join(lines)

def product_to_text_block(p: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"### DOC START | id={p['id']} | category={p.get('category','')}")
    lines.append(f"PRODUCT: {p['title']}")
    if p.get("url"): lines.append(f"URL: {p['url']}")
    if p.get("category"): lines.append(f"CATEGORY: {p['category']}")
    if p.get("brand"): lines.append(f"BRAND: {p['brand']}")
    if p.get("sku"): lines.append(f"SKU: {p['sku']}")
    if p.get("sold_text"): lines.append(f"SALES: {p['sold_text']}")
    pd = p.get("price_display") or p.get("listing_price_display")
    if pd: lines.append(f"PRICE: {pd}")
    if p.get("original_price_display"): lines.append(f"ORIGINAL: {p['original_price_display']}")
    if p.get("discount_display"): lines.append(f"DISCOUNT: {p['discount_display']}")
    if p.get("rating_average") is not None:
        rc = p.get("rating_count")
        lines.append(f"RATING: {p['rating_average']}/5" + (f" ({rc} ratings)" if rc is not None else ""))
    if p.get("colors"): lines.append("COLORS: " + ", ".join(p["colors"]))
    if p.get("sizes"): lines.append("SIZES: " + ", ".join(p["sizes"]))
    if p.get("variants"):
        lines.append("VARIANTS:")
        for v in p["variants"]:
            lines.append(f"- {v}")
    if p.get("delivery_options"):
        lines.append("DELIVERY:")
        for d in p["delivery_options"]:
            if isinstance(d, dict):
                bits = []
                if d.get("title"): bits.append(d["title"])
                if d.get("time"): bits.append(d["time"])
                if d.get("fee"): bits.append(d["fee"])
                if bits: lines.append("- " + " — ".join(bits))
    if p.get("return_and_warranty"):
        lines.append("RETURNS/WARRANTY:")
        for rw in p["return_and_warranty"]:
            if rw: lines.append(f"- {rw}")
    if p.get("seller_name"):
        line = f"SELLER: {p['seller_name']}"
        if p.get("seller_link"): line += f" ({p['seller_link']})"
        lines.append(line)
    metrics = p.get("seller_metrics") or {}
    if metrics:
        mparts = []
        for k, v in metrics.items():
            if v: mparts.append(f"{k}: {v}")
        if mparts: lines.append("SELLER METRICS: " + " | ".join(mparts))
    if p.get("description"):
        lines.append("DESCRIPTION:")
        lines.append(p["description"])
    if p.get("images"):
        lines.append("IMAGES:")
        for im in p["images"]:
            lines.append(f"- {im}")
    src = f"{p.get('category_dir')} / products.json"
    lines.append(f"SOURCE: {src}")
    lines.append("### DOC END")
    lines.append("")  # newline
    return "\n".join(lines)

def product_to_jsonl_record(p: Dict[str, Any]) -> Dict[str, Any]:
    # Compact text for embeddings
    text_lines = []
    text_lines.append(f"{p['title']} (ID: {p['id']})")
    if p.get("brand"): text_lines.append(f"Brand: {p['brand']}")
    if p.get("category"): text_lines.append(f"Category: {p['category']}")
    pd = p.get("price_display") or p.get("listing_price_display")
    if pd: text_lines.append(f"Price: {pd}")
    if p.get("discount_display"): text_lines.append(f"Discount: {p['discount_display']}")
    if p.get("rating_average") is not None:
        rc = p.get("rating_count")
        text_lines.append(f"Rating: {p['rating_average']}/5" + (f" ({rc} ratings)" if rc is not None else ""))
    if p.get("sold_text"): text_lines.append(f"Sales: {p['sold_text']}")
    if p.get("colors"): text_lines.append("Colors: " + ", ".join(p["colors"]))
    if p.get("sizes"): text_lines.append("Sizes: " + ", ".join(p["sizes"]))
    if p.get("description"): text_lines.append("Description: " + p["description"])
    if p.get("url"): text_lines.append(f"URL: {p['url']}")

    metadata = {
        "id": p.get("id"),
        "title": p.get("title"),
        "brand": p.get("brand"),
        "category": p.get("category"),
        "sku": p.get("sku"),
        "url": p.get("url"),
        "price_display": p.get("price_display") or p.get("listing_price_display"),
        "original_price_display": p.get("original_price_display"),
        "discount_display": p.get("discount_display"),
        "discount_percent": p.get("discount_percent"),
        "rating_average": p.get("rating_average"),
        "rating_count": p.get("rating_count"),
        "sold_text": p.get("sold_text"),
        "seller_name": p.get("seller_name"),
        "category_dir": p.get("category_dir"),
        "source_file": p.get("source_file"),
    }
    return {"id": p["id"], "text": "\n".join(text_lines), "metadata": metadata}

# ----------------------------
# Main
# ----------------------------
def main():
    if not INPUT_ROOT.exists():
        raise SystemExit(f"Input folder not found: {INPUT_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare outputs (streaming)
    with MD_PATH.open("w", encoding="utf-8") as fmd, \
         TXT_PATH.open("w", encoding="utf-8") as ftxt, \
         JSONL_PATH.open("w", encoding="utf-8") as fjsonl:

        write_markdown_header(fmd)
        ftxt.write("DARAZ PRODUCT CORPUS\n\n")

        products_files = sorted(INPUT_ROOT.glob("*/products.json"))
        if not products_files:
            raise SystemExit(f"No products.json files found under: {INPUT_ROOT}")

        seen_ids = set()
        total_loaded = 0
        total_written = 0
        categories = set()

        for pfile in products_files:
            cat_dir = pfile.parent
            categories.add(cat_dir.name)

            try:
                data = json.loads(pfile.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    continue
            except Exception as e:
                print(f"[WARN] Failed to parse {pfile}: {e}")
                continue

            for raw in data:
                total_loaded += 1

                uid = unique_stable_key(raw)
                if uid in seen_ids:
                    continue  # dedupe
                seen_ids.add(uid)

                unified = unify_product(raw, str(cat_dir), pfile)

                # Markdown
                fmd.write(product_to_markdown_block(unified))

                # Plain text
                ftxt.write(product_to_text_block(unified))

                # JSONL
                rec = product_to_jsonl_record(unified)
                fjsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

                total_written += 1
                if total_written % PRINT_EVERY == 0:
                    print(f"[INFO] Processed {total_written:,} products "
                          f"(loaded so far: {total_loaded:,}, unique IDs: {len(seen_ids):,})")

        # Summary footer (optional)
        fmd.write(f"\n> Summary: {total_written} products from {len(categories)} categories.\n")
        ftxt.write(f"\nSUMMARY: {total_written} products from {len(categories)} categories.\n")

    print("[OK] Markdown:", MD_PATH)
    print("[OK] Text    :", TXT_PATH)
    print("[OK] JSONL   :", JSONL_PATH)
    print(f"[DONE] Wrote {total_written} unique products across {len(categories)} categories.")

if __name__ == "__main__":
    main()