# FILE: build_corpus.py
import json
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

print("--- STARTING CORPUS BUILD ---")

# --- Configuration ---
DARAZ_INPUT_ROOT = Path("./result")
STARTECH_INPUT_FILE = Path("./startech_products.csv")
OUT_DIR = Path("out")
MD_PATH = OUT_DIR / "combined_corpus.md"
JSONL_PATH = OUT_DIR / "combined_corpus.jsonl"

# Limits (from your original script)
MAX_IMAGES = 8
MAX_DESC_CHARS = 2500
PRINT_EVERY = 1000

# --- Utilities (borrowed from your script) ---
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def clean_text(t: Optional[str]) -> Optional[str]:
    if not t: return t
    t = re.sub(r"\s+\n", "\n", t).strip()
    if len(t) > MAX_DESC_CHARS:
        t = t[:MAX_DESC_CHARS].rstrip() + " …"
    return t

def normalize_url(u: Optional[str], base: str = "") -> Optional[str]:
    if not u: return None
    u = u.strip()
    if u.startswith("//"): return "https:" + u
    if u.startswith("http"): return u
    if base: return urljoin(base, u)
    return u

# --- Unification Functions ---

def unify_daraz_product(prod: Dict[str, Any], category_folder: str) -> Dict[str, Any]:
    """Converts a complex Daraz product dict to our simple unified format."""
    detail = prod.get("detail") or {}
    
    # Get readable category
    prefix = "www_daraz_com_bd_"
    if category_folder.startswith(prefix):
        category_folder = category_folder[len(prefix):]
    category_slug = category_folder.replace("_", " ").strip()

    # Get price
    price_disp = (
        safe_get(detail, "price", "display") or 
        prod.get("product_price") or 
        "N/A"
    )
    price_val = safe_get(detail, "price", "value")
    if price_val is None:
        try:
            price_val = float(re.sub(r"[^\d.]", "", price_disp))
        except Exception:
            price_val = None

    return {
        "id": prod.get("data_item_id") or prod.get("data_sku_simple") or str(prod.get("product_detail_url")),
        "source": "Daraz",
        "title": detail.get("name") or prod.get("product_title") or "Unknown Product",
        "brand": detail.get("brand"),
        "category": category_slug,
        "url": normalize_url(detail.get("url") or prod.get("detail_url") or prod.get("product_detail_url")),
        "price_display": price_disp,
        "price_value": price_val,
        "rating_average": safe_get(detail, "rating", "average"),
        "rating_count": safe_get(detail, "rating", "count"),
        "description": clean_text(
            safe_get(detail, "details", "description_text") or
            safe_get(detail, "details", "raw_text")
        ),
        "images": [normalize_url(i) for i in (detail.get("images") or [prod.get("image_url")]) if i][:MAX_IMAGES],
    }

def unify_startech_product(row: Dict[str, str]) -> Dict[str, Any]:
    """Converts a simple Startech CSV row to our simple unified format."""
    price_disp = row.get("price", "N/A")
    price_val = None
    try:
        price_val = float(re.sub(r"[^\d.]", "", price_disp))
    except Exception:
        price_val = None

    return {
        "id": row.get("url"), # Use URL as ID
        "source": "Startech",
        "title": row.get("name", "Unknown Product"),
        "brand": None, # Startech scraper doesn't get brand
        "category": row.get("category", "unknown"),
        "url": normalize_url(row.get("url")),
        "price_display": price_disp,
        "price_value": price_val,
        "rating_average": None, # Startech scraper doesn't get rating
        "rating_count": None,
        "description": f"Status: {row.get('status', 'N/A')}", # Put status in desc
        "images": [], # Startech scraper doesn't get images
    }

# --- Corpus Writer ---

def product_to_markdown_block(p: Dict[str, Any]) -> str:
    """Writes the Unified Product to a Markdown block for the RAG app."""
    lines = []
    # Header for parsing
    lines.append(f"")
    
    lines.append(f"## {p['title']}  \n**DocID:** {p['id']}")
    lines.append("")
    
    meta = []
    meta.append(f"**Source:** {p['source']}") # NEW: Show the source
    if p.get("category"): meta.append(f"**Category:** {p['category']}")
    if p.get("brand"): meta.append(f"**Brand:** {p['brand']}")
    if p.get("url"): meta.append(f"**URL:** {p['url']}")
    lines.append("  \n".join(meta))
    lines.append("")

    price_bits = []
    if p.get("price_display"): price_bits.append(f"**Price:** {p['price_display']}")
    
    rating_bits = []
    if p.get("rating_average") is not None:
        rc = p.get("rating_count")
        rating_bits.append(f"**Rating:** {p['rating_average']}/5" + (f" ({rc} ratings)" if rc else ""))
    
    if price_bits or rating_bits:
        lines.append("  \n".join(price_bits + rating_bits))
        lines.append("")

    if p.get("description"):
        lines.append("**Description:**")
        lines.append(p["description"])
        lines.append("")
    
    if p.get("images"):
        lines.append("**Images (sample):**")
        for im in p["images"]:
            if im: lines.append(f"- {im}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("")
    return "\n".join(lines)

def product_to_jsonl_record(p: Dict[str, Any]) -> Dict[str, Any]:
    """Writes a unified record for JSONL."""
    text_lines = [f"{p['title']} (ID: {p['id']})"]
    if p.get("brand"): text_lines.append(f"Brand: {p['brand']}")
    if p.get("category"): text_lines.append(f"Category: {p['category']}")
    if p.get("price_display"): text_lines.append(f"Price: {p['price_display']}")
    if p.get("description"): text_lines.append("Description: " + p["description"])
    
    metadata = p.copy() # Just store everything in metadata
    return {"id": p["id"], "text": "\n".join(text_lines), "metadata": metadata}


# --- Main Execution ---
def main():
    seen_ids = set()
    total_written = 0
    categories = set()

    with MD_PATH.open("w", encoding="utf-8") as fmd, \
         JSONL_PATH.open("w", encoding="utf-8") as fjsonl:
        
        fmd.write("# Combined Daraz & Startech Product Corpus\n\n")
        
        # 1. Process Daraz Data
        print(f"Processing Daraz data from: {DARAZ_INPUT_ROOT}")
        daraz_files = sorted(DARAZ_INPUT_ROOT.glob("*/products.json"))
        for pfile in daraz_files:
            cat_dir = pfile.parent.name
            try:
                data = json.loads(pfile.read_text(encoding="utf-8"))
                if not isinstance(data, list): continue
            except Exception as e:
                print(f"[WARN] Failed to parse {pfile}: {e}")
                continue
                
            for raw in data:
                unified = unify_daraz_product(raw, cat_dir)
                uid = unified.get("id")
                if not uid or uid in seen_ids:
                    continue
                seen_ids.add(uid)
                categories.add(unified.get("category"))
                
                fmd.write(product_to_markdown_block(unified))
                fjsonl.write(json.dumps(product_to_jsonl_record(unified), ensure_ascii=False) + "\n")
                total_written += 1

        print(f"Processed {len(daraz_files)} Daraz files. Total products so far: {total_written}")

        # 2. Process Startech Data
        print(f"Processing Startech data from: {STARTECH_INPUT_FILE}")
        if not STARTECH_INPUT_FILE.exists():
            print(f"[WARN] Startech file not found: {STARTECH_INPUT_FILE}. Skipping.")
        else:
            try:
                with STARTECH_INPUT_FILE.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        unified = unify_startech_product(row)
                        uid = unified.get("id")
                        if not uid or uid in seen_ids:
                            continue
                        seen_ids.add(uid)
                        categories.add(unified.get("category"))
                        
                        fmd.write(product_to_markdown_block(unified))
                        fjsonl.write(json.dumps(product_to_jsonl_record(unified), ensure_ascii=False) + "\n")
                        total_written += 1
            except Exception as e:
                print(f"[WARN] Failed to process Startech CSV: {e}")

        # Final Summary
        fmd.write(f"\n> Summary: {total_written} products from {len(categories)} categories.\n")
        
        print(f"\n[OK] Markdown: {MD_PATH}")
        print(f"[OK] JSONL   : {JSONL_PATH}")
        print(f"[DONE] Wrote {total_written} unique products from {len(categories)} categories.")

if __name__ == "__main__":
    main()