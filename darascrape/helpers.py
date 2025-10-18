import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

# ======================= HTML helpers =======================

def normalize_url(src: Optional[str], base: str) -> Optional[str]:
    if not src:
        return None
    s = src.strip()
    if not s:
        return None
    if s.startswith("data:"):
        return None
    if s.startswith("//"):
        return "https:" + s
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return urljoin(base, s)

def soupify(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")

def text_or_none(el: Optional[Tag]) -> Optional[str]:
    if not el:
        return None
    t = el.get_text(" ", strip=True)
    return t if t else None

def clean_spaces(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r"\s+", " ", s).strip()

def parse_price_number(price_text: Optional[str]) -> Optional[float]:
    if not price_text:
        return None
    raw = re.sub(r"[^\d.,]", "", price_text)
    if not raw:
        return None
    # normalize "1,299.00" or "1,299"
    raw = raw.replace(",", "")
    try:
        return float(raw)
    except Exception:
        return None

def try_jsonld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    out = []
    for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
        txt = sc.string
        if not txt:
            continue
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                out.append(data)
            elif isinstance(data, list):
                out.extend([d for d in data if isinstance(d, dict)])
        except Exception:
            continue
    return out

def get_product_jsonld(jsonlds: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for d in jsonlds:
        if d.get("@type") == "Product":
            return d
        g = d.get("@graph")
        if isinstance(g, list):
            for x in g:
                if isinstance(x, dict) and x.get("@type") == "Product":
                    return x
    return None