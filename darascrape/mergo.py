from typing import Any, Dict, List

from helpers import (
    text_or_none, clean_spaces,
)

from bs4 import BeautifulSoup


# ======================= Merge helpers =======================
def _dedup_case_insensitive(values: List[str]) -> List[str]:
    seen, out = set(), []
    for v in values:
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out

def _parse_details_fragment(fragment_html: str) -> Dict[str, Any]:
    out = {
        "highlights": [],
        "description_text": None,
        "description_html": None,
        "specifications": [],
        "whats_in_the_box": None,
        "raw_text": None,
    }
    if not fragment_html:
        return out
    frag = BeautifulSoup(fragment_html, "lxml")

    for p in frag.select(".pdp-product-highlights p, .pdp-product-highlights li"):
        t = clean_spaces(p.get_text())
        if t:
            out["highlights"].append(t)

    desc = frag.select_one(".detail-content")
    if desc:
        out["description_text"] = clean_spaces(desc.get_text(" "))
        out["description_html"] = str(desc)

    for li in frag.select(".pdp-mod-specification .specification-keys .key-li"):
        key = clean_spaces(text_or_none(li.select_one(".key-title")))
        val = clean_spaces(text_or_none(li.select_one(".key-value")))
        if key or val:
            out["specifications"].append({"key": key, "value": val})

    box_html = frag.select_one(".pdp-mod-specification .box-content .box-content-html")
    if box_html:
        out["whats_in_the_box"] = clean_spaces(box_html.get_text(" "))

    out["raw_text"] = clean_spaces(frag.get_text(" "))
    return out