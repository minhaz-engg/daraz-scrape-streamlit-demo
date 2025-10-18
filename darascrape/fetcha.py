import re
from typing import Any, Dict, List, Optional, Tuple

from helpers import (
    normalize_url, soupify, text_or_none, clean_spaces,
    parse_price_number, try_jsonld, get_product_jsonld
)


from bs4 import BeautifulSoup, Tag



# ======================= extractors =======================

def extract_title(soup: BeautifulSoup, pjson: Optional[Dict[str, Any]]) -> Optional[str]:
    el = soup.select_one("h1.pdp-mod-product-badge-title")
    if el:
        return clean_spaces(el.get_text())
    if pjson and pjson.get("name"):
        return clean_spaces(pjson["name"])
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return clean_spaces(og["content"])
    return None

def extract_brand(soup: BeautifulSoup, pjson: Optional[Dict[str, Any]]) -> Optional[str]:
    a = soup.select_one(".pdp-product-brand a.pdp-product-brand__brand-link")
    if a:
        return clean_spaces(a.get_text())
    for li in soup.select(".pdp-mod-specification .specification-keys .key-li"):
        k = text_or_none(li.select_one(".key-title"))
        v = text_or_none(li.select_one(".key-value"))
        if k and "brand" in k.lower():
            return clean_spaces(v)
    if pjson:
        b = pjson.get("brand")
        if isinstance(b, dict):
            n = b.get("name")
            if n:
                return clean_spaces(n)
        elif isinstance(b, str):
            return clean_spaces(b)
    return None

def extract_prices(soup: BeautifulSoup, pjson: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float], Optional[float]]:
    price_el = soup.select_one(".pdp-product-price [class*='pdp-price_type_normal'], .pdp-product-price .pdp-price")
    orig_el  = soup.select_one(".pdp-product-price [class*='pdp-price_type_deleted']")
    disc_el  = soup.select_one(".pdp-product-price .pdp-product-price__discount, .pdp-price__discount, .pdp-product-price__discount")

    price_text = text_or_none(price_el)
    orig_text  = text_or_none(orig_el)
    disc_text  = text_or_none(disc_el)

    price_num = parse_price_number(price_text)
    orig_num  = parse_price_number(orig_text)

    if not price_text and pjson:
        offers = pjson.get("offers")
        if isinstance(offers, dict) and offers.get("price") is not None:
            price_text = str(offers.get("price"))
            price_num  = parse_price_number(price_text)

    return price_text, orig_text, disc_text, price_num, orig_num

def compute_discount_pct(price_text: Optional[str], orig_text: Optional[str], disc_text: Optional[str]) -> Optional[float]:
    if disc_text:
        m = re.search(r"-?\s*([0-9]+(?:\.[0-9]+)?)\s*%", disc_text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    p = parse_price_number(price_text)
    o = parse_price_number(orig_text)
    if p and o and o > 0 and p <= o:
        return round((1 - p / o) * 100.0, 2)
    return None

def extract_images(soup: BeautifulSoup, base_url: str, pjson: Optional[Dict[str, Any]]) -> List[str]:
    urls: List[str] = []
    gallery = soup.select_one("#module_item_gallery_1, .pdp-block__gallery, .item-gallery")
    if gallery:
        for img in gallery.select("img"):
            u = normalize_url(img.get("src"), base_url)
            if u:
                urls.append(u)
    if pjson:
        imgs = pjson.get("image")
        if isinstance(imgs, str):
            u = normalize_url(imgs, base_url)
            if u:
                urls.append(u)
        elif isinstance(imgs, list):
            for i in imgs:
                u = normalize_url(str(i), base_url)
                if u:
                    urls.append(u)
    seen, dedup = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup

def extract_rating(soup: BeautifulSoup, pjson: Optional[Dict[str, Any]]) -> Dict[str, Optional[Any]]:
    if pjson and isinstance(pjson.get("aggregateRating"), dict):
        ar = pjson["aggregateRating"]
        avg = float(ar.get("ratingValue")) if ar.get("ratingValue") not in (None, "") else None
        cnt = int(str(ar.get("ratingCount")).replace(",", "")) if ar.get("ratingCount") not in (None, "") else None
        return {"average": avg, "count": cnt, "raw": None}

    avg, cnt, raw = None, None, None
    avg_el = soup.select_one(".pdp-mod-review .score .score-average")
    if avg_el and avg_el.text.strip():
        try:
            avg = float(avg_el.text.strip())
        except Exception:
            pass

    cnt_el = soup.select_one(".pdp-mod-review .summary .count, .pdp-review-summary__link, .pdp-review-summary .pdp-review-summary__link")
    if cnt_el:
        raw = cnt_el.get_text(" ", strip=True)
        m = re.search(r"([\d,]+)\s*Ratings?", raw, re.I) or re.search(r"Ratings?\s*([\d,]+)", raw, re.I)
        if m:
            try:
                cnt = int(m.group(1).replace(",", ""))
            except Exception:
                pass
        elif "No Ratings" in raw:
            cnt = 0
            if avg is None:
                avg = 0.0
    return {"average": avg, "count": cnt, "raw": raw}

def find_section_by_h6(soup: BeautifulSoup, title_contains: str) -> Optional[Tag]:
    for sec in soup.select("div.sku-prop, div.pdp-mod-product-info-section"):
        h6 = sec.select_one("h6.section-title")
        if h6 and title_contains.lower() in h6.get_text(strip=True).lower():
            return sec
    for h6 in soup.select("h6.section-title"):
        if title_contains.lower() in h6.get_text(strip=True).lower():
            return h6.find_parent("div")
    return None

def extract_colors(soup: BeautifulSoup) -> List[str]:
    out: List[str] = []
    sec = find_section_by_h6(soup, "Color Family")
    if sec:
        header = sec.select_one(".sku-prop-content-header .sku-name")
        if header:
            t = clean_spaces(header.get_text())
            if t:
                out.append(t)
        for sw in sec.select(".sku-prop-content img"):
            alt = (sw.get("alt") or "").strip()
            if alt:
                out.append(alt)
    for li in soup.select(".pdp-mod-specification .specification-keys .key-li"):
        k = text_or_none(li.select_one(".key-title"))
        v = text_or_none(li.select_one(".key-value"))
        if k and "color" in k.lower():
            vals = [clean_spaces(x) for x in re.split(r"[,/|]", v or "") if clean_spaces(x)]
            out.extend(vals)
            break
    seen, dedup = set(), []
    for c in out:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            dedup.append(c)
    return dedup

def extract_sizes(soup: BeautifulSoup) -> List[str]:
    out: List[str] = []
    sec = find_section_by_h6(soup, "Size")
    if not sec:
        return out
    for sp in sec.select(".sku-variable-size, .sku-variable-size-selected"):
        t = clean_spaces(sp.get_text())
        if t:
            out.append(t)
    seen, dedup = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup

def extract_delivery(soup: BeautifulSoup) -> List[Dict[str, Optional[str]]]:
    results: List[Dict[str, Optional[str]]] = []
    for body in soup.select("#module_seller_delivery .delivery__option .delivery-option-item__body"):
        title = text_or_none(body.select_one(".delivery-option-item__title"))
        time_ = text_or_none(body.select_one(".delivery-option-item__time"))
        fee   = text_or_none(body.select_one(".delivery-option-item__shipping-fee"))
        results.append({"title": title, "time": time_, "fee": fee})
    return results

def extract_return_warranty(soup: BeautifulSoup) -> List[str]:
    out: List[str] = []
    for t in soup.select("#module_seller_warranty .warranty__option-item .delivery-option-item__title"):
        val = clean_spaces(t.get_text())
        if val:
            out.append(val)
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup

def extract_seller(soup: BeautifulSoup, base_url: str) -> Dict[str, Optional[Any]]:
    name_el = soup.select_one("#module_seller_info .seller-name__detail-name, #module_seller_info a[href*='/shop/']")
    name = text_or_none(name_el)
    link = normalize_url(name_el.get("href") if name_el else None, base_url) if name_el else None
    metrics: Dict[str, Optional[str]] = {}
    for info in soup.select("#module_seller_info .pdp-seller-info-pc .info-content"):
        k = text_or_none(info.select_one(".seller-info-title"))
        v = text_or_none(info.select_one(".seller-info-value"))
        if k:
            metrics[k] = v
    return {"name": name, "link": link, "metrics": metrics}

def extract_details(soup: BeautifulSoup, pjson: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "highlights": [],
        "description_text": None,
        "description_html": None,
        "specifications": [],
        "whats_in_the_box": None,
        "raw_text": None,
    }
    root = soup.select_one("#module_product_detail .pdp-product-detail, .pdp-product-detail")
    if root:
        for p in root.select(".pdp-product-highlights p, .pdp-product-highlights li"):
            t = clean_spaces(p.get_text())
            if t:
                out["highlights"].append(t)

        desc = root.select_one(".detail-content")
        if desc:
            out["description_text"] = clean_spaces(desc.get_text(" "))
            out["description_html"] = str(desc)

        for li in root.select(".pdp-mod-specification .specification-keys .key-li"):
            key = clean_spaces(text_or_none(li.select_one(".key-title")))
            val = clean_spaces(text_or_none(li.select_one(".key-value")))
            if key or val:
                out["specifications"].append({"key": key, "value": val})

        box_html = root.select_one(".pdp-mod-specification .box-content .box-content-html")
        if box_html:
            out["whats_in_the_box"] = clean_spaces(box_html.get_text(" "))

        out["raw_text"] = clean_spaces(root.get_text(" "))
    if (not out["raw_text"]) and pjson and isinstance(pjson.get("description"), str):
        desc = clean_spaces(pjson["description"])
        out["description_text"] = out["description_text"] or desc
        out["raw_text"] = out["raw_text"] or desc
    return out

def parse_all(html: str, url: str) -> Dict[str, Any]:
    soup = soupify(html)
    jsonlds = try_jsonld(soup)
    pjson = get_product_jsonld(jsonlds)

    name = extract_title(soup, pjson)
    brand = extract_brand(soup, pjson)
    price_text, orig_text, disc_text, price_num, orig_num = extract_prices(soup, pjson)
    discount_pct = compute_discount_pct(price_text, orig_text, disc_text)

    images = extract_images(soup, url, pjson)
    rating = extract_rating(soup, pjson)
    colors = extract_colors(soup)
    sizes = extract_sizes(soup)
    delivery = extract_delivery(soup)
    retwar = extract_return_warranty(soup)
    seller = extract_seller(soup, url)
    details = extract_details(soup, pjson)

    return {
        "url": url,
        "name": name,
        "brand": brand,
        "price": {
            "display": price_text,
            "value": price_num,
            "original_display": orig_text,
            "original_value": orig_num,
            "discount_display": disc_text,
            "discount_percent": discount_pct,
        },
        "rating": rating,
        "images": images,
        "colors": colors,
        "sizes": sizes,
        "delivery_options": delivery,
        "return_and_warranty": retwar,
        "seller": seller,
        "details": details,
        "variants": [],  # filled by Playwright extras
    }