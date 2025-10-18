import asyncio
import logging
from typing import Any, Dict

from fetcha import (
    parse_all,
)
from playwrighto import render_html_with_playwright
from mergo import _dedup_case_insensitive, _parse_details_fragment
from politea import PoliteSession


# Shared polite session
HTTP = PoliteSession(min_delay_s=1.5, max_retries=4, timeout=25)


# ======================= Logging =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOG = logging.getLogger("pdp-scraper")


# ======================= Orchestrator =======================
def fetch_html(url: str, timeout: int = 25) -> str:
    # Wrap polite session
    return HTTP.get(url)

def scrape_product(url: str, render_if_missing: bool = True, debug: bool = False) -> Dict[str, Any]:
    if debug:
        LOG.setLevel(logging.DEBUG)

    # 1) Static pass (polite HTTP)
    LOG.info("Fetching (static): %s", url)
    html = fetch_html(url)
    data = parse_all(html, url)

    # 2) Decide if rendering is needed
    need_render = False
    if not data["price"]["display"] or not data["seller"]["name"] or not data["details"]["raw_text"]:
        need_render = True
    if data["rating"]["average"] is None or len(data["colors"]) <= 1:
        need_render = True

    # 3) Render, with polite pacing and no stealth/evasion
    if render_if_missing and need_render:
        LOG.info("Static parse incomplete; rendering with Playwright (polite scrolls + bounded waits).")
        try:
            html2, extras = asyncio.run(render_html_with_playwright(url))
            data2 = parse_all(html2, url)

            def prefer(a, b):
                return b if (a in [None, "", [], {}]) else a

            # Price
            data["price"] = {
                "display": prefer(data["price"]["display"], data2["price"]["display"]),
                "value": prefer(data["price"]["value"], data2["price"]["value"]),
                "original_display": prefer(data["price"]["original_display"], data2["price"]["original_display"]),
                "original_value": prefer(data["price"]["original_value"], data2["price"]["original_value"]),
                "discount_display": prefer(data["price"]["discount_display"], data2["price"]["discount_display"]),
                "discount_percent": prefer(data["price"]["discount_percent"], data2["price"]["discount_percent"]),
            }

            # Simple merges
            for key in ["name", "brand", "images", "sizes", "delivery_options",
                        "return_and_warranty", "seller", "details"]:
                if not data[key]:
                    data[key] = data2[key]

            # Colors
            colors_union = (data.get("colors") or []) + (data2.get("colors") or []) + (extras.get("colors_all") or [])
            data["colors"] = _dedup_case_insensitive(colors_union)

            # Rating
            if extras.get("rating_avg") is not None:
                data["rating"]["average"] = extras["rating_avg"]
            if extras.get("rating_count") is not None:
                data["rating"]["count"] = extras["rating_count"]
            if extras.get("rating_raw"):
                data["rating"]["raw"] = extras["rating_raw"]
            if data["rating"]["average"] is None and data2["rating"]["average"] is not None:
                data["rating"]["average"] = data2["rating"]["average"]
            if data["rating"]["count"] is None and data2["rating"]["count"] is not None:
                data["rating"]["count"] = data2["rating"]["count"]
            if not data["rating"]["raw"] and data2["rating"]["raw"]:
                data["rating"]["raw"] = data2["rating"]["raw"]

            # Seller
            if (not data["seller"]["name"]) and extras.get("seller"):
                data["seller"] = extras["seller"]

            # Details
            if not data["details"].get("raw_text") and extras.get("details_html"):
                parsed = _parse_details_fragment(extras["details_html"])
                for k, v in parsed.items():
                    if (not data["details"].get(k)) and v:
                        data["details"][k] = v

            # Variants (per-color price/images/sizes)
            if extras.get("variants_by_color"):
                data["variants"] = extras["variants_by_color"]

        except PermissionError as pe:
            LOG.warning(str(pe))
        except Exception as e:
            LOG.warning("Render fallback failed: %s", e)

    return data