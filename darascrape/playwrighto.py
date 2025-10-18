import re
from typing import Any, Dict, List, Optional, Tuple

from helpers import (
    normalize_url,
    parse_price_number
)
from fetcha import (
    compute_discount_pct,
)


# ======================= HTTP & Politeness =======================

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}

CAPTCHA_HINTS = [
    "are you human", "verify you are human", "unusual traffic",
    "complete the security check", "captcha", "cf-chl-cap",
    "bot detection", "challenge-form"
]

# ======================= Playwright render + EXTRAS =======================

async def render_html_with_playwright(url: str, timeout_ms: int = 26000) -> Tuple[str, Dict[str, Any]]:
    """
    Render with Playwright and return:
      - final HTML
      - extras:
        {
          "colors_all": [...],
          "rating_avg": float|None, "rating_count": int|None, "rating_raw": str|None,
          "seller": {"name":..., "link":..., "metrics": {...}} or {},
          "details_html": "<div class='pdp-product-detail'>...</div>" or None,
          "variants_by_color": [ {color, price{...}, images[], sizes[]} ]
        }
    """
    from playwright.async_api import async_playwright

    extras: Dict[str, Any] = {
        "colors_all": [],
        "rating_avg": None,
        "rating_count": None,
        "rating_raw": None,
        "seller": {},
        "details_html": None,
        "variants_by_color": [],
    }

    def _looks_like_captcha_text(txt: str) -> bool:
        low = txt.lower()
        return any(h in low for h in CAPTCHA_HINTS)

    def _compute_discount(price_text: Optional[str], orig_text: Optional[str], disc_text: Optional[str]) -> Optional[float]:
        return compute_discount_pct(price_text, orig_text, disc_text)

    async def _read_price_dom(page) -> Dict[str, Optional[Any]]:
        price_loc = page.locator(".pdp-product-price .pdp-price_type_normal, .pdp-product-price .pdp-price").first
        orig_loc  = page.locator(".pdp-product-price .pdp-price_type_deleted").first
        disc_loc  = page.locator(".pdp-product-price .pdp-product-price__discount, .pdp-price__discount").first
        price_text = (await price_loc.text_content() or "").strip() if await price_loc.count() > 0 else None
        orig_text  = (await orig_loc.text_content() or "").strip() if await orig_loc.count() > 0 else None
        disc_text  = (await disc_loc.text_content() or "").strip() if await disc_loc.count() > 0 else None
        return {
            "display": price_text,
            "value": parse_price_number(price_text),
            "original_display": orig_text,
            "original_value": parse_price_number(orig_text),
            "discount_display": disc_text,
            "discount_percent": _compute_discount(price_text, orig_text, disc_text),
        }

    async def _read_images_dom(page, base_url: str) -> List[str]:
        imgs = []
        for loc in ["#module_item_gallery_1 img", ".item-gallery img", ".pdp-block__gallery img"]:
            nodes = page.locator(loc)
            n = await nodes.count()
            for i in range(n):
                src = await nodes.nth(i).get_attribute("src")
                u = normalize_url(src, base_url)
                if u and u not in imgs:
                    imgs.append(u)
        return imgs

    async def _read_sizes_dom(page) -> List[str]:
        out = []
        size_section = page.locator(".sku-prop:has(h6.section-title:has-text('Size'))")
        if await size_section.count() == 0:
            return out
        chips = size_section.locator(".sku-prop-content .sku-variable-size, .sku-prop-content .sku-variable-size-selected")
        n = await chips.count()
        for i in range(n):
            t = (await chips.nth(i).text_content() or "").strip()
            if t and t not in out:
                out.append(t)
        return out

    # Use the same UA and language as requests, no stealth flags or automation bypasses.
    ua = DEFAULT_HEADERS["User-Agent"]
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,  # keep headless; no stealth flags or evasion
            args=["--no-sandbox"]
        )
        context = await browser.new_context(
            user_agent=ua,
            viewport={"width": 1366, "height": 900},
            java_script_enabled=True,
            extra_http_headers={
                "Accept-Language": DEFAULT_HEADERS["Accept-Language"],
                "Accept": DEFAULT_HEADERS["Accept"],
            },
        )
        page = await context.new_page()

        # Navigate with timeout and early bot-wall detection
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            # quick bot-wall sniff
            body_txt = (await page.content()) or ""
            if _looks_like_captcha_text(body_txt):
                raise PermissionError("Suspected bot wall in Playwright; stopping politely.")
        except Exception:
            pass

        # Progressive scroll to trigger lazy modules (modest pacing)
        try:
            for _ in range(6):
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight/6)")
                await page.wait_for_timeout(300)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(700)
        except Exception:
            pass

        # --- Enumerate colors + capture per-color price/images/sizes
        try:
            color_section = page.locator(".sku-prop:has(h6.section-title:has-text('Color Family'))")
            if await color_section.count() == 0:
                color_section = page.locator(".sku-prop:has(h6.section-title:has-text('Color'))")

            if await color_section.count() > 0:
                name_loc = color_section.locator(".sku-prop-content-header .sku-name").first
                swatches = color_section.locator(".sku-prop-content .sku-variable-img-wrap, .sku-prop-content .sku-variable-img-wrap-selected")
                n = await swatches.count()
                seen = set()

                for i in range(n):
                    sw = swatches.nth(i)
                    try:
                        await sw.scroll_into_view_if_needed()
                        await sw.click(force=True)
                        await page.wait_for_timeout(400)  # allow DOM to settle
                    except Exception:
                        continue

                    color_name = None
                    try:
                        if await name_loc.count() > 0:
                            color_name = ((await name_loc.text_content()) or "").strip()
                    except Exception:
                        pass
                    if not color_name:
                        alt = await sw.locator("img").first.get_attribute("alt")
                        color_name = (alt or "").strip() or f"Color #{i+1}"

                    key = color_name.lower()
                    if key in seen:
                        continue
                    seen.add(key)

                    price_dict = await _read_price_dom(page)
                    images = await _read_images_dom(page, url)
                    sizes = await _read_sizes_dom(page)

                    extras["colors_all"].append(color_name)
                    extras["variants_by_color"].append({
                        "color": color_name,
                        "price": price_dict,
                        "images": images,
                        "sizes": sizes,
                    })
        except Exception:
            pass

        # --- Rating (scroll into view)
        try:
            review_section = page.locator("#module_product_review")
            if await review_section.count() > 0:
                await review_section.scroll_into_view_if_needed()
                await page.wait_for_timeout(500)

            avg_loc = page.locator(".pdp-mod-review .score .score-average").first
            if await avg_loc.count() > 0:
                avg_text = (await avg_loc.text_content() or "").strip()
                if avg_text:
                    extras["rating_avg"] = float(avg_text)

            cnt_loc = page.locator(".pdp-mod-review .summary .count, .pdp-review-summary__link").first
            if await cnt_loc.count() > 0:
                raw = (await cnt_loc.text_content() or "").strip()
                extras["rating_raw"] = raw
                m = re.search(r"([\d,]+)\s*Ratings?", raw, re.I) or re.search(r"Ratings?\s*([\d,]+)", raw, re.I)
                if m:
                    extras["rating_count"] = int(m.group(1).replace(",", ""))
                elif "No Ratings" in raw:
                    extras["rating_count"] = 0
                    if extras["rating_avg"] is None:
                        extras["rating_avg"] = 0.0
        except Exception:
            pass

        # --- Seller
        try:
            seller_module = page.locator("#module_seller_info")
            if await seller_module.count() > 0:
                await seller_module.scroll_into_view_if_needed()
                await page.wait_for_timeout(500)

                name_a = seller_module.locator(".seller-name__detail-name, a[href*='/shop/']").first
                name = (await name_a.text_content() or "").strip() if await name_a.count() > 0 else None
                href = await name_a.get_attribute("href") if await name_a.count() > 0 else None
                link = normalize_url(href, url) if href else None

                metrics: Dict[str, Optional[str]] = {}
                rows = seller_module.locator(".pdp-seller-info-pc .info-content")
                n = await rows.count()
                for i in range(n):
                    row = rows.nth(i)
                    title = (await row.locator(".seller-info-title").text_content() or "").strip()
                    value = (await row.locator(".seller-info-value").text_content() or "").strip()
                    if title:
                        metrics[title] = value

                if name or link or metrics:
                    extras["seller"] = {"name": name or None, "link": link, "metrics": metrics}
        except Exception:
            pass

        # --- Details
        try:
            details_module = page.locator("#module_product_detail")
            if await details_module.count() > 0:
                await details_module.scroll_into_view_if_needed()
                await page.wait_for_timeout(500)
                more_btn = details_module.locator("text=/read more|see more|view more/i").first
                if await more_btn.count() > 0:
                    try:
                        await more_btn.click(force=True)
                        await page.wait_for_timeout(300)
                    except Exception:
                        pass
                detail_root = details_module.locator(".pdp-product-detail").first
                if await detail_root.count() == 0:
                    detail_root = details_module
                extras["details_html"] = await detail_root.inner_html()
        except Exception:
            pass

        html = await page.content()
        await context.close()
        await browser.close()
        return html, extras