# FILE: scraper_daraz_modular.py
import os
import re
import json
import math
import time
import random
import asyncio
from typing import List, Dict, Tuple, Iterable, Optional, Any
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlResult,
    CrawlerRunConfig,
    GeolocationConfig,
    LLMConfig,
    JsonCssExtractionStrategy,
)
from dotenv import load_dotenv

# (Keep all your helper functions: load_dotenv, __cur_dir__, RESULT_DIR, SCHEMA_FILE,
#  all Tunables, folder_name_from_url, ensure_category_dir, atomic_write_json,
#  load_json_if_exists, set_query_param, build_page_url, parse_total_items_from_html,
#  detect_total_pages, get_product_id, dedupe_products, BLOCK_KEYWORDS, BlockedError,
#  looks_blocked, load_or_generate_schema, normalize_url,
#  extract_detail_url_from_card, crawl_once, fetch_until_new,
#  _fetch_detail_for_product, enrich_products_with_details)

# PASTE ALL YOUR HELPER FUNCTIONS HERE...
# ... (from line 24 to 534 in your original code) ...
# FOR BREVITY, I AM OMITTING THEM, BUT YOU MUST PASTE THEM

# ----- Main flow -----
# RENAMED from demo_css_structured_extraction_no_schema
async def scrape_daraz_category(link: str):
    print(f"\n[Daraz] === Starting Category: {link} ===")
    # Prepare per-category paths
    paths = ensure_category_dir(link)
    category_dir = paths["dir"]
    schema_path = paths["schema"]
    products_path = paths["products"]
    pages_index_path = paths["pages_index"]
    detail_cache_path = paths["details_cache"]

    # Load detail cache
    detail_cache: Dict[str, Any] = load_json_if_exists(detail_cache_path, default={})
    if not isinstance(detail_cache, dict):
        detail_cache = {}

    sample_html = """
    <div class="Bm3ON" data-qa-locator="product-item">
        ... (Your sample HTML) ...
    </div>
    """
    schema = await load_or_generate_schema(link, sample_html)
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode=CacheMode.DISABLED,
        geolocation=GeolocationConfig(latitude=23.8103, longitude=90.4125), # BD Geo
        prettiify=True,
        wait_for_images=True,
        delay_before_return_html=True,
        mean_delay=0.7,
        scroll_delay=0.6,
        verbose=True,
    )

    # Load existing products for THIS category folder
    existing_products_list = load_json_if_exists(products_path, default=[])
    existing_products_list = existing_products_list if isinstance(existing_products_list, list) else []
    all_products_by_id: Dict[str, dict] = {pid: p for p in existing_products_list if (pid := get_product_id(p))}
    known_ids = set(all_products_by_id.keys())
    pages_index = load_json_if_exists(pages_index_path, default={})
    browser_config = BrowserConfig()
    total_new_added = 0

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            total_pages = await detect_total_pages(crawler, link)
            
            # REMOVED TEST LIMIT
            # if total_pages >= 1:
            #     print(f"Total pages {total_pages} > 1, limiting to 1 for testing.")
            #     total_pages = 1
            
            urls = [build_page_url(link, page) for page in range(1, total_pages + 1)]

            for page_idx, base_url in enumerate(urls, start=1):
                # This loop will be interrupted if the 5-min timer expires
                products, ids, attempts_used = await fetch_until_new(
                    crawler=crawler,
                    base_url=base_url,
                    config=config,
                    known_ids=known_ids,
                    page_idx=page_idx,
                )
                
                # ... (rest of your loop logic from line 630 to 690) ...
                # ... (saving products, enriching details, etc.) ...
                
                if not products:
                    print(f"⚠️ [Daraz] Page {page_idx}: no products found after {attempts_used} attempts.")
                    await asyncio.sleep(random.uniform(*PAGE_PAUSE_RANGE) + 1.5)
                    continue

                page_new_products: List[dict] = []
                for p in products:
                    pid = get_product_id(p)
                    if pid and pid not in known_ids:
                        known_ids.add(pid)
                        all_products_by_id[pid] = p
                        page_new_products.append(p)
                
                pages_index[str(page_idx)] = ids
                to_save_page = page_new_products if SAVE_ONLY_UNIQUE else products

                if ENRICH_WITH_DETAIL and to_save_page:
                    await enrich_products_with_details(
                        items=to_save_page,
                        base_url=base_url,
                        detail_cache=detail_cache,
                        concurrency=DETAIL_CONCURRENCY,
                    )
                    for p in to_save_page:
                        pid = get_product_id(p)
                        if pid:
                            all_products_by_id[pid] = p
                    atomic_write_json(detail_cache_path, detail_cache)

                if to_save_page:
                    atomic_write_json(category_dir / f"page_{page_idx}.json", to_save_page)
                    total_new_added += len(page_new_products)
                    print(f"✅ [Daraz] Page {page_idx}: saved {len(to_save_page)} items.")
                
                await asyncio.sleep(random.uniform(*PAGE_PAUSE_RANGE))

    except asyncio.CancelledError:
        print(f"🚫 [Daraz] Task for {link} was cancelled (timeout).")
    except Exception as e:
        print(f"❌ [Daraz] Scraper for {link} failed: {e}")
    finally:
        # Save whatever we got before being cancelled
        all_products_unique = list(all_products_by_id.values())
        atomic_write_json(products_path, all_products_unique)
        atomic_write_json(pages_index_path, pages_index)
        print(f"✅ [Daraz] Finished category {link}. Total unique: {len(all_products_unique)}")

# NEW main function to run multiple categories concurrently
async def main(category_links: List[str]):
    print(f"=== [Daraz] Starting scrape for {len(category_links)} categories ===")
    tasks = [scrape_daraz_category(link) for link in category_links]
    await asyncio.gather(*tasks)
    print("=== [Daraz] All tasks complete ===")

# NO if __name__ == "__main__": block