# FILE: scraper_startech_modular.py
import asyncio
import os
import logging
import time
from urllib.parse import urljoin, urlparse
import csv
from crawl4ai import AsyncWebCrawler
from firecrawl import Firecrawl
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
import nest_asyncio

# Patch the running event loop
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Constants ---
MAX_WORKERS = 3
BATCH_SIZE = 50
REQUEST_DELAY = 2
CSV_FILE = "startech_products.csv" # This will be created in the root
FIRECRAWL_API_KEY = "fc-ee108318c6624602a12257fd72388cf1" # Use your key
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)


# (Keep all your helper functions: init_csv, save_products_csv, fetch_html,
#  scrape_category, process_category_chunk)

# PASTE ALL YOUR HELPER FUNCTIONS HERE...
# ... (from line 28 to 157 in your original code) ...
# FOR BREVITY, I AM OMITTING THEM, BUT YOU MUST PASTE THEM


# MODIFIED: This function is now the main entry point
async def run_scraper(category_links: list[str]):
    init_csv(CSV_FILE)
    
    categories = category_links
    
    if not categories:
        logging.error("[Startech] No categories provided to scrape.")
        return

    logging.info(f"✅ [Startech] Starting scraper for {len(categories)} provided category links.")

    all_products = []
    processed = 0
    total_saved = 0
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    chunk_size = MAX_WORKERS * 2
    for i in range(0, len(categories), chunk_size):
        category_chunk = categories[i:i + chunk_size]
        try:
            # This loop will be interrupted if the 5-min timer expires
            products = await process_category_chunk(category_chunk, semaphore)
            if products:
                all_products.extend(products)
                processed += len(category_chunk)
                if len(all_products) >= BATCH_SIZE:
                    saved = save_products_csv(all_products, CSV_FILE)
                    total_saved += saved
                    all_products = []
                logging.info(f"[Startech] Progress: {processed}/{len(categories)} categories processed")
        except asyncio.CancelledError:
            logging.warning(f"[Startech] Chunk processing cancelled (timeout).")
            break # Exit the loop if cancelled
        except Exception as e:
            logging.error(f"[Startech] Error processing chunk: {e}")
        await asyncio.sleep(1)

    if all_products:
        total_saved += save_products_csv(all_products, CSV_FILE)

    logging.info(f"✅ [Startech] Scraping completed! Total products saved: {total_saved}")

# NO if __name__ == "__main__": block