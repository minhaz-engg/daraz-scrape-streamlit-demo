import asyncio
import os
import logging
import time
from urllib.parse import urljoin, urlparse
import os
import csv
import logging
import asyncio
import csv

from crawl4ai import AsyncWebCrawler
from firecrawl import Firecrawl
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
import nest_asyncio

# Patch the running event loop (needed for Colab/Jupyter)
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Constants ---
MAX_WORKERS = 3
BATCH_SIZE = 50
REQUEST_DELAY = 2
CSV_FILE = "startech_products.csv"

# Provide your Firecrawl API key if using their cloud service
FIRECRAWL_API_KEY = "fc-ee108318c6624602a12257fd72388cf1"  # or use os.getenv("FIRECRAWL_API_KEY") if you prefer environment variables
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)


# === CSV Setup ===
def init_csv(file_path):
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "name", "price", "status", "url"])
        logging.info(f"Initialized new CSV file: {file_path}")
    else:
        logging.info(f"Appending to existing CSV file: {file_path}")


def save_products_csv(products, file_path):
    if not products:
        return 0
    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for product in products:
                writer.writerow([
                    product.get('category', ''),
                    product.get('name', ''),
                    product.get('price', ''),
                    product.get('status', ''),
                    product.get('url', '')
                ])
        logging.info(f"Saved {len(products)} products to CSV.")
        return len(products)
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")
        return 0


# === Web Scraping Logic ===
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_html(url: str) -> str:
    await asyncio.sleep(REQUEST_DELAY)
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, use_playwright=True)
            if result and result.html:
                return result.html
    except Exception as e:
        logging.warning(f"Crawl4AI failed for {url}: {e}")

    try:
        result = firecrawl.crawl(url)
        if isinstance(result, dict) and "html" in result:
            return result["html"]
        return result
    except Exception as e:
        logging.error(f"FireCrawl failed for {url}: {e}")
        raise

# -----------------------------------------------------------------
# REMOVED discover_categories function as requested
# -----------------------------------------------------------------


async def scrape_category(url: str):
    logging.info(f"==== Scraping category page: {url}")
    html = await fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    products = []
    # Try to get a more specific category name from the URL path
    try:
        category = urlparse(url).path.strip("/").split('/')[-1]
    except Exception:
        category = urlparse(url).path.strip("/") # Fallback
        

    for product in soup.select(".p-item, .product-layout"):
        try:
            title_elem = product.select_one(".p-item-name, .product-name, h4 a")
            price_elem = product.select_one(".p-item-price, .price-new, .price")
            status_elem = product.select_one(".p-item-stock, .stock-status, .status")
            url_elem = product.select_one("h4 a, .p-item-name a, .product-name a")

            if not title_elem:
                continue

            product_url = urljoin(url, url_elem["href"]) if url_elem else None
            if not product_url:
                continue

            title = title_elem.text.strip()
            price = price_elem.text.strip() if price_elem else "N/A"
            status = status_elem.text.strip() if status_elem else "N/A"

            product_data = {
                "url": product_url,
                "category": category,
                "name": title,  # <-- FIXED: Changed from 'title' to 'name' to match CSV
                "price": price,
                "status": status,
                "scraped_from": url
            }

            products.append(product_data)

        except Exception as e:
            logging.error(f"Error extracting product: {e}")
            continue

    logging.info(f"Found {len(products)} products in {category}")
    return products


async def process_category_chunk(chunk, semaphore):
    processed_products = []
    for category in chunk:
        async with semaphore:
            products = await scrape_category(category)
            processed_products.extend(products)
    return processed_products


# MODIFIED: This function now takes your list of categories as an argument
async def run_scraper(category_links: list[str]):
    init_csv(CSV_FILE)
    
    # We no longer discover categories, we use the list you provided
    categories = category_links
    
    if not categories:
        logging.error("No categories provided to scrape. Please check 'my_target_categories' at the bottom.")
        return

    logging.info(f"✅ Starting scraper for {len(categories)} provided category links.")

    all_products = []
    processed = 0
    total_saved = 0
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    chunk_size = MAX_WORKERS * 2
    for i in range(0, len(categories), chunk_size):
        category_chunk = categories[i:i + chunk_size]
        try:
            products = await process_category_chunk(category_chunk, semaphore)
            if products:
                all_products.extend(products)
                processed += len(category_chunk)
                if len(all_products) >= BATCH_SIZE:
                    saved = save_products_csv(all_products, CSV_FILE)
                    total_saved += saved
                    all_products = []
                logging.info(f"Progress: {processed}/{len(categories)} categories processed, "
                             f"{total_saved} products saved.")
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
        await asyncio.sleep(1)

    if all_products:
        total_saved += save_products_csv(all_products, CSV_FILE)

    logging.info(f"✅ Scraping completed! Total products saved: {total_saved}")


# For Colab/Jupyter, simply call:
if __name__ == "__main__":
    
    # -----------------------------------------------------------------
    # === ⭐️ EDIT THIS LIST WITH YOUR CATEGORY LINKS ===
    # -----------------------------------------------------------------
    my_target_categories = [
        "https://www.startech.com.bd/laptop-notebook/laptop",
        "https://www.startech.com.bd/component/processor",
        "https://www.startech.com.bd/component/graphics-card",
        "https://www.startech.com.bd/component/motherboard",
        # Add as many category links as you want here
    ]
    # -----------------------------------------------------------------
    
    try:
        # We pass your list of links directly into the scraper
        asyncio.run(run_scraper(my_target_categories))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")