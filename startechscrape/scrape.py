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

ROOT_URL = "https://www.startech.com.bd/"

MAX_WORKERS = 3
BATCH_SIZE = 50
REQUEST_DELAY = 2
CSV_FILE = "startech_products.csv"

# Provide your Firecrawl API key if using their cloud service
FIRECRAWL_API_KEY = "fc-bf71dc80a07e4f6198db328c9e705a8c"  # or use os.getenv("FIRECRAWL_API_KEY") if you prefer environment variables
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


async def discover_categories(root_url: str):
    logging.info(f"==== Extracting categories from homepage: {root_url}")
    html = await fetch_html(root_url)
    if not html:
        logging.error("Failed to fetch homepage.")
        return []

    soup = BeautifulSoup(html, "html.parser")
    categories = set()
    root_categories = set()

    for a in soup.select("nav.navbar .nav-item > a.nav-link"):
        href = a.get("href", "").strip()
        if not href or href == "#" or href == "/":
            continue
        full_url = urljoin(root_url, href)
        path = urlparse(full_url).path.strip("/")
        if "/" not in path:
            root_categories.add(full_url)
            categories.add(full_url)

    for a in soup.select("nav.navbar .dropdown-menu a"):
        href = a.get("href", "").strip()
        if not href or href == "#" or href == "/":
            continue
        full_url = urljoin(root_url, href)
        path = urlparse(full_url).path.strip("/")
        path_parts = path.split("/")
        if len(path_parts) == 2 and any(root_cat.endswith(path_parts[0]) for root_cat in root_categories):
            categories.add(full_url)

    filtered = sorted(list(categories))
    logging.info(f"Found {len(filtered)} categories.")
    return filtered


async def scrape_category(url: str):
    logging.info(f"==== Scraping category page: {url}")
    html = await fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    products = []
    category = urlparse(url).path.strip("/").split("/")[0]

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
                "title": title,
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


async def run_scraper():
    init_csv(CSV_FILE)
    categories = await discover_categories(ROOT_URL)
    if not categories:
        logging.error("No categories found.")
        return

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
    try:
        asyncio.run(run_scraper())
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")