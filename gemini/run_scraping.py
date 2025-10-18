# FILE: run_scraping.py
import asyncio
import time
import os
import nest_asyncio

# Apply nest_asyncio to allow asyncio.run() to work with other running loops
# (like the one crawl4ai might use)
nest_asyncio.apply()

# Import our new modular scrapers
import scraper_daraz_modular
import scraper_startech_modular

# --- 1. CONFIGURE YOUR TARGETS HERE ---
DARAZ_CATEGORIES = [
    "https://www.daraz.com.bd/mini-cameras/",
    "https://www.daraz.com.bd/smart-watches/",
    # Add more Daraz links here
]

STARTECH_CATEGORIES = [
    "https://www.startech.com.bd/laptop-notebook/laptop",
    "https://www.startech.com.bd/component/processor",
    # Add more Startech links here
]

# Total duration you want the scrapers to run
SCRAPE_DURATION_SECONDS = 5 * 60  # 5 minutes

async def main():
    print(f"--- STARTING COMBINED SCRAPE ---")
    print(f"Target duration: {SCRAPE_DURATION_SECONDS} seconds")
    
    # Ensure the 'result' directory exists for the Daraz scraper
    os.makedirs("result", exist_ok=True)

    start_time = time.time()

    # Create the two main scraping tasks
    daraz_task = asyncio.create_task(
        scraper_daraz_modular.main(DARAZ_CATEGORIES)
    )
    
    startech_task = asyncio.create_task(
        scraper_startech_modular.run_scraper(STARTECH_CATEGORIES)
    )

    # This is the core logic: run both tasks together, but enforce a timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(daraz_task, startech_task),
            timeout=SCRAPE_DURATION_SECONDS
        )
        print("--- All scrapers finished *before* the time limit. ---")

    except asyncio.TimeoutError:
        print(f"--- {SCRAPE_DURATION_SECONDS}s limit reached. Sending cancellation signal... ---")
        
        # This is crucial: we tell the tasks to cancel
        daraz_task.cancel()
        startech_task.cancel()
        
        # We wait for them to acknowledge the cancellation (or fail)
        # return_exceptions=True prevents one failed task from stopping the other
        await asyncio.gather(daraz_task, startech_task, return_exceptions=True)
        
        print("--- Scraper tasks have been cancelled. ---")
        
    except Exception as e:
        print(f"--- An unexpected error occurred: {e} ---")

    finally:
        end_time = time.time()
        print(f"--- SCRAPING PHASE COMPLETE ---")
        print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Next steps:")
        print(f"1. Check 'result/' folders for Daraz JSON files.")
        print(f"2. Check 'startech_products.csv' for Startech CSV.")
        print(f"3. Run 'python build_corpus.py' to unify them.")

if __name__ == "__main__":
    asyncio.run(main())