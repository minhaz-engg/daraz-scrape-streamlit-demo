import logging
import random
import time
from typing import Dict
from urllib.parse import urlparse

import requests
import urllib.robotparser as robotparser

# ======================= Logging =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOG = logging.getLogger("pdp-scraper")

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


class PoliteSession:
    """
    A small helper that:
      - Respects robots.txt (for UA '*')
      - Applies per-host pacing
      - Retries with exponential backoff + jitter
      - Honors Retry-After
      - Maintains simple ETag/Last-Modified body cache
      - Detects bot walls and aborts
    """
    def __init__(self, min_delay_s: float = 1.5, max_retries: int = 4, timeout: int = 25):
        self.sess = requests.Session()
        self.sess.headers.update(DEFAULT_HEADERS)
        self.min_delay_s = max(0.0, float(min_delay_s))
        self.max_retries = max(0, int(max_retries))
        self.timeout = timeout
        self._last_req_ts: Dict[str, float] = {}  # per-host last request ts
        self._robots: Dict[str, robotparser.RobotFileParser] = {}
        self._etag: Dict[str, str] = {}
        self._last_mod: Dict[str, str] = {}
        self._body_cache: Dict[str, str] = {}

    def _sleep_if_needed(self, host: str):
        now = time.time()
        last = self._last_req_ts.get(host, 0)
        wait = self.min_delay_s - (now - last)
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.25))  # modest jitter
        self._last_req_ts[host] = time.time()

    def _robots_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            robots_url = f"{base}/robots.txt"
            LOG.info("Fetching robots.txt: %s", robots_url)
            rp = robotparser.RobotFileParser()
            try:
                r = self.sess.get(robots_url, timeout=self.timeout)
                if r.status_code == 200 and r.text:
                    rp.parse(r.text.splitlines())
                else:
                    # If robots.txt missing/unreadable, default to allow
                    rp.parse([])
            except Exception:
                rp.parse([])
            self._robots[base] = rp
        rp = self._robots[base]
        return rp.can_fetch("*", url)

    def _maybe_add_conditional_headers(self, url: str, headers: Dict[str, str]):
        etag = self._etag.get(url)
        if etag:
            headers["If-None-Match"] = etag
        last_mod = self._last_mod.get(url)
        if last_mod:
            headers["If-Modified-Since"] = last_mod

    def _looks_like_captcha(self, text: str) -> bool:
        lower = text.lower()
        return any(hint in lower for hint in CAPTCHA_HINTS)

    def get(self, url: str) -> str:
        if not self._robots_allowed(url):
            raise PermissionError(f"robots.txt disallows fetching this URL: {url}")

        host = urlparse(url).netloc
        self._sleep_if_needed(host)

        attempt = 0
        backoff = 1.2
        while True:
            attempt += 1
            headers = dict(self.sess.headers)
            self._maybe_add_conditional_headers(url, headers)

            try:
                r = self.sess.get(url, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt > self.max_retries:
                    raise
                LOG.warning("Network error (%s). Retrying in %.1fs ...", e, backoff)
                time.sleep(backoff + random.uniform(0, 0.3))
                backoff *= 2
                continue

            # Honor Retry-After on 429/503
            if r.status_code in (429, 503):
                if attempt > self.max_retries:
                    r.raise_for_status()
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                delay = min(delay, 30.0)
                LOG.warning("%s received (%d). Backing off %.1fs ...", url, r.status_code, delay)
                time.sleep(delay + random.uniform(0, 0.3))
                backoff = min(backoff * 2, 30.0)
                continue

            # 304 Not Modified -> use cached
            if r.status_code == 304 and url in self._body_cache:
                LOG.info("HTTP 304 Not Modified for %s; using cached body.", url)
                return self._body_cache[url]

            r.raise_for_status()
            text = r.text or ""

            # Save caching headers
            if et := r.headers.get("ETag"):
                self._etag[url] = et
            if lm := r.headers.get("Last-Modified"):
                self._last_mod[url] = lm
            self._body_cache[url] = text

            # Detect bot wall
            if self._looks_like_captcha(text):
                raise PermissionError("Suspected bot wall / challenge page encountered; stopping politely.")

            return text