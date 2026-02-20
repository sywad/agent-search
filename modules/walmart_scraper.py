import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import random
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class WalmartScraper:
    USER_AGENTS = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    ]

    def __init__(self):
        """Initialize Walmart scraper with headers to mimic browser."""
        self.session = requests.Session()
        ua = random.choice(self.USER_AGENTS)
        self.headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        self.base_url = 'https://www.walmart.com'
        self.session_initialized = False

    def scrape_search_results(self, query: str, max_results: int = 60) -> List[Dict]:
        """
        Scrape Walmart search results for a given query.

        Args:
            query: Search query
            max_results: Maximum number of results to scrape (default 60)

        Returns:
            List of product dictionaries with title, price, rating, reviews, url, image, product_id, source
        """
        if not self.session_initialized:
            try:
                print("Initializing session with Walmart homepage...")
                homepage_resp = self.session.get(self.base_url, headers=self.headers, timeout=10)
                print(f"Walmart homepage status: {homepage_resp.status_code}")
                time.sleep(0.5)
                self.session_initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize Walmart session: {e}")

        products = []
        page = 1
        max_pages = (max_results // 40) + 1  # Walmart shows ~40 results per page

        while len(products) < max_results and page <= max_pages:
            try:
                if page > 1:
                    time.sleep(random.uniform(0.3, 0.7))

                url = f"{self.base_url}/search"
                params = {'q': query}
                if page > 1:
                    params['page'] = page

                response = None
                for attempt in range(2):
                    if attempt > 0:
                        self.session = requests.Session()
                        self.headers['User-Agent'] = random.choice(self.USER_AGENTS)
                        print(f"Walmart attempt {attempt+1}: retrying in 1.5s with fresh session...")
                        time.sleep(1.5)
                        try:
                            self.session.get(self.base_url, headers=self.headers, timeout=10)
                            time.sleep(0.5)
                        except:
                            pass
                    response = self.session.get(url, headers=self.headers, params=params, timeout=15, allow_redirects=True)
                    if response.status_code == 200 and len(response.content) > 5000:
                        break
                    print(f"Walmart attempt {attempt+1}: status={response.status_code}, len={len(response.content)}")

                print(f"Walmart search page {page} status: {response.status_code}, Content length: {len(response.content)}")

                if response.status_code != 200:
                    print(f"Failed to fetch Walmart page {page}: Status {response.status_code}")
                    break

                soup = BeautifulSoup(response.content, 'lxml')

                # Try to extract product data from embedded JSON (Walmart uses Next.js)
                page_products = self._extract_from_json(response.text)

                if not page_products:
                    # Fallback: parse HTML directly
                    page_products = self._extract_from_html(soup)

                if not page_products:
                    print(f"No Walmart items found on page {page}.")
                    if 'captcha' in response.text.lower() or 'robot' in response.text.lower():
                        print("ERROR: Walmart is blocking requests with CAPTCHA")
                    with open(f'/tmp/walmart_debug_page{page}.html', 'w') as f:
                        f.write(response.text)
                    print(f"HTML saved to /tmp/walmart_debug_page{page}.html for debugging")
                    break

                extracted_count = 0
                for product in page_products:
                    if product:
                        products.append(product)
                        extracted_count += 1
                        if len(products) >= max_results:
                            break

                print(f"Walmart: extracted {extracted_count} items on page {page}")
                page += 1

            except Exception as e:
                print(f"Error scraping Walmart page {page}: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"Total Walmart products scraped for '{query}': {len(products)}")
        return products[:max_results]

    def _extract_from_json(self, html_text: str) -> List[Dict]:
        """Try to extract product data from Walmart's embedded JSON/script tags."""
        products = []
        try:
            # Walmart often embeds product data in __NEXT_DATA__ script tag
            match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html_text, re.DOTALL)
            if not match:
                return products

            data = json.loads(match.group(1))
            # Navigate the JSON structure to find search results
            props = data.get('props', {}).get('pageProps', {})
            initial_data = props.get('initialData', {})
            search_result = initial_data.get('searchResult', {})
            item_stacks = search_result.get('itemStacks', [])

            for stack in item_stacks:
                items = stack.get('items', [])
                for item in items:
                    product = self._parse_json_item(item)
                    if product:
                        products.append(product)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Walmart JSON extraction error: {e}")

        return products

    def _parse_json_item(self, item: dict) -> Dict:
        """Parse a single product item from Walmart's JSON data."""
        try:
            title = item.get('name', '')
            if not title:
                return None

            product_id = str(item.get('usItemId', '') or item.get('id', ''))
            price_info = item.get('priceInfo', {}) or {}
            # Walmart uses linePrice for current price, linePriceDisplay for formatted
            price = price_info.get('linePrice', '') or price_info.get('linePriceDisplay', '') or price_info.get('itemPrice', '')
            if not price:
                price = 'N/A'

            rating_info = item.get('averageRating', 0)
            rating = str(round(float(rating_info), 1)) if rating_info else 'N/A'

            reviews = str(item.get('numberOfReviews', 0) or 0)

            canonical_url = item.get('canonicalUrl', '')
            if canonical_url and not canonical_url.startswith('http'):
                url = f"{self.base_url}{canonical_url}"
            elif canonical_url:
                url = canonical_url
            else:
                url = f"{self.base_url}/ip/{product_id}" if product_id else ''

            image_info = item.get('imageInfo', {}) or {}
            image = image_info.get('thumbnailUrl', '')
            if not image:
                image = item.get('image', '')

            return {
                'title': title,
                'price': price,
                'rating': rating,
                'reviews': reviews,
                'url': url,
                'image': image,
                'asin': '',
                'product_id': product_id,
                'source': 'walmart',
            }
        except Exception:
            return None

    def _extract_from_html(self, soup) -> List[Dict]:
        """Fallback: extract products from HTML elements."""
        products = []

        # Try common Walmart search result selectors
        items = soup.select('[data-item-id]')
        if not items:
            items = soup.select('[data-product-id]')
        if not items:
            items = soup.select('.search-result-gridview-item')

        for item in items:
            try:
                product = {}

                # Title
                title_elem = item.select_one('[data-automation-id="product-title"], .product-title-link, a[class*="product-title"]')
                if not title_elem:
                    title_elem = item.find('a', {'class': re.compile(r'product', re.I)})
                if not title_elem:
                    title_elem = item.find('span', {'class': re.compile(r'title', re.I)})

                if title_elem:
                    product['title'] = title_elem.get_text(strip=True)
                    href = title_elem.get('href', '')
                    if href and href.startswith('/'):
                        product['url'] = self.base_url + href
                    elif href:
                        product['url'] = href
                    else:
                        product['url'] = ''
                else:
                    continue

                if not product.get('title'):
                    continue

                # Price
                price_elem = item.select_one('[data-automation-id="product-price"], [class*="price"]')
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    price_match = re.search(r'\$[\d,.]+', price_text)
                    product['price'] = price_match.group(0) if price_match else 'N/A'
                else:
                    product['price'] = 'N/A'

                # Rating
                rating_elem = item.select_one('[class*="rating"], [class*="stars"]')
                if rating_elem:
                    rating_text = rating_elem.get('aria-label', '') or rating_elem.get_text(strip=True)
                    rating_match = re.search(r'([\d.]+)\s*(?:out of|stars|/)', rating_text)
                    product['rating'] = rating_match.group(1) if rating_match else 'N/A'
                else:
                    product['rating'] = 'N/A'

                # Reviews
                review_elem = item.select_one('[class*="review-count"], [class*="ratings-count"]')
                if review_elem:
                    review_text = review_elem.get_text(strip=True)
                    review_match = re.search(r'([\d,]+)', review_text)
                    product['reviews'] = review_match.group(1) if review_match else '0'
                else:
                    product['reviews'] = '0'

                # Image
                img_elem = item.find('img')
                product['image'] = img_elem.get('src', '') if img_elem else ''

                # Product ID
                product_id = item.get('data-item-id', '') or item.get('data-product-id', '')
                product['asin'] = ''
                product['product_id'] = product_id
                product['source'] = 'walmart'

                products.append(product)
            except Exception:
                continue

        return products

    def scrape_multiple_queries(self, queries: List[str], max_results_per_query: int = 60) -> Dict[str, List[Dict]]:
        """
        Scrape Walmart for multiple queries in parallel.

        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query

        Returns:
            Dictionary mapping query to list of products
        """
        # Initialize session once to get cookies
        if not self.session_initialized:
            try:
                print("Initializing session with Walmart homepage...")
                self.session.get(self.base_url, headers=self.headers, timeout=10)
                time.sleep(0.5)
                self.session_initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize Walmart session: {e}")

        init_cookies = self.session.cookies.copy()

        def _scrape_one(query_text):
            scraper = WalmartScraper()
            scraper.session.cookies.update(init_cookies)
            scraper.session_initialized = True  # skip homepage visit
            print(f"[Walmart] Scraping results for: {query_text}")
            products = scraper.scrape_search_results(query_text, max_results_per_query)
            print(f"[Walmart] Found {len(products)} products for '{query_text}'")
            return query_text, products

        results = {}
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [executor.submit(_scrape_one, q) for q in queries]
            for future in as_completed(futures):
                query_text, products = future.result()
                results[query_text] = products

        return results
