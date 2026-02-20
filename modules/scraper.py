import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


class AmazonScraper:
    USER_AGENTS = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    ]

    def __init__(self):
        """Initialize Amazon scraper with headers to mimic browser."""
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
        self.base_url = 'https://www.amazon.com'
        self.session_initialized = False

    def scrape_search_results(self, query: str, max_results: int = 60) -> List[Dict]:
        """
        Scrape Amazon search results for a given query.

        Args:
            query: Search query
            max_results: Maximum number of results to scrape (default 60)

        Returns:
            List of product dictionaries with title, price, rating, reviews, url, image
        """
        # Initialize session by visiting homepage (only once)
        if not self.session_initialized:
            try:
                print("Initializing session with Amazon homepage...")
                homepage_resp = self.session.get(self.base_url, headers=self.headers, timeout=10)
                cookies_dict = {c.name: c.value[:20] + '...' if len(c.value) > 20 else c.value for c in self.session.cookies}
                print(f"Homepage status: {homepage_resp.status_code}, Cookies: {cookies_dict}")
                time.sleep(0.5)
                self.session_initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize session: {e}")

        products = []
        page = 1
        max_pages = (max_results // 20) + 1  # Amazon shows ~20 results per page

        while len(products) < max_results and page <= max_pages:
            try:
                # Add random delay to avoid rate limiting
                if page > 1:
                    time.sleep(random.uniform(0.3, 0.7))

                url = f"{self.base_url}/s"
                params = {
                    'k': query,
                    'page': page,
                }

                # Retry with a fresh session + UA on failure
                response = None
                for attempt in range(2):
                    if attempt > 0:
                        # Reset session entirely with new UA
                        self.session = requests.Session()
                        self.headers['User-Agent'] = random.choice(self.USER_AGENTS)
                        print(f"Attempt {attempt+1}: retrying in 1.5s with fresh session...")
                        time.sleep(1.5)
                        # Re-visit homepage to get fresh cookies
                        try:
                            self.session.get(self.base_url, headers=self.headers, timeout=10)
                            time.sleep(0.5)
                        except:
                            pass
                    response = self.session.get(url, headers=self.headers, params=params, timeout=15, allow_redirects=True)
                    if response.status_code == 200 and len(response.content) > 5000:
                        break
                    print(f"Attempt {attempt+1}: status={response.status_code}, len={len(response.content)}")

                print(f"Search page {page} status: {response.status_code}, Content length: {len(response.content)}")

                if response.status_code != 200:
                    print(f"Failed to fetch page {page}: Status {response.status_code}")
                    # Save HTML for debugging
                    with open(f'/tmp/amazon_error_page{page}.html', 'w') as f:
                        f.write(response.text)
                    print(f"Response saved to /tmp/amazon_error_page{page}.html")
                    break

                soup = BeautifulSoup(response.content, 'lxml')

                # Try multiple selectors for product cards
                items = soup.find_all('div', {'data-component-type': 's-search-result'})
                print(f"Found {len(items)} items with primary selector")

                if not items:
                    # Try multiple alternative selectors
                    items = soup.select('[data-asin]:not([data-asin=""])')
                    if not items:
                        items = soup.select('.s-result-item[data-asin]')
                    if not items:
                        items = soup.find_all('div', {'class': 's-result-item', 'data-asin': True})

                    if items:
                        print(f"Using alternative selector, found {len(items)} items")

                if not items:
                    print(f"No items found on page {page}. Checking for CAPTCHA or blocks...")
                    # Check if blocked
                    if 'captcha' in response.text.lower() or 'robot' in response.text.lower():
                        print("ERROR: Amazon is blocking requests with CAPTCHA")
                        print("Please try:")
                        print("1. Use a VPN or proxy")
                        print("2. Add cookies from an authenticated browser session")
                        print("3. Use Amazon's official Product Advertising API instead")
                    # Save HTML for debugging
                    with open(f'/tmp/amazon_debug_page{page}.html', 'w') as f:
                        f.write(response.text)
                    print(f"HTML saved to /tmp/amazon_debug_page{page}.html for debugging")
                    break

                extracted_count = 0
                for item in items:
                    try:
                        product = self._extract_product_info(item)
                        if product:
                            products.append(product)
                            extracted_count += 1
                            if len(products) >= max_results:
                                break
                    except Exception as e:
                        print(f"Error extracting product: {e}")
                        continue

                print(f"Successfully extracted {extracted_count} out of {len(items)} items on page {page}")

                page += 1

            except Exception as e:
                print(f"Error scraping page {page}: {e}")
                import traceback
                traceback.print_exc()
                break

        print(f"Total products scraped for '{query}': {len(products)}")
        return products[:max_results]

    def _extract_product_info(self, item) -> Dict:
        """Extract product information from a search result item."""
        product = {}

        # Title - try multiple selectors
        title_elem = item.find('h2')
        if title_elem:
            title_link = title_elem.find('a')
            if title_link:
                product['title'] = title_link.get_text(strip=True)
                href = title_link.get('href', '')
                # Handle relative URLs
                if href.startswith('/'):
                    product['url'] = self.base_url + href
                else:
                    product['url'] = href
            else:
                # Some items might not have clickable titles
                product['title'] = title_elem.get_text(strip=True)
                product['url'] = ''
        else:
            # Skip items without titles
            return None

        # Skip if title is empty
        if not product.get('title'):
            return None

        # Price
        price_whole = item.find('span', {'class': 'a-price-whole'})
        price_fraction = item.find('span', {'class': 'a-price-fraction'})
        if price_whole:
            price = price_whole.get_text(strip=True)
            if price_fraction:
                price += price_fraction.get_text(strip=True)
            product['price'] = price
        else:
            product['price'] = 'N/A'

        # Rating
        rating_elem = item.find('span', {'class': 'a-icon-alt'})
        if rating_elem:
            rating_text = rating_elem.get_text(strip=True)
            product['rating'] = rating_text.split()[0] if rating_text else 'N/A'
        else:
            product['rating'] = 'N/A'

        # Number of reviews - look for element with text like "X ratings"
        product['reviews'] = '0'
        for elem in item.find_all(attrs={'aria-label': True}):
            aria_label = elem.get('aria-label', '')
            # Look specifically for pattern like "31,679 ratings"
            if 'rating' in aria_label.lower() and re.search(r'\d', aria_label):
                # Skip if it contains "star" (that's the rating, not review count)
                if 'star' not in aria_label.lower():
                    match = re.search(r'([\d,]+)', aria_label)
                    if match:
                        product['reviews'] = match.group(1)
                        break

        # Image — also use alt text as a fuller title if available
        img_elem = item.find('img', {'class': 's-image'})
        if img_elem:
            img_url = img_elem.get('src', '')
            # Upgrade Amazon thumbnail to higher-res version
            if 'm.media-amazon.com' in img_url:
                img_url = re.sub(r'\._[A-Z]{2}_[A-Z0-9_]+_\.', '._AC_SL500_.', img_url)
            product['image'] = img_url
            alt_text = img_elem.get('alt', '').strip()
            if alt_text and len(alt_text) > len(product.get('title', '')):
                product['title'] = alt_text
        else:
            product['image'] = ''

        # ASIN (Amazon Standard Identification Number)
        asin = item.get('data-asin', '')
        product['asin'] = asin
        product['product_id'] = asin
        product['source'] = 'amazon'

        # Use canonical detail page URL when ASIN is available
        if asin:
            product['url'] = f'{self.base_url}/dp/{asin}'

        return product

    def scrape_multiple_queries(self, queries: List[str], max_results_per_query: int = 60) -> Dict[str, List[Dict]]:
        """
        Scrape Amazon for multiple queries in parallel.

        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query

        Returns:
            Dictionary mapping query to list of products
        """
        # Initialize session once to get cookies
        if not self.session_initialized:
            try:
                print("Initializing session with Amazon homepage...")
                self.session.get(self.base_url, headers=self.headers, timeout=10)
                time.sleep(0.5)
                self.session_initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize session: {e}")

        init_cookies = self.session.cookies.copy()

        def _scrape_one(query_text):
            scraper = AmazonScraper()
            scraper.session.cookies.update(init_cookies)
            scraper.session_initialized = True  # skip homepage visit
            print(f"Scraping results for: {query_text}")
            products = scraper.scrape_search_results(query_text, max_results_per_query)
            print(f"Found {len(products)} products for '{query_text}'")
            return query_text, products

        results = {}
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [executor.submit(_scrape_one, q) for q in queries]
            for future in as_completed(futures):
                query_text, products = future.result()
                results[query_text] = products

        return results
