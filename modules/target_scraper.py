import requests
from typing import List, Dict
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


class TargetScraper:
    """Target scraper using the Redsky JSON API (no HTML parsing needed)."""

    REDSKY_URL = 'https://redsky.target.com/redsky_aggregations/v1/web/plp_search_v2'
    REDSKY_KEY = 'ff457966e64d5e877fdbad070f276d18ecec4a01'

    USER_AGENTS = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15',
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.target.com',
            'Referer': 'https://www.target.com/',
        })

    def scrape_search_results(self, query: str, max_results: int = 60) -> List[Dict]:
        """Fetch Target search results via Redsky API."""
        products = []
        offset = 0
        page_size = min(24, max_results)

        while len(products) < max_results:
            params = {
                'key': self.REDSKY_KEY,
                'channel': 'WEB',
                'count': str(page_size),
                'default_purchasability_filter': 'true',
                'keyword': query,
                'offset': str(offset),
                'page': f'/s/{query}',
                'pricing_store_id': '3991',
                'store_ids': '3991',
                'visitor_id': 'target_search_bot',
            }

            try:
                resp = self.session.get(self.REDSKY_URL, params=params, timeout=15)
                if resp.status_code != 200:
                    print(f"[Target] API error: status={resp.status_code}")
                    break

                data = resp.json()
                items = data.get('data', {}).get('search', {}).get('products', [])

                if not items:
                    break

                for item in items:
                    product = self._parse_api_item(item)
                    if product:
                        products.append(product)
                        if len(products) >= max_results:
                            break

                print(f"[Target] Fetched {len(items)} items at offset {offset}")
                offset += page_size

                if len(items) < page_size:
                    break  # last page

                if offset > 0:
                    time.sleep(random.uniform(0.3, 0.7))

            except Exception as e:
                print(f"[Target] API error: {e}")
                break

        print(f"[Target] Total products for '{query}': {len(products)}")
        return products[:max_results]

    def _parse_api_item(self, item: dict) -> Dict:
        """Parse a product from the Redsky API response."""
        try:
            tcin = str(item.get('tcin', ''))
            item_data = item.get('item', {})
            desc = item_data.get('product_description', {})
            enrichment = item_data.get('enrichment', {})
            price_data = item.get('price', {})
            rnr = item.get('ratings_and_reviews', {})

            title = desc.get('title', '')
            if not title:
                return None
            # Clean HTML entities
            title = title.replace('&#8482;', '\u2122').replace('&#174;', '\u00ae').replace('&amp;', '&')

            # Price
            price = price_data.get('formatted_current_price', '')
            if not price:
                current_retail = price_data.get('current_retail', '')
                price = f'${current_retail}' if current_retail else 'N/A'

            # Rating/reviews
            stats = rnr.get('statistics', {}).get('rating', {})
            avg = stats.get('average', 0)
            rating = str(round(float(avg), 1)) if avg else 'N/A'
            reviews = str(stats.get('count', 0) or 0)

            # URL
            buy_url = enrichment.get('buy_url', '')
            url = buy_url if buy_url else f'https://www.target.com/p/-/A-{tcin}'

            # Image
            images = enrichment.get('images', {})
            image = images.get('primary_image_url', '')

            return {
                'title': title,
                'price': price,
                'rating': rating,
                'reviews': reviews,
                'url': url,
                'image': image,
                'asin': '',
                'product_id': tcin,
                'source': 'target',
            }
        except Exception:
            return None

    def scrape_multiple_queries(self, queries: List[str], max_results_per_query: int = 60) -> Dict[str, List[Dict]]:
        """Fetch Target results for multiple queries in parallel."""
        def _scrape_one(query_text):
            scraper = TargetScraper()
            print(f"[Target] Searching for: {query_text}")
            products = scraper.scrape_search_results(query_text, max_results_per_query)
            return query_text, products

        results = {}
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [executor.submit(_scrape_one, q) for q in queries]
            for future in as_completed(futures):
                query_text, products = future.result()
                results[query_text] = products

        return results
