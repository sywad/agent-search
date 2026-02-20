import os
import time
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


class ShopifyScraper:
    """Shopify (Shop.app) scraper using the Shopify Catalog API."""

    AUTH_URL = 'https://api.shopify.com/auth/access_token'
    SEARCH_URL = 'https://discover.shopifyapps.com/global/v2/search'

    def __init__(self):
        self.client_id = os.environ.get('SHOPIFY_CLIENT_ID', '')
        self.client_secret = os.environ.get('SHOPIFY_CLIENT_SECRET', '')
        self._token = None
        self._token_expiry = 0
        self.session = requests.Session()

        if not self.client_id or not self.client_secret or \
           self.client_id == 'your_client_id' or self.client_secret == 'your_client_secret':
            print("[Shopify] Warning: SHOPIFY_CLIENT_ID / SHOPIFY_CLIENT_SECRET not configured. Shopify results will be empty.")

    def _get_token(self) -> str:
        """Get a valid bearer token, refreshing if expired."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        try:
            resp = self.session.post(self.AUTH_URL, json={
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'client_credentials',
            }, timeout=10)

            if resp.status_code != 200:
                print(f"[Shopify] Auth error: status={resp.status_code}")
                return ''

            data = resp.json()
            self._token = data.get('access_token', '')
            expires_in = data.get('expires_in', 3600)
            self._token_expiry = time.time() + expires_in
            return self._token

        except Exception as e:
            print(f"[Shopify] Auth error: {e}")
            return ''

    def _search(self, query: str, limit: int = 10) -> list:
        """Single API search call."""
        token = self._get_token()
        if not token:
            return []

        try:
            resp = self.session.get(self.SEARCH_URL, params={
                'query': query,
                'limit': min(limit, 10),
                'available_for_sale': '1',
                'ships_to': 'US',
            }, headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json',
            }, timeout=15)

            if resp.status_code != 200:
                print(f"[Shopify] Search error: status={resp.status_code}")
                return []

            data = resp.json()
            # Response may be a list or wrapped in a key
            if isinstance(data, list):
                return data
            return data.get('products', data.get('results', []))

        except Exception as e:
            print(f"[Shopify] Search error: {e}")
            return []

    def _parse_product(self, item: dict) -> Dict:
        """Map API response to standard product dict."""
        try:
            title = item.get('title', '')
            if not title:
                return None

            # Price — handle both cents (integer) and dollars (float/string)
            price_range = item.get('priceRange', item.get('price_range', {}))
            min_price = price_range.get('min', price_range.get('minVariantPrice', {}))
            if isinstance(min_price, dict):
                amount = min_price.get('amount', 0)
            else:
                amount = min_price or 0

            # Convert: if amount looks like cents (integer > 100), divide by 100
            try:
                amount = float(amount)
                if amount > 0 and amount == int(amount) and amount >= 100:
                    amount = amount / 100.0
                price = f'${amount:.2f}'
            except (ValueError, TypeError):
                price = 'N/A'

            # Rating
            rating_data = item.get('rating', {})
            if isinstance(rating_data, dict):
                rating = str(rating_data.get('rating', rating_data.get('average', 'N/A')))
                reviews = str(rating_data.get('count', '0'))
            else:
                rating = 'N/A'
                reviews = '0'

            # URL
            url = item.get('url', item.get('onlineStoreUrl', ''))

            # Image
            media = item.get('media', item.get('images', []))
            if isinstance(media, list) and len(media) > 0:
                first = media[0]
                image = first.get('url', first.get('src', '')) if isinstance(first, dict) else str(first)
            else:
                image = ''

            return {
                'title': title,
                'price': price,
                'rating': rating,
                'reviews': reviews,
                'url': url,
                'image': image,
                'product_id': str(item.get('id', '')),
                'asin': '',
                'source': 'shopify',
            }
        except Exception:
            return None

    def scrape_search_results(self, query: str, max_results: int = 60) -> List[Dict]:
        """Fetch Shopify search results (API caps at 10 per request)."""
        if not self.client_id or self.client_id == 'your_client_id':
            return []

        items = self._search(query, 10)
        products = []
        for item in items:
            product = self._parse_product(item)
            if product:
                products.append(product)

        print(f"[Shopify] Total products for '{query}': {len(products)}")
        return products[:max_results]

    def scrape_multiple_queries(self, queries: List[str], max_results_per_query: int = 60) -> Dict[str, List[Dict]]:
        """Fetch Shopify results for multiple queries in parallel."""
        def _scrape_one(query_text):
            scraper = ShopifyScraper()
            print(f"[Shopify] Searching for: {query_text}")
            products = scraper.scrape_search_results(query_text, max_results_per_query)
            return query_text, products

        results = {}
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [executor.submit(_scrape_one, q) for q in queries]
            for future in as_completed(futures):
                query_text, products = future.result()
                results[query_text] = products

        return results
