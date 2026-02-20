import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

from .llm_client import generate

SYSTEM_INSTRUCTION = (
    "You are a helpful shopping assistant. Write concise, factual product summaries "
    "based on the detail page content provided. Focus on unique differentiating details "
    "that are NOT already visible on the product card (title, price, rating, review count "
    "are already shown). Instead highlight: specific technical specs, materials, what's "
    "included, compatibility, standout pros/cons from reviews, and who it's best suited for."
)

USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
]

HEADERS_TEMPLATE = {
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


class DetailSummarizer:

    def _fetch_detail_page(self, url):
        """Fetch and extract text from a product detail page. Returns '' on any error."""
        if not url:
            return ''
        try:
            headers = {**HEADERS_TEMPLATE, 'User-Agent': random.choice(USER_AGENTS)}
            resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            if resp.status_code != 200:
                print(f"Detail page fetch failed: status={resp.status_code} url={url[:80]}")
                return ''

            soup = BeautifulSoup(resp.content, 'lxml')

            # Remove non-content elements
            for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript']):
                tag.decompose()

            sections = []

            # Feature bullets
            feature_div = soup.find('div', id='feature-bullets') or soup.find('ul', class_='a-unordered-list a-vertical a-spacing-mini')
            if feature_div:
                bullets = [li.get_text(strip=True) for li in feature_div.find_all('li')]
                if bullets:
                    sections.append("Key Features:\n" + "\n".join(f"- {b}" for b in bullets[:10]))

            # Product description
            desc_div = soup.find('div', id='productDescription')
            if desc_div:
                desc_text = desc_div.get_text(strip=True)
                if desc_text:
                    sections.append(f"Description: {desc_text[:800]}")

            # Tech specs / product details table
            tech_table = soup.find('table', id='productDetails_techSpec_section_1')
            if not tech_table:
                tech_table = soup.find('table', class_='a-keyvalue')
            if tech_table:
                rows = []
                for tr in tech_table.find_all('tr'):
                    th = tr.find('th')
                    td = tr.find('td')
                    if th and td:
                        rows.append(f"{th.get_text(strip=True)}: {td.get_text(strip=True)}")
                if rows:
                    sections.append("Specifications:\n" + "\n".join(rows[:15]))

            # Review snippets
            review_section = soup.find('div', id='cm-cr-dp-review-list')
            if review_section:
                snippets = []
                for review_div in review_section.find_all('div', {'data-hook': 'review'}, limit=3):
                    body = review_div.find('span', {'data-hook': 'review-body'})
                    if body:
                        snippets.append(body.get_text(strip=True)[:200])
                if snippets:
                    sections.append("Top Reviews:\n" + "\n".join(f'- "{s}"' for s in snippets))

            if sections:
                page_text = "\n\n".join(sections)
            else:
                # Fallback: generic body text
                body_text = soup.get_text(separator=' ', strip=True)
                page_text = body_text[:3000]

            # Truncate total to ~3000 chars
            return page_text[:3000]

        except Exception as e:
            print(f"Detail page error: {e} url={url[:80]}")
            return ''

    def _fetch_and_summarize(self, product, query, model):
        """Fetch detail page and generate summary for one product."""
        rank = product.get('rank', 0)
        url = product.get('url', '')
        title = product.get('title', 'Unknown')
        price = product.get('price', 'N/A')
        rating = product.get('rating', 'N/A')
        reviews = product.get('reviews', '0')

        page_text = self._fetch_detail_page(url)

        prompt = (
            f'Based on the product detail page below, write a concise summary for a shopper searching for "{query}".\n'
            f'Product: {title}\n\n'
            f'{page_text or "No detail page content available."}\n\n'
            f'The shopper can already see the product title, price, rating, and review count on the card. '
            f'Do NOT repeat those. Instead write 2-3 sentences focusing on: unique specs or materials, '
            f'what\'s in the box, compatibility details, standout pros or cons mentioned in reviews, '
            f'and who this product is best suited for.'
        )

        try:
            summary_text, debug_info = generate(model, prompt, SYSTEM_INSTRUCTION, temperature=0)
        except Exception as e:
            print(f"LLM summary error for rank {rank}: {e}")
            summary_text = ''
            debug_info = {'error': str(e)}

        return rank, summary_text, debug_info

    def summarize_products_stream(self, products, query, model):
        """
        Generator that yields (rank, summary_text, debug_info) as each product summary completes.
        Results arrive out-of-order as workers finish.
        """
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._fetch_and_summarize, p, query, model): p
                for p in products
            }
            for future in as_completed(futures):
                try:
                    rank, summary, debug_info = future.result()
                    yield rank, summary, debug_info
                except Exception as e:
                    product = futures[future]
                    print(f"Summary worker error for rank {product.get('rank', '?')}: {e}")
                    yield product.get('rank', 0), '', {'error': str(e)}
