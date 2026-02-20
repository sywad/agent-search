import time
import json
import requests
from bs4 import BeautifulSoup

from .llm_client import generate

# Module-level timestamp for rate limiting requests
_last_wc_request = 0.0
_MIN_SPACING = 2.0  # seconds between requests

USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
)

SYSTEM_INSTRUCTION = (
    "You are a helpful shopping research assistant. You analyze expert editorial reviews "
    "from Wirecutter (NYT) to extract professional product insights."
)


class WirecutterResearcher:

    def _rate_limited_get(self, url, params=None):
        """GET request with minimum 2s spacing between calls."""
        global _last_wc_request
        now = time.time()
        wait = _MIN_SPACING - (now - _last_wc_request)
        if wait > 0:
            time.sleep(wait)

        headers = {'User-Agent': USER_AGENT}
        for attempt in range(2):
            _last_wc_request = time.time()
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', 10))
                    if attempt == 0:
                        print(f"Wirecutter 429, retrying in {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    return None
                if resp.status_code != 200:
                    print(f"Wirecutter request error: status={resp.status_code}")
                    return None
                return resp
            except Exception as e:
                print(f"Wirecutter request error: {e}")
                if attempt == 0:
                    time.sleep(2)
                    continue
                return None
        return None

    def _search_wirecutter(self, query, limit=5):
        """Search Wirecutter's own site search for relevant articles."""
        url = 'https://www.nytimes.com/wirecutter/search/'
        params = {'s': query}
        resp = self._rate_limited_get(url, params)
        if not resp:
            return []

        soup = BeautifulSoup(resp.text, 'lxml')
        results = []

        for card in soup.select('a[href*="/wirecutter/reviews/"], a[href*="/wirecutter/money/"]'):
            href = card.get('href', '')

            # Get title from heading inside the link
            h = card.select_one('h2, h3')
            title = h.get_text(strip=True) if h else card.get_text(strip=True)[:80]

            if not title or len(title) < 10:
                continue

            # Look for snippet in sibling paragraph
            snippet = ''
            parent = card.parent
            if parent:
                p = parent.select_one('p')
                if p:
                    snippet = p.get_text(strip=True)

            # Deduplicate by URL
            if any(r['url'] == href for r in results):
                continue

            results.append({
                'title': title,
                'url': href,
                'snippet': snippet,
            })

        return results[:limit]

    def _scrape_article(self, url):
        """Fetch a Wirecutter article and extract key content.

        Prioritizes article structure (headings + first paragraph per section)
        so the LLM sees all products mentioned, not just the first 2000 chars.
        """
        resp = self._rate_limited_get(url)
        if not resp:
            return ''

        try:
            soup = BeautifulSoup(resp.text, 'lxml')

            # Remove script/style noise
            for tag in soup.select('script, style, nav, footer, header'):
                tag.decompose()

            # Get article title
            title = ''
            title_el = soup.select_one('h1')
            if title_el:
                title = title_el.get_text(strip=True)

            # Extract structured content: headings + their following paragraphs
            # This captures ALL product picks rather than just the first 2000 chars
            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")

            # Gather all h2/h3 headings (contain pick categories and product names)
            headings = []
            for h in soup.select('h2, h3'):
                text = h.get_text(strip=True)
                if text and len(text) < 200:
                    headings.append(text)
            if headings:
                content_parts.append("Sections: " + ' | '.join(headings))

            # Walk through the article collecting heading+paragraph pairs
            seen_paras = 0
            for el in soup.select('h2, h3, article p, main p'):
                tag = el.name
                text = el.get_text(strip=True)
                if not text or len(text) < 20:
                    continue

                if tag in ('h2', 'h3'):
                    content_parts.append(f"\n## {text}")
                    seen_paras = 0  # reset counter for new section
                else:
                    # Keep first 2 paragraphs per section for context
                    if seen_paras < 2:
                        content_parts.append(text)
                        seen_paras += 1

            full_text = '\n'.join(content_parts)
            return full_text[:4000]
        except Exception as e:
            print(f"Wirecutter scrape error: {e}")
            return ''

    def _extract_search_terms(self, products, model):
        """Single batch LLM call to extract brand+model search terms for all products."""
        titles = []
        ranks = []
        for p in products:
            titles.append(p.get('title', ''))
            ranks.append(p.get('rank', 0))

        prompt = (
            "Given these product titles, extract a short search term (brand + model name) "
            "for each that would work well as a Wirecutter search query. Return ONLY a JSON array "
            "of strings, one per product, in the same order.\n\n"
        )
        for i, title in enumerate(titles):
            prompt += f"{i + 1}. {title}\n"
        prompt += "\nReturn a JSON array like: [\"Brand Model\", \"Brand Model2\", ...]"

        try:
            response_text, _ = generate(model, prompt, SYSTEM_INSTRUCTION, temperature=0)
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start >= 0 and end > start:
                terms = json.loads(response_text[start:end])
                if len(terms) == len(products):
                    return {ranks[i]: terms[i] for i in range(len(terms))}
        except Exception as e:
            print(f"Wirecutter search term extraction error: {e}")

        # Fallback: first 4 words of each title
        result = {}
        for p in products:
            words = p.get('title', '').split()[:4]
            result[p.get('rank', 0)] = ' '.join(words)
        return result

    def _synthesize_insight(self, product_title, search_term, article_texts, model):
        """LLM-synthesize expert review insights from Wirecutter article excerpts."""
        if not article_texts:
            return ''

        excerpts = ''
        for i, text in enumerate(article_texts[:5]):
            excerpts += f"\n--- Article {i + 1} ---\n{text[:1200]}"

        prompt = (
            f'Based on these Wirecutter (NYT) article excerpts, summarize what Wirecutter says '
            f'about this product: "{product_title}" (searched as "{search_term}").\n\n'
            f'Article excerpts:{excerpts}\n\n'
            f'Summarize any mentions, impressions, reviews, or opinions about this product or very '
            f'similar products. Note whether Wirecutter recommends it, praises it, criticizes it, '
            f'or compares it to alternatives. If the product is mentioned even briefly (e.g. first '
            f'impressions, announcements), include that information. '
            f'Only say "not mentioned" if the product truly does not appear at all. '
            f'Be concise — 2-3 sentences max.'
        )

        try:
            insight_text, _ = generate(model, prompt, SYSTEM_INSTRUCTION, temperature=0)
            return insight_text.strip()
        except Exception as e:
            print(f"Wirecutter insight synthesis error: {e}")
            return ''

    def research_products_stream(self, products, query, model):
        """
        Generator yielding Wirecutter research results as they complete.
        Yields tuples: (rank, search_term, articles, insight)
        - rank='_global' for the global query search (insight=None)
        - rank=int for per-product results
        """
        # Step 1: Extract search terms for all products in one batch LLM call
        search_terms = self._extract_search_terms(products, model)

        # Step 2: Global search for the overall query
        global_articles = self._search_wirecutter(query, limit=5)
        yield ('_global', query, global_articles, None)

        # Step 3: Scrape top global articles for context
        global_context = []
        for article in global_articles[:3]:
            text = self._scrape_article(article['url'])
            if text:
                global_context.append(text)

        # Step 4: Per-product search + scrape + synthesis
        for product in products:
            rank = product.get('rank', 0)
            title = product.get('title', '')
            search_term = search_terms.get(rank, title.split()[:4])
            if isinstance(search_term, list):
                search_term = ' '.join(search_term)

            articles = self._search_wirecutter(search_term, limit=3)

            # Scrape top 1-2 articles for this product (prioritized over global)
            article_texts = []
            for article in articles[:2]:
                text = self._scrape_article(article['url'])
                if text:
                    article_texts.append(text)
            # Append global context after product-specific articles
            article_texts.extend(global_context)

            insight = self._synthesize_insight(title, search_term, article_texts, model)
            yield (rank, search_term, articles, insight)
