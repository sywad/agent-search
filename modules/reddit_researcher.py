import time
import json
import requests

from .llm_client import generate

# Module-level timestamp for rate limiting Reddit API calls
_last_reddit_request = 0.0
_MIN_SPACING = 2.0  # seconds between Reddit API requests

USER_AGENT = 'ProductSearchBot/1.0 (product research tool)'

SYSTEM_INSTRUCTION = (
    "You are a helpful shopping research assistant. You analyze Reddit community discussions "
    "about products to extract useful consumer insights."
)


class RedditResearcher:

    def _rate_limited_get(self, url, params):
        """GET request to Reddit with minimum 2s spacing between calls."""
        global _last_reddit_request
        now = time.time()
        wait = _MIN_SPACING - (now - _last_reddit_request)
        if wait > 0:
            time.sleep(wait)

        headers = {'User-Agent': USER_AGENT}
        for attempt in range(2):
            _last_reddit_request = time.time()
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', 10))
                    if attempt == 0:
                        print(f"Reddit 429, retrying in {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    return None
                if resp.status_code != 200:
                    print(f"Reddit API error: status={resp.status_code}")
                    return None
                return resp.json()
            except Exception as e:
                print(f"Reddit request error: {e}")
                if attempt == 0:
                    time.sleep(2)
                    continue
                return None
        return None

    def _extract_search_terms(self, products, model):
        """Single batch LLM call to extract brand+model search terms for all products."""
        titles = []
        ranks = []
        for p in products:
            titles.append(p.get('title', ''))
            ranks.append(p.get('rank', 0))

        prompt = (
            "Given these product titles, extract a short search term (brand + model name) "
            "for each that would work well as a Reddit search query. Return ONLY a JSON array "
            "of strings, one per product, in the same order.\n\n"
        )
        for i, title in enumerate(titles):
            prompt += f"{i + 1}. {title}\n"
        prompt += "\nReturn a JSON array like: [\"Brand Model\", \"Brand Model2\", ...]"

        try:
            response_text, _ = generate(model, prompt, SYSTEM_INSTRUCTION, temperature=0)
            # Extract JSON array from response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start >= 0 and end > start:
                terms = json.loads(response_text[start:end])
                if len(terms) == len(products):
                    return {ranks[i]: terms[i] for i in range(len(terms))}
        except Exception as e:
            print(f"Search term extraction error: {e}")

        # Fallback: first 4 words of each title
        result = {}
        for p in products:
            words = p.get('title', '').split()[:4]
            result[p.get('rank', 0)] = ' '.join(words)
        return result

    def _search_reddit(self, query, limit=5, sort='relevance', time_filter='year'):
        """Search Reddit via old.reddit.com/search.json."""
        url = 'https://old.reddit.com/search.json'
        params = {
            'q': query,
            'limit': limit,
            'sort': sort,
            't': time_filter,
            'type': 'link',
        }
        data = self._rate_limited_get(url, params)
        if not data:
            return []

        threads = []
        for child in data.get('data', {}).get('children', []):
            d = child.get('data', {})
            selftext = d.get('selftext', '') or ''
            threads.append({
                'title': d.get('title', ''),
                'subreddit': d.get('subreddit', ''),
                'score': d.get('score', 0),
                'num_comments': d.get('num_comments', 0),
                'permalink': 'https://reddit.com' + d.get('permalink', ''),
                'selftext_snippet': selftext[:300] if selftext else '',
            })
        return threads

    def _synthesize_insight(self, product_title, search_term, threads, model):
        """LLM-synthesize community sentiment from Reddit thread titles and snippets."""
        if not threads:
            return ''

        thread_text = ''
        for i, t in enumerate(threads[:5]):
            thread_text += f"\n{i + 1}. r/{t['subreddit']} ({t['score']} pts, {t['num_comments']} comments): {t['title']}"
            if t['selftext_snippet']:
                thread_text += f"\n   Snippet: {t['selftext_snippet'][:200]}"

        prompt = (
            f'Based on these Reddit discussions about "{search_term}", synthesize 2-3 sentences '
            f'of community sentiment about this product: "{product_title}"\n\n'
            f'Reddit threads:{thread_text}\n\n'
            f'Focus on: common praise, complaints, comparisons to alternatives, and value-for-money '
            f'consensus. If threads are not relevant to this specific product, say so briefly. '
            f'Be concise and factual.'
        )

        try:
            insight_text, _ = generate(model, prompt, SYSTEM_INSTRUCTION, temperature=0)
            return insight_text.strip()
        except Exception as e:
            print(f"Reddit insight synthesis error: {e}")
            return ''

    def research_products_stream(self, products, query, model):
        """
        Generator yielding Reddit research results as they complete.
        Yields tuples: (rank, search_term, threads, insight)
        - rank='_global' for the global query search (insight=None)
        - rank=int for per-product results
        """
        # Step 1: Extract search terms for all products in one batch LLM call
        search_terms = self._extract_search_terms(products, model)

        # Step 2: Global search for the overall query
        global_threads = self._search_reddit(query, limit=8, sort='relevance', time_filter='year')
        yield ('_global', query, global_threads, None)

        # Step 3: Per-product search + synthesis (sequential due to rate limits)
        for product in products:
            rank = product.get('rank', 0)
            title = product.get('title', '')
            search_term = search_terms.get(rank, title.split()[:4])
            if isinstance(search_term, list):
                search_term = ' '.join(search_term)

            threads = self._search_reddit(search_term, limit=5, sort='relevance', time_filter='year')
            insight = self._synthesize_insight(title, search_term, threads, model)
            yield (rank, search_term, threads, insight)
