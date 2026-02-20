from typing import List, Dict
import json
import math
import requests
from concurrent.futures import ThreadPoolExecutor
from modules.llm_client import generate, generate_stream


class Reranker:
    DEFAULT_INSTRUCTIONS = """Rank the products based on:
1. Query relevance — how well does this product match what the user is actually looking for? Consider the product's purpose, features, and category fit (most important)
2. Visual match — does the product image confirm it is the right type of item for the query?
3. Product quality — rating and reviews as secondary signals
4. Retailer diversity — include the best products from EACH retailer/source, not just one

Focus your reasoning on WHY each product is a good match for the user's specific query. Explain what features, attributes, or use-case alignment make it relevant. Do NOT just cite review counts or ratings — those are tiebreakers, not reasons."""

    SYSTEM_INSTRUCTION = """You are an expert product ranker for e-commerce. You will receive a user query, product images labeled by ID, and product metadata. Rank them according to the provided instructions. Always return ONLY a JSON array, no other text."""

    SYSTEM_INSTRUCTION_TEXT_ONLY = """You are an expert product ranker for e-commerce. You will receive a user query and product metadata. Rank them according to the provided instructions. Always return ONLY a JSON array, no other text."""

    def _prepare_rerank_inputs(self, original_query, products, top_n, custom_instructions, use_images=True):
        """Shared setup: dedup, cap, download images, build prompt & content_parts.
        Returns (products, content_parts, prompt_for_log, top_n_effective)."""
        # Deduplicate products by (source, product_id), then by normalized title
        unique_products = {}
        for product in products:
            source = product.get('source', 'amazon')
            product_id = product.get('product_id', '') or product.get('asin', '')
            if product_id:
                dedup_key = (source, product_id)
            else:
                dedup_key = (source, product.get('url', ''))
            if dedup_key not in unique_products:
                unique_products[dedup_key] = product

        # Second pass: deduplicate by normalized title prefix (catches color/size
        # variants with different ASINs but near-identical titles like
        # "...Earbuds, Black" vs "...Earbuds, White")
        seen_titles = {}
        deduped = {}
        for key, product in unique_products.items():
            norm_title = product.get('title', '').strip().lower()
            if norm_title:
                # Use first 90% of title as dedup key to catch trailing variants
                prefix_len = max(20, int(len(norm_title) * 0.9))
                title_key = norm_title[:prefix_len]
                if title_key in seen_titles:
                    continue
                seen_titles[title_key] = key
            deduped[key] = product

        products = list(deduped.values())

        # Cap at 60 products for LLM, allocated evenly across retailers
        max_for_llm = 60
        if len(products) > max_for_llm:
            by_source = {}
            for p in products:
                by_source.setdefault(p.get('source', 'amazon'), []).append(p)
            n_sources = len(by_source)
            per_source = max_for_llm // n_sources
            selected = []
            for source, source_products in by_source.items():
                top = sorted(source_products, key=self._heuristic_score, reverse=True)[:per_source]
                selected.extend(top)
            selected_set = set(id(p) for p in selected)
            remaining = [p for p in products if id(p) not in selected_set]
            remaining.sort(key=self._heuristic_score, reverse=True)
            selected.extend(remaining[:max_for_llm - len(selected)])
            products = selected

        source_counts = {}
        for p in products:
            s = p.get('source', 'amazon')
            source_counts[s] = source_counts.get(s, 0) + 1
        print(f"Reranker: {len(unique_products)} unique -> {len(products)} sent to LLM (top_n={top_n}) | per source: {source_counts}")

        # Download product images in parallel (skip if use_images=False)
        if use_images:
            image_results = self._download_images(products)
            downloaded = sum(1 for img in image_results if img is not None)
            print(f"Reranker: downloaded {downloaded} / {len(products)} images")
        else:
            image_results = [None] * len(products)
            print(f"Reranker: skipping image downloads (use_images=False)")

        # Prepare compact product information for LLM
        product_summaries = []
        for idx, product in enumerate(products):
            title = product.get('title', 'N/A')
            if len(title) > 100:
                title = title[:100] + '...'
            summary = {
                'id': idx,
                'title': title,
                'price': product.get('price', 'N/A'),
                'rating': product.get('rating', 'N/A'),
                'reviews': product.get('reviews', '0'),
                'source': product.get('source', 'amazon')
            }
            product_summaries.append(summary)

        instructions = custom_instructions.strip() if custom_instructions else self.DEFAULT_INSTRUCTIONS.strip()

        # Build interleaved multimodal content
        content_parts = []
        for idx, product in enumerate(products):
            title = product.get('title', 'N/A')
            if len(title) > 60:
                title = title[:60] + '...'
            content_parts.append({'type': 'text', 'text': f'Product {idx}: {title}'})
            img = image_results[idx]
            if img is not None:
                content_parts.append({'type': 'image', 'data': img['data'], 'mime_type': img['mime_type']})

        top_n_effective = min(top_n, len(products))
        main_prompt = f"""{instructions}

User Query: "{original_query}"

Products metadata:
{json.dumps(product_summaries, separators=(',', ':'))}

Return the top {top_n_effective} as JSON: [{{"id":0,"reason":"why this product matches the query"}},{{"id":5,"reason":"why this product matches the query"}}]
Each reason should be 1-2 sentences explaining why the product is a good match for the query."""

        content_parts.append({'type': 'text', 'text': main_prompt})

        return products, content_parts, main_prompt, top_n_effective, use_images

    def rerank_products(self, original_query: str, products: List[Dict], top_n: int = 20, custom_instructions: str = None, model: str = 'gemini-2.5-flash', use_images: bool = True) -> tuple[List[Dict], str, str]:
        if not products:
            return [], "", "No products to rank"

        products, content_parts, prompt_for_log, top_n_eff, imgs = self._prepare_rerank_inputs(
            original_query, products, top_n, custom_instructions, use_images=use_images)

        sys_instruction = self.SYSTEM_INSTRUCTION if imgs else self.SYSTEM_INSTRUCTION_TEXT_ONLY
        try:
            result_text, debug_info = generate(model, prompt_for_log, sys_instruction,
                                               temperature=0, content_parts=content_parts)
            llm_output = result_text

            # Extract JSON array from response
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            ranked_data = json.loads(result_text)

            # Reorder products based on ranking with reasons
            ranked_products = []
            for rank, item in enumerate(ranked_data, 1):
                product_id = item.get('id') if isinstance(item, dict) else item
                reason = item.get('reason', 'Ranked by AI') if isinstance(item, dict) else 'Ranked by AI'

                if product_id < len(products):
                    product = products[product_id].copy()
                    product['rank'] = rank
                    product['relevance_score'] = round((len(ranked_data) - rank + 1) / len(ranked_data) * 100, 2)
                    product['rank_reason'] = reason
                    ranked_products.append(product)

            return ranked_products, prompt_for_log, llm_output, debug_info

        except Exception as e:
            print(f"Error reranking products: {e}")
            import traceback
            traceback.print_exc()
            fallback_products = self._fallback_ranking(products, top_n)
            return fallback_products, prompt_for_log, f"Error: {str(e)} (used fallback ranking)", {}

    @staticmethod
    def _parse_incremental_json_array(text_buffer):
        """Find complete JSON objects in a growing text buffer that looks like [{...},{...},...].
        Returns (list_of_parsed_objects, remaining_unparsed_offset)."""
        objects = []
        i = 0
        n = len(text_buffer)

        # Skip until first '['
        while i < n and text_buffer[i] != '[':
            i += 1
        if i >= n:
            return objects, 0
        i += 1  # skip '['

        while i < n:
            # Skip whitespace and commas
            while i < n and text_buffer[i] in ' \t\r\n,':
                i += 1
            if i >= n or text_buffer[i] == ']':
                break
            if text_buffer[i] != '{':
                i += 1
                continue

            # Try to find matching closing brace
            start = i
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < n:
                ch = text_buffer[j]
                if escape:
                    escape = False
                    j += 1
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    j += 1
                    continue
                if ch == '"':
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            # Complete object found
                            obj_str = text_buffer[start:j + 1]
                            try:
                                objects.append(json.loads(obj_str))
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
                j += 1
            else:
                # Incomplete object — stop here
                break

        return objects, i

    def rerank_products_stream(self, original_query: str, products: List[Dict],
                               top_n: int = 20, custom_instructions: str = None,
                               model: str = 'gemini-2.5-flash', use_images: bool = True):
        """Generator that yields ranked product dicts one at a time as the LLM streams.
        Final yield is a sentinel: {'_metadata': (prompt, full_text, debug_info)}."""
        if not products:
            yield {'_metadata': ('', '', {})}
            return

        products, content_parts, prompt_for_log, top_n_eff, imgs = self._prepare_rerank_inputs(
            original_query, products, top_n, custom_instructions, use_images=use_images)

        sys_instruction = self.SYSTEM_INSTRUCTION if imgs else self.SYSTEM_INSTRUCTION_TEXT_ONLY
        try:
            stream = generate_stream(model, prompt_for_log, sys_instruction,
                                     temperature=0, content_parts=content_parts)

            text_buffer = ""
            yielded_count = 0
            rank = 1
            total_expected = top_n_eff

            for chunk in stream:
                text_buffer += chunk
                # Try to parse new complete objects
                parsed, _ = self._parse_incremental_json_array(text_buffer)
                while yielded_count < len(parsed) and rank <= total_expected:
                    item = parsed[yielded_count]
                    product_id = item.get('id') if isinstance(item, dict) else item
                    reason = item.get('reason', 'Ranked by AI') if isinstance(item, dict) else 'Ranked by AI'
                    if isinstance(product_id, int) and product_id < len(products):
                        product = products[product_id].copy()
                        product['rank'] = rank
                        product['relevance_score'] = round((total_expected - rank + 1) / total_expected * 100, 2)
                        product['rank_reason'] = reason
                        yield product
                        rank += 1
                    yielded_count += 1

            # Yield metadata sentinel
            yield {'_metadata': (prompt_for_log, stream.full_text, stream.debug_info)}

        except Exception as e:
            print(f"Error in streaming rerank: {e}")
            import traceback
            traceback.print_exc()
            fallback_products = self._fallback_ranking(products, top_n)
            for product in fallback_products:
                yield product
            yield {'_metadata': (prompt_for_log, f"Error: {str(e)} (used fallback ranking)", {})}

    def _download_images(self, products: List[Dict]) -> list:
        """Download product images in parallel. Returns list aligned with products (None for failures)."""
        image_urls = [p.get('image', '') or None for p in products]

        def _fetch(url):
            if not url:
                return None
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200 and len(resp.content) > 100:
                    ct = resp.headers.get('content-type', 'image/jpeg')
                    mime = ct.split(';')[0].strip()
                    return {'data': resp.content, 'mime_type': mime}
            except:
                pass
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(_fetch, image_urls))

    @staticmethod
    def _heuristic_score(product):
        try:
            rating = float(product.get('rating', '0').split()[0]) if product.get('rating') != 'N/A' else 0
            reviews_text = product.get('reviews', '0').replace(',', '')
            reviews = int(reviews_text) if reviews_text.isdigit() else 0
            return rating * math.log(reviews + 1)
        except:
            return 0

    def _fallback_ranking(self, products: List[Dict], top_n: int) -> List[Dict]:
        sorted_products = sorted(products, key=self._heuristic_score, reverse=True)
        ranked = []
        for rank, product in enumerate(sorted_products[:top_n], 1):
            product_copy = product.copy()
            product_copy['rank'] = rank
            product_copy['relevance_score'] = round((top_n - rank + 1) / top_n * 100, 2)
            ranked.append(product_copy)
        return ranked
