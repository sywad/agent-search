from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules import QueryPlanner, AmazonScraper, WalmartScraper, TargetScraper, Reranker, DetailSummarizer, RedditResearcher, WirecutterResearcher, MODELS
from modules.llm_client import generate as llm_generate

# Load environment variables
load_dotenv()

app = Flask(__name__)


def _heuristic_score(product):
    try:
        rating = float(product.get('rating', '0').split()[0]) if product.get('rating') != 'N/A' else 0
        reviews_text = product.get('reviews', '0').replace(',', '')
        reviews = int(reviews_text) if reviews_text.isdigit() else 0
        return rating * __import__('math').log(reviews + 1)
    except:
        return 0


def _run_recommendation_phase(ranked_products, summaries, reddit_insights, query, model, wirecutter_insights=None):
    """Single LLM call to decide which products to recommend for purchase.

    Yields SSE-ready dicts: running, per-product results, and done.
    """
    if wirecutter_insights is None:
        wirecutter_insights = {}
    yield {'step': 'recommending', 'status': 'running'}
    t0 = time.time()

    # Build product descriptions for the prompt
    product_lines = []
    for p in ranked_products:
        rank = p['rank']
        title = p.get('title', 'Unknown')
        price = p.get('price', 'N/A')
        rating = p.get('rating', 'N/A')
        reviews = p.get('reviews', 'N/A')
        line = f'{rank}. [Rank #{rank}] {title} — {price} — {rating} ({reviews} reviews)'
        if rank in summaries:
            line += f'\n   Summary: {summaries[rank]}'
        if rank in wirecutter_insights:
            line += f'\n   Wirecutter: {wirecutter_insights[rank]}'
        if rank in reddit_insights:
            line += f'\n   Reddit: {reddit_insights[rank]}'
        product_lines.append(line)

    products_block = '\n'.join(product_lines)

    prompt = f"""You are a shopping advisor. Based on the research below, decide which products to recommend for purchase. Consider: product quality signals from detail pages, expert editorial reviews from Wirecutter, community sentiment from Reddit, price-to-value ratio, and how well the product matches the query "{query}".

For each product, return recommended: true/false and a 1-sentence reason.
Only recommend products that are genuinely good buys — typically 1-3 out of the set.

Products:
{products_block}

Return ONLY a JSON array (no markdown fences): [{{"rank": 1, "recommended": true, "reason": "..."}}, ...]"""

    try:
        response_text, _ = llm_generate(model, prompt, temperature=0)
        # Parse JSON from response
        text = response_text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
            text = text.rsplit('```', 1)[0]
        recommendations = json.loads(text)
    except Exception as e:
        print(f"Recommendation LLM error: {e}")
        recommendations = []

    for rec in recommendations:
        yield {
            'step': 'recommending',
            'status': 'product',
            'rank': rec.get('rank'),
            'recommended': rec.get('recommended', False),
            'reason': rec.get('reason', '')
        }

    rec_time = round(time.time() - t0, 2)
    yield {'step': 'recommending', 'status': 'done', 'time': rec_time}


@app.route('/')
def index():
    """Render the main search page."""
    return render_template('index.html', models=MODELS)


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests with streaming progress."""
    query = request.form.get('query', '').strip()
    max_results = int(request.form.get('max_results', 60))
    top_n = int(request.form.get('top_n', 20))
    user_rerank_instructions = request.form.get('rerank_instructions', '').strip()
    rerank_edited = request.form.get('rerank_edited', '0') == '1'
    model = request.form.get('model', 'gemini-3.1-flash-lite')
    retailers_raw = request.form.get('retailers', 'amazon').strip()
    selected_retailers = [r.strip() for r in retailers_raw.split(',') if r.strip()]

    # Research & images toggles: "auto" = use planner, "1"/"0" = user override
    wc_setting = request.form.get('enable_wirecutter', 'auto')
    reddit_setting = request.form.get('enable_reddit', 'auto')
    images_setting = request.form.get('use_images', 'auto')

    if not query:
        return jsonify({'error': 'Please provide a search query'}), 400

    if model not in MODELS:
        return jsonify({'error': f'Unknown model: {model}'}), 400

    valid_retailers = {'amazon', 'walmart', 'target'}
    selected_retailers = [r for r in selected_retailers if r in valid_retailers]
    if not selected_retailers:
        selected_retailers = ['amazon']

    max_results = max(10, min(100, max_results))
    top_n = max(5, min(50, top_n))

    SCRAPER_MAP = {
        'amazon': AmazonScraper,
        'walmart': WalmartScraper,
        'target': TargetScraper,
    }

    def generate():
        try:
            planner = QueryPlanner()
            reranker = Reranker()
            timings = {}
            total_start = time.time()

            # Step 1: Query Planning
            yield f"data: {json.dumps({'step': 'query_planning', 'status': 'running'})}\n\n"
            t0 = time.time()
            query_plan, qp_prompt, qp_response, qp_debug = planner.generate_plan(query, model)
            timings['query_planning'] = round(time.time() - t0, 2)

            generated_queries = query_plan['search_queries']

            # Override resolution: planner decides defaults, explicit user choices win
            effective_wirecutter = query_plan.get('enable_wirecutter', False) if wc_setting == 'auto' else (wc_setting == '1')
            effective_reddit = query_plan.get('enable_reddit', False) if reddit_setting == 'auto' else (reddit_setting == '1')
            effective_use_images = query_plan.get('use_images', True) if images_setting == 'auto' else (images_setting == '1')

            # Rerank instructions: if user edited textarea, use theirs; otherwise append planner's to defaults
            planner_rerank = query_plan.get('rerank_instructions', '')
            if rerank_edited and user_rerank_instructions:
                effective_rerank = user_rerank_instructions
            elif planner_rerank:
                effective_rerank = Reranker.DEFAULT_INSTRUCTIONS + "\n\nQuery-specific criteria:\n" + planner_rerank
            else:
                effective_rerank = None

            plan_payload = {
                'search_queries': generated_queries,
                'rerank_instructions': planner_rerank,
                'enable_wirecutter': query_plan.get('enable_wirecutter', False),
                'enable_reddit': query_plan.get('enable_reddit', False),
                'use_images': query_plan.get('use_images', True),
                'reasoning': query_plan.get('reasoning', ''),
                'effective': {
                    'enable_wirecutter': effective_wirecutter,
                    'enable_reddit': effective_reddit,
                    'use_images': effective_use_images,
                }
            }

            yield f"data: {json.dumps({'step': 'query_planning', 'status': 'done', 'time': timings['query_planning'], 'queries': generated_queries, 'plan': plan_payload, 'query_planning': {'input': query, 'output': generated_queries, 'prompt': qp_prompt, 'llm_response': qp_response, 'debug': qp_debug}})}\n\n"

            # Step 2: Scraping — run all selected retailers in parallel
            yield f"data: {json.dumps({'step': 'scraping', 'status': 'running', 'query_count': len(generated_queries), 'retailers': selected_retailers})}\n\n"
            t0 = time.time()

            def _scrape_retailer(retailer_name):
                scraper_cls = SCRAPER_MAP[retailer_name]
                scraper = scraper_cls()
                return retailer_name, scraper.scrape_multiple_queries(generated_queries, max_results)

            all_search_results = {}  # retailer -> {query -> [products]}
            with ThreadPoolExecutor(max_workers=len(selected_retailers)) as executor:
                futures = [executor.submit(_scrape_retailer, r) for r in selected_retailers]
                for future in as_completed(futures):
                    retailer_name, retailer_results = future.result()
                    all_search_results[retailer_name] = retailer_results

            timings['scraping'] = round(time.time() - t0, 2)

            # Merge: top 20 per query per retailer
            all_products = []
            per_query_results = {}
            per_retailer_results = {}

            for retailer_name, retailer_results in all_search_results.items():
                retailer_products = []
                for query_text, products in retailer_results.items():
                    sorted_prods = sorted(products, key=_heuristic_score, reverse=True)
                    top_prods = sorted_prods[:20]
                    all_products.extend(top_prods)
                    retailer_products.extend(products)

                    # Group per_query_results with retailer prefix
                    pq_key = f"[{retailer_name.title()}] {query_text}"
                    per_query_results[pq_key] = products

                per_retailer_results[retailer_name] = retailer_products

            yield f"data: {json.dumps({'step': 'scraping', 'status': 'done', 'time': timings['scraping'], 'product_count': len(all_products)})}\n\n"

            # Emit scraping_data so the frontend can render per-query items immediately
            yield f"data: {json.dumps({'step': 'scraping_data', 'per_query_results': per_query_results, 'generated_queries': generated_queries, 'total_products': len(all_products)})}\n\n"

            if not all_products:
                timings['total'] = round(time.time() - total_start, 2)
                yield f"data: {json.dumps({'step': 'complete', 'data': {'timings': timings, 'model_used': model, 'retailers': selected_retailers, 'error': 'No products found for the given queries'}})}\n\n"
                return

            # Step 3: Reranking (streaming)
            yield f"data: {json.dumps({'step': 'reranking', 'status': 'running', 'product_count': len(all_products)})}\n\n"
            t0 = time.time()

            ranked_products = []
            rerank_prompt = ''
            rerank_response = ''
            rerank_debug = {}
            rank_counter = 0

            for item in reranker.rerank_products_stream(query, all_products, top_n, effective_rerank, model, use_images=effective_use_images):
                if '_metadata' in item:
                    rerank_prompt, rerank_response, rerank_debug = item['_metadata']
                else:
                    rank_counter += 1
                    ranked_products.append(item)
                    yield f"data: {json.dumps({'step': 'reranking', 'status': 'product', 'rank': item['rank'], 'product': item})}\n\n"

            timings['reranking'] = round(time.time() - t0, 2)
            timings['total'] = round(time.time() - total_start, 2)
            yield f"data: {json.dumps({'step': 'reranking', 'status': 'done', 'time': timings['reranking']})}\n\n"

            print(f"Total time: {timings['total']}s | QP: {timings['query_planning']}s | Scraping: {timings['scraping']}s | Reranking: {timings['reranking']}s | Retailers: {selected_retailers}")

            result = {
                'reranker': {
                    'input_count': len(all_products),
                    'output_count': len(ranked_products),
                    'top_n_requested': top_n,
                    'prompt': rerank_prompt,
                    'llm_response': rerank_response,
                    'debug': rerank_debug
                },
                'timings': timings,
                'model_used': model,
                'retailers': selected_retailers
            }

            yield f"data: {json.dumps({'step': 'complete', 'data': result})}\n\n"

            # Step 4: Detail page summaries (cards already visible)
            summaries = {}
            reddit_insights = {}
            if ranked_products:
                yield f"data: {json.dumps({'step': 'summarizing', 'status': 'running'})}\n\n"
                summarizer = DetailSummarizer()
                t0_sum = time.time()
                for rank, summary, s_debug in summarizer.summarize_products_stream(ranked_products, query, model):
                    if summary:
                        summaries[rank] = summary
                        yield f"data: {json.dumps({'step': 'summarizing', 'status': 'product', 'rank': rank, 'summary': summary})}\n\n"
                sum_time = round(time.time() - t0_sum, 2)
                yield f"data: {json.dumps({'step': 'summarizing', 'status': 'done', 'time': sum_time})}\n\n"

            # Step 5: Wirecutter research (optional)
            wirecutter_insights = {}
            if effective_wirecutter and ranked_products:
                yield f"data: {json.dumps({'step': 'wirecutter', 'status': 'running'})}\n\n"
                wc_researcher = WirecutterResearcher()
                t0_wc = time.time()
                for result in wc_researcher.research_products_stream(ranked_products, query, model):
                    rank, search_term, articles, insight = result
                    if rank == '_global':
                        yield f"data: {json.dumps({'step': 'wirecutter', 'status': 'global', 'query': search_term, 'articles': articles})}\n\n"
                    else:
                        if insight:
                            wirecutter_insights[rank] = insight
                        yield f"data: {json.dumps({'step': 'wirecutter', 'status': 'product', 'rank': rank, 'search_term': search_term, 'articles': articles, 'insight': insight})}\n\n"
                wc_time = round(time.time() - t0_wc, 2)
                yield f"data: {json.dumps({'step': 'wirecutter', 'status': 'done', 'time': wc_time})}\n\n"

            # Step 6: Reddit research (optional)
            if effective_reddit and ranked_products:
                yield f"data: {json.dumps({'step': 'reddit_research', 'status': 'running'})}\n\n"
                researcher = RedditResearcher()
                t0_reddit = time.time()
                for result in researcher.research_products_stream(ranked_products, query, model):
                    rank, search_term, threads, insight = result
                    if rank == '_global':
                        yield f"data: {json.dumps({'step': 'reddit_research', 'status': 'global', 'query': search_term, 'threads': threads})}\n\n"
                    else:
                        if insight:
                            reddit_insights[rank] = insight
                        yield f"data: {json.dumps({'step': 'reddit_research', 'status': 'product', 'rank': rank, 'search_term': search_term, 'threads': threads, 'insight': insight})}\n\n"
                reddit_time = round(time.time() - t0_reddit, 2)
                yield f"data: {json.dumps({'step': 'reddit_research', 'status': 'done', 'time': reddit_time})}\n\n"

            # Step 7: Purchase Advisor recommendations
            if ranked_products:
                for event in _run_recommendation_phase(ranked_products, summaries, reddit_insights, query, model, wirecutter_insights):
                    yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"

    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response


@app.route('/rerank', methods=['POST'])
def rerank():
    """Rerank cached products without re-scraping."""
    data = request.get_json()
    query = data.get('query', '').strip()
    products = data.get('products', [])
    top_n = int(data.get('top_n', 20))
    rerank_instructions = data.get('rerank_instructions', '').strip() or None
    model = data.get('model', 'gemini-3.1-flash-lite')
    enable_wirecutter = bool(data.get('enable_wirecutter', False))
    enable_reddit = bool(data.get('enable_reddit', False))
    use_images = data.get('use_images', True)
    if isinstance(use_images, str):
        use_images = use_images == '1' or use_images.lower() == 'true'
    else:
        use_images = bool(use_images)

    if not products:
        return jsonify({'error': 'No cached products to rerank'}), 400
    if model not in MODELS:
        return jsonify({'error': f'Unknown model: {model}'}), 400

    top_n = max(5, min(50, top_n))

    def generate():
        try:
            reranker = Reranker()
            t0 = time.time()
            yield f"data: {json.dumps({'step': 'reranking', 'status': 'running', 'product_count': len(products)})}\n\n"

            ranked_products = []
            rerank_prompt = ''
            rerank_response = ''
            rerank_debug = {}

            for item in reranker.rerank_products_stream(query, products, top_n, rerank_instructions, model, use_images=use_images):
                if '_metadata' in item:
                    rerank_prompt, rerank_response, rerank_debug = item['_metadata']
                else:
                    ranked_products.append(item)
                    yield f"data: {json.dumps({'step': 'reranking', 'status': 'product', 'rank': item['rank'], 'product': item})}\n\n"

            rerank_time = round(time.time() - t0, 2)

            yield f"data: {json.dumps({'step': 'reranking', 'status': 'done', 'time': rerank_time})}\n\n"
            print(f"Rerank only: {rerank_time}s | {len(products)} products -> {len(ranked_products)} ranked | use_images={use_images}")

            result = {
                'reranker': {
                    'input_count': len(products),
                    'output_count': len(ranked_products),
                    'top_n_requested': top_n,
                    'prompt': rerank_prompt,
                    'llm_response': rerank_response,
                    'debug': rerank_debug
                },
                'timings': {'reranking': rerank_time, 'total': rerank_time},
                'model_used': model
            }

            yield f"data: {json.dumps({'step': 'complete', 'data': result})}\n\n"

            # Step 4: Detail page summaries (cards already visible)
            summaries = {}
            reddit_insights = {}
            if ranked_products:
                yield f"data: {json.dumps({'step': 'summarizing', 'status': 'running'})}\n\n"
                summarizer = DetailSummarizer()
                t0_sum = time.time()
                for rank, summary, s_debug in summarizer.summarize_products_stream(ranked_products, query, model):
                    if summary:
                        summaries[rank] = summary
                        yield f"data: {json.dumps({'step': 'summarizing', 'status': 'product', 'rank': rank, 'summary': summary})}\n\n"
                sum_time = round(time.time() - t0_sum, 2)
                yield f"data: {json.dumps({'step': 'summarizing', 'status': 'done', 'time': sum_time})}\n\n"

            # Step: Wirecutter research (optional)
            wirecutter_insights = {}
            if enable_wirecutter and ranked_products:
                yield f"data: {json.dumps({'step': 'wirecutter', 'status': 'running'})}\n\n"
                wc_researcher = WirecutterResearcher()
                t0_wc = time.time()
                for result in wc_researcher.research_products_stream(ranked_products, query, model):
                    rank, search_term, articles, insight = result
                    if rank == '_global':
                        yield f"data: {json.dumps({'step': 'wirecutter', 'status': 'global', 'query': search_term, 'articles': articles})}\n\n"
                    else:
                        if insight:
                            wirecutter_insights[rank] = insight
                        yield f"data: {json.dumps({'step': 'wirecutter', 'status': 'product', 'rank': rank, 'search_term': search_term, 'articles': articles, 'insight': insight})}\n\n"
                wc_time = round(time.time() - t0_wc, 2)
                yield f"data: {json.dumps({'step': 'wirecutter', 'status': 'done', 'time': wc_time})}\n\n"

            # Step: Reddit research (optional)
            if enable_reddit and ranked_products:
                yield f"data: {json.dumps({'step': 'reddit_research', 'status': 'running'})}\n\n"
                researcher = RedditResearcher()
                t0_reddit = time.time()
                for result in researcher.research_products_stream(ranked_products, query, model):
                    rank, search_term, threads, insight = result
                    if rank == '_global':
                        yield f"data: {json.dumps({'step': 'reddit_research', 'status': 'global', 'query': search_term, 'threads': threads})}\n\n"
                    else:
                        if insight:
                            reddit_insights[rank] = insight
                        yield f"data: {json.dumps({'step': 'reddit_research', 'status': 'product', 'rank': rank, 'search_term': search_term, 'threads': threads, 'insight': insight})}\n\n"
                reddit_time = round(time.time() - t0_reddit, 2)
                yield f"data: {json.dumps({'step': 'reddit_research', 'status': 'done', 'time': reddit_time})}\n\n"

            # Step: Purchase Advisor recommendations
            if ranked_products:
                for event in _run_recommendation_phase(ranked_products, summaries, reddit_insights, query, model, wirecutter_insights):
                    yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            print(f"Rerank error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"

    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
