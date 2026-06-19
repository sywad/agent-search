"""Search tool for the voice shopping agent.

Bridges Gemini Live function calls to the repo's existing retrieval modules:
scrape one or more retailers, then blend-rank with Gemini. Returns a compact
result for the model to speak about, plus richer cards for the browser.

The model chooses which retailers to search (default Amazon) based on the
conversation — e.g. the user can say "also check Walmart". The shared modules are
synchronous (requests / thread pools), so `run_search` offloads them to a worker
thread to keep the async server's event loop free.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from google.genai import types

from modules.scraper import AmazonScraper
from modules.walmart_scraper import WalmartScraper
from modules.target_scraper import TargetScraper
from modules.reranker import Reranker
from modules.query_understanding import QueryPlanner
from modules.detail_summarizer import DetailSummarizer

# Text model for ranking + query expansion (the Live model id is audio-only and
# not valid here).
RERANK_MODEL = "gemini-3.1-flash-lite"

# Scrape breadth PER expanded query, tuned for voice latency, not exhaustiveness.
SCRAPE_RESULTS = 24
# Cap on how many search strings we expand a request into.
EXPAND_MAX = 3

SCRAPER_MAP = {
    "amazon": AmazonScraper,
    "walmart": WalmartScraper,
    "target": TargetScraper,
}
RETAILER_LABELS = {"amazon": "Amazon", "walmart": "Walmart", "target": "Target"}
DEFAULT_RETAILERS = ["amazon"]


SEARCH_PRODUCTS = types.FunctionDeclaration(
    name="search_products",
    description=(
        "Search one or more retailers for products and return a short ranked list. "
        "Call this once you know what the user wants to buy. Include any budget or "
        "preferences they mentioned so the results can be filtered and ranked."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "query": types.Schema(
                type="STRING",
                description="What to search for, e.g. 'noise cancelling headphones'.",
            ),
            "retailers": types.Schema(
                type="ARRAY",
                items=types.Schema(type="STRING", enum=["amazon", "walmart", "target"]),
                description=(
                    "Which stores to search. Defaults to Amazon only if omitted. "
                    "Include walmart and/or target when the user asks to compare "
                    "stores or names a specific one."
                ),
            ),
            "min_price": types.Schema(
                type="NUMBER", description="Minimum price in USD, if the user gave one."
            ),
            "max_price": types.Schema(
                type="NUMBER", description="Maximum budget in USD, if the user gave one."
            ),
            "features": types.Schema(
                type="STRING",
                description=(
                    "Specific features, brands, or use-case the user mentioned, e.g. "
                    "'wireless, for running, long battery life'."
                ),
            ),
        },
        required=["query"],
    ),
)


GET_PRODUCT_DETAILS = types.FunctionDeclaration(
    name="get_product_details",
    description=(
        "Fetch deeper info from a product's detail page (specs, what's included, "
        "standout review points, who it's best for) for ONE product from the most "
        "recent search results. Use it when the user asks for more detail about a "
        "specific item, or to compare two items closely (call it once per item)."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "rank": types.Schema(
                type="INTEGER",
                description="Rank number of the product from the last results (1 = top pick).",
            ),
        },
        required=["rank"],
    ),
)

HIGHLIGHT_PRODUCT = types.FunctionDeclaration(
    name="highlight_product",
    description=(
        "Scroll to and visually highlight one product card on screen — use it when you "
        "mention a specific item so the user can find and tap it to open it."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "rank": types.Schema(
                type="INTEGER", description="Rank number of the product to highlight."
            ),
        },
        required=["rank"],
    ),
)


ARRANGE_RESULTS = types.FunctionDeclaration(
    name="arrange_results",
    description=(
        "Re-arrange the product cards already on screen by giving the ranks to show, "
        "in the order to display them. One tool for sorting, filtering, and limiting: "
        "include all ranks reordered to SORT (e.g. 'by brand', 'cheapest first'), "
        "include only some ranks to FILTER (e.g. 'only Nike'), or include just the "
        "first few to LIMIT (e.g. 'show the top 3'). Only ranks from the most recent "
        "search are valid; this does not fetch anything new."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "order": types.Schema(
                type="ARRAY",
                items=types.Schema(type="INTEGER"),
                description="Ranks to display, in display order. Omit a rank to hide it.",
            ),
            "title": types.Schema(
                type="STRING",
                description="Optional short heading for this view, e.g. 'Nike sneakers' or 'Sorted by brand'.",
            ),
        },
        required=["order"],
    ),
)


def tool() -> types.Tool:
    return types.Tool(function_declarations=[
        SEARCH_PRODUCTS, GET_PRODUCT_DETAILS, HIGHLIGHT_PRODUCT, ARRANGE_RESULTS])


def _parse_price(value) -> float | None:
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _clean_price(value) -> str:
    """Normalize prices to a bare amount (no '$'); retailers differ — Amazon has
    no symbol, Walmart/Target include one. Returns 'N/A' when unknown."""
    if value in (None, "", "N/A"):
        return "N/A"
    return str(value).replace("$", "").strip()


def _normalize_retailers(retailers) -> list:
    """Validate/dedupe the model's retailer list; fall back to Amazon."""
    if not retailers:
        return list(DEFAULT_RETAILERS)
    seen = []
    for r in retailers:
        key = str(r).strip().lower()
        if key in SCRAPER_MAP and key not in seen:
            seen.append(key)
    return seen or list(DEFAULT_RETAILERS)


def _build_instructions(min_price, max_price, features, multi_retailer) -> str:
    extra = []
    if max_price:
        extra.append(f"Budget: at or under ${max_price:.0f} — strongly favor cheaper options.")
    if min_price:
        extra.append(f"Price should be at least ${min_price:.0f}.")
    if features:
        extra.append(f"The user specifically wants: {features}. Prioritize products that match.")
    if multi_retailer:
        extra.append("Include the best matches from EACH store, not just one.")
    if not extra:
        return None
    return Reranker.DEFAULT_INSTRUCTIONS + "\n\nQuery-specific criteria:\n" + " ".join(extra)


def _expand_queries(query: str, features: str) -> list:
    """Reuse the QueryPlanner to turn one request into 2-3 diverse search strings
    for better recall. Always keeps the raw query; falls back to it on error."""
    seed = f"{query}. {features}".strip() if features else query
    try:
        plan, _, _, _ = QueryPlanner().generate_plan(seed, RERANK_MODEL)
        queries = [q.strip() for q in plan.get("search_queries", []) if q and q.strip()]
    except Exception as e:
        print(f"query expansion failed: {e}")
        queries = []
    if query not in queries:
        queries = [query] + queries
    return queries[:EXPAND_MAX] or [query]


def _scrape_one(retailer: str, queries: list) -> list:
    try:
        results = SCRAPER_MAP[retailer]().scrape_multiple_queries(queries, SCRAPE_RESULTS)
        return [p for prods in results.values() for p in prods]
    except Exception as e:
        print(f"{retailer} scrape error: {e}")
        return []


def _search_sync(query, retailers, min_price, max_price, features) -> list:
    """Blocking multi-retailer scrape + blended rerank. Runs in a worker thread."""
    queries = _expand_queries(query, features)
    print(f"search_products: '{query}' -> queries={queries} retailers={retailers}")
    products = []
    with ThreadPoolExecutor(max_workers=len(retailers)) as ex:
        for prods in ex.map(lambda r: _scrape_one(r, queries), retailers):
            products.extend(prods)

    # Hard price filter (the reranker only soft-prefers price). Fall back to the
    # unfiltered set if the filter would leave us with nothing to show.
    if min_price or max_price:
        kept = []
        for p in products:
            price = _parse_price(p.get("price"))
            if price is None:
                kept.append(p)
                continue
            if min_price and price < min_price:
                continue
            if max_price and price > max_price:
                continue
            kept.append(p)
        products = kept or products

    if not products:
        return []

    # Show a fuller grid; even more when blending stores so each source shows.
    top_n = 9 if len(retailers) == 1 else 12
    instructions = _build_instructions(min_price, max_price, features, multi_retailer=len(retailers) > 1)
    ranked, _, _, _ = Reranker().rerank_products(
        query, products, top_n=top_n, custom_instructions=instructions,
        model=RERANK_MODEL, use_images=False,
    )
    return ranked


async def run_search(args: dict) -> tuple[dict, list]:
    """Execute a search_products tool call.

    Returns (compact_result_for_model, cards_for_browser).
    """
    query = (args.get("query") or "").strip()
    retailers = _normalize_retailers(args.get("retailers"))
    min_price = args.get("min_price")
    max_price = args.get("max_price")
    features = (args.get("features") or "").strip()

    if not query:
        return {"error": "no query provided"}, []

    ranked = await asyncio.to_thread(_search_sync, query, retailers, min_price, max_price, features)
    store_labels = [RETAILER_LABELS[r] for r in retailers]

    if not ranked:
        return {
            "count": 0,
            "retailers": store_labels,
            "message": f"No results came back from {', '.join(store_labels)}.",
            "instruction": (
                "Tell the user nothing came up — the store may have temporarily "
                "blocked the request or nothing matched. Offer to try again or "
                "refine the search. Keep it short and spoken."
            ),
        }, []

    def label(p):
        return RETAILER_LABELS.get(p.get("source", "amazon"), p.get("source", "amazon"))

    # Compact list the model speaks from — no URLs/images, short fields.
    compact = {
        "count": len(ranked),
        "retailers": store_labels,
        "products": [
            {
                "rank": p.get("rank"),
                "store": label(p),
                "title": (p.get("title", "")[:90]),
                "price": f"${_clean_price(p.get('price'))}" if _clean_price(p.get("price")) != "N/A" else "price n/a",
                "rating": p.get("rating", "N/A"),
                "reviews": p.get("reviews", "0"),
                "why": p.get("rank_reason", ""),
            }
            for p in ranked
        ],
        "instruction": (
            "Briefly tell the user about the top 1-3 picks and why. If more than one "
            "store was searched, mention which store a pick is from. Keep it short and spoken."
        ),
    }

    # Full cards for the browser UI.
    cards = [
        {
            "rank": p.get("rank"),
            "store": label(p),
            "source": p.get("source", "amazon"),
            "title": p.get("title", ""),
            "price": _clean_price(p.get("price")),
            "rating": p.get("rating", "N/A"),
            "reviews": p.get("reviews", "0"),
            "image": p.get("image", ""),
            "url": p.get("url", ""),
            "why": p.get("rank_reason", ""),
        }
        for p in ranked
    ]

    return compact, cards


async def fetch_details(card: dict, query: str) -> str:
    """Fetch + summarize one product's detail page. Returns a short, spoken-friendly
    summary (empty string if the page couldn't be read)."""
    summarizer = DetailSummarizer()
    seed = query or card.get("title", "")
    _, summary, _ = await asyncio.to_thread(
        summarizer._fetch_and_summarize, card, seed, RERANK_MODEL
    )
    return summary or ""
