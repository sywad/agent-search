from typing import List
import json
from modules.llm_client import generate


class QueryPlanner:
    SYSTEM_INSTRUCTION = """You are an intelligent shopping search planner. Given a user's product search query, produce a structured plan that determines:
1. Optimized search queries for Amazon/retailers
2. Reranking instructions specific to this query
3. Whether expert review sources (Wirecutter) would help
4. Whether community discussion (Reddit) would help
5. Whether product images are important for reranking

Return ONLY a JSON object, no other text."""

    FALLBACK_PLAN = {
        "search_queries": [],
        "rerank_instructions": "",
        "enable_wirecutter": False,
        "enable_reddit": False,
        "use_images": True,
        "reasoning": "Fallback plan used due to LLM error."
    }

    def generate_plan(self, user_query: str, model: str = 'gemini-2.5-flash') -> tuple:
        """
        Returns:
            Tuple of (plan_dict, prompt used, LLM response text, debug_info dict)
        """
        prompt = f"""Given the following user search query, generate a complete search plan.

User Query: "{user_query}"

Return a JSON object with these fields:

1. "search_queries": Array of 2-3 optimized Amazon/retailer search queries. Make them specific and diverse to capture different aspects. Use keywords that work well on shopping sites.

2. "rerank_instructions": A short paragraph of query-specific reranking criteria. Focus on what matters most for THIS query (e.g., price constraints, specific features, material quality, brand preferences). Leave empty string "" if the default ranking criteria are sufficient.

3. "enable_wirecutter": Boolean. Set true when expert editorial reviews would help — typically for electronics, appliances, tech accessories, home goods, fitness equipment. Set false for commodity items, fashion, groceries, very niche items Wirecutter wouldn't cover.

4. "enable_reddit": Boolean. Set true when community opinions add value — typically for electronics, headphones, keyboards, monitors, mattresses, shoes, skincare. Set false for simple commodity purchases, basic supplies, fashion where personal taste dominates.

5. "use_images": Boolean. Set true when visual appearance matters for ranking — fashion, furniture, decor, shoes, jewelry, food. Set false when specs/reviews matter more than looks — cables, batteries, supplements, memory cards.

6. "reasoning": 1-2 sentences explaining your decisions.

Guidelines with examples:
- "best noise cancelling headphones" → wirecutter: true, reddit: true, use_images: true (tech product, expert + community reviews valuable)
- "USB-C cable 6ft" → wirecutter: false, reddit: false, use_images: false (commodity, specs matter not looks)
- "summer dress for wedding" → wirecutter: false, reddit: false, use_images: true (fashion, visual appearance critical)
- "desk under 50 dollars" → wirecutter: false, reddit: true, use_images: true (budget constraint important, reddit has budget furniture advice)
- "best robot vacuum" → wirecutter: true, reddit: true, use_images: false (tech product, expert reviews critical)
- "organic face moisturizer" → wirecutter: false, reddit: true, use_images: false (skincare, community reviews important)

Return ONLY the JSON object, no markdown fences or other text.
"""

        try:
            result_text, debug_info = generate(model, prompt, self.SYSTEM_INSTRUCTION)
            llm_output = result_text

            # Extract JSON from response
            text = result_text.strip()
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
                text = text.strip()

            plan = json.loads(text)

            # Validate required fields and apply defaults
            if not isinstance(plan.get('search_queries'), list) or len(plan['search_queries']) < 1:
                plan['search_queries'] = [user_query]
            if len(plan['search_queries']) < 2:
                plan['search_queries'].append(user_query)
            plan['search_queries'] = plan['search_queries'][:3]

            plan.setdefault('rerank_instructions', '')
            plan.setdefault('enable_wirecutter', False)
            plan.setdefault('enable_reddit', False)
            plan.setdefault('use_images', True)
            plan.setdefault('reasoning', '')

            return plan, prompt, llm_output, debug_info

        except Exception as e:
            print(f"Error generating plan: {e}")
            fallback = dict(self.FALLBACK_PLAN)
            fallback['search_queries'] = [user_query]
            return fallback, prompt, f"Error: {str(e)}", {}


# Backward compatibility alias
QueryUnderstanding = QueryPlanner
