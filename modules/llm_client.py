import os
import time
import base64
import json as _json

MODELS = {
    'gemini-3.1-flash-lite': {'provider': 'gemini', 'id': 'gemini-3.1-flash-lite-preview', 'label': 'Gemini 3.1 Flash Lite'},
    'gemini-2.5-flash': {'provider': 'gemini', 'id': 'gemini-2.5-flash', 'label': 'Gemini 2.5 Flash', 'thinking_budget': 0},
    'gemini-2.5-flash-thinking': {'provider': 'gemini', 'id': 'gemini-2.5-flash', 'label': 'Gemini 2.5 Flash (Thinking)', 'thinking_budget': 1024},
    'gemini-2.0-flash': {'provider': 'gemini', 'id': 'gemini-2.0-flash', 'label': 'Gemini 2.0 Flash'},
    'gemini-3-flash': {'provider': 'gemini', 'id': 'gemini-3-flash-preview', 'label': 'Gemini 3 Flash (Preview)', 'thinking_level': 'LOW'},
    'gemini-2.5-pro': {'provider': 'gemini', 'id': 'gemini-2.5-pro', 'label': 'Gemini 2.5 Pro'},
    'gpt-4o-mini': {'provider': 'openai', 'id': 'gpt-4o-mini', 'label': 'GPT-4o Mini'},
    'gpt-4o': {'provider': 'openai', 'id': 'gpt-4o', 'label': 'GPT-4o'},
    'claude-haiku': {'provider': 'anthropic', 'id': 'claude-haiku-4-5-20251001', 'label': 'Claude Haiku 4.5'},
    'claude-sonnet': {'provider': 'anthropic', 'id': 'claude-sonnet-4-5-20250929', 'label': 'Claude Sonnet 4.5'},
}

_gemini_client = None
_openai_client = None
_anthropic_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=api_key, http_options={'timeout': 60_000})
    return _gemini_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def generate(model_key: str, prompt: str, system_instruction: str = "",
             temperature: float = 0, content_parts: list = None) -> tuple:
    """
    Returns:
        Tuple of (text: str, debug_info: dict)
        debug_info keys: prompt_tokens, output_tokens, thinking_tokens, total_tokens, model_id
    """
    info = MODELS[model_key]
    provider = info['provider']
    model_id = info['id']

    import re
    for attempt in range(2):
        try:
            if provider == 'gemini':
                thinking_budget = info.get('thinking_budget')
                thinking_level = info.get('thinking_level')
                return _generate_gemini_rest(model_id, system_instruction, temperature, thinking_level, thinking_budget, prompt, content_parts)
            elif provider == 'openai':
                return _generate_openai(model_id, system_instruction, temperature, prompt, content_parts)
            elif provider == 'anthropic':
                return _generate_anthropic(model_id, system_instruction, temperature, prompt, content_parts)
        except Exception as e:
            err_str = str(e).lower()
            is_quota = 'quota' in err_str or 'billing' in err_str or 'credit' in err_str
            is_rate_limit = '429' in str(e) or 'rate' in err_str
            if is_rate_limit and not is_quota and attempt == 0:
                match = re.search(r'retry in ([\d.]+)s', str(e))
                wait = min(float(match.group(1)), 15) if match else 10
                print(f"Rate limited ({model_key}), retrying in {wait:.0f}s")
                time.sleep(wait)
            else:
                raise


def _generate_gemini(model_id, system_instruction, temperature, thinking_budget, prompt, content_parts):
    from google.genai import types
    client = _get_gemini()

    config_kwargs = {'temperature': temperature}
    if system_instruction:
        config_kwargs['system_instruction'] = system_instruction
    if thinking_budget is not None:
        config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=thinking_budget)

    if content_parts:
        parts = []
        for part in content_parts:
            if part['type'] == 'text':
                parts.append(types.Part.from_text(text=part['text']))
            elif part['type'] == 'image':
                parts.append(types.Part.from_bytes(data=part['data'], mime_type=part['mime_type']))
        contents = parts
    else:
        contents = prompt

    t0 = time.time()
    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    latency = round(time.time() - t0, 2)

    # === RAW GEMINI DEBUG OUTPUT ===
    print(f"\n{'='*60}")
    print(f"GEMINI RAW RESPONSE — model={model_id} latency={latency}s")
    print(f"{'='*60}")
    print(f"usage_metadata: {response.usage_metadata}")
    print(f"model_version: {getattr(response, 'model_version', 'N/A')}")
    # Print all candidates info
    if response.candidates:
        for i, c in enumerate(response.candidates):
            print(f"candidate[{i}].finish_reason: {c.finish_reason}")
            print(f"candidate[{i}].avg_logprobs: {getattr(c, 'avg_logprobs', 'N/A')}")
            if c.content and c.content.parts:
                for j, part in enumerate(c.content.parts):
                    thought = getattr(part, 'thought', None)
                    if thought:
                        text_preview = part.text[:500] if part.text else ''
                        print(f"candidate[{i}].part[{j}] THOUGHT ({len(part.text)} chars): {text_preview}")
                    else:
                        text_preview = part.text[:300] if part.text else ''
                        print(f"candidate[{i}].part[{j}] TEXT ({len(part.text)} chars): {text_preview}")
    print(f"{'='*60}\n")
    # === END DEBUG ===

    usage = response.usage_metadata
    prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
    output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
    total_tokens = getattr(usage, 'total_token_count', 0) or 0
    thinking_tokens = getattr(usage, 'thoughts_token_count', None)
    if thinking_tokens is None:
        thinking_tokens = max(0, total_tokens - prompt_tokens - output_tokens)

    debug_info = {
        'model_id': model_id,
        'provider': 'gemini',
        'prompt_tokens': prompt_tokens,
        'output_tokens': output_tokens,
        'thinking_tokens': thinking_tokens,
        'total_tokens': total_tokens,
        'api_latency': latency,
    }

    return response.text.strip(), debug_info


def _generate_gemini_rest(model_id, system_instruction, temperature, thinking_level, thinking_budget, prompt, content_parts):
    """Use REST API for all Gemini models. Supports thinkingLevel and thinkingBudget."""
    import requests as req
    api_key = os.getenv('GEMINI_API_KEY')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

    # Build parts
    parts = []
    if content_parts:
        for part in content_parts:
            if part['type'] == 'text':
                parts.append({"text": part['text']})
            elif part['type'] == 'image':
                b64 = base64.b64encode(part['data']).decode('utf-8')
                parts.append({"inline_data": {"mime_type": part['mime_type'], "data": b64}})
    else:
        parts.append({"text": prompt})

    gen_config = {"temperature": temperature}
    if thinking_level:
        gen_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    elif thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": gen_config
    }
    if system_instruction:
        body["system_instruction"] = {"parts": [{"text": system_instruction}]}

    t0 = time.time()
    resp = req.post(url, json=body, timeout=60)
    latency = round(time.time() - t0, 2)

    data = resp.json()
    if 'error' in data:
        raise Exception(f"Gemini API error: {data['error']['message']}")

    usage = data.get('usageMetadata', {})
    prompt_tokens = usage.get('promptTokenCount', 0)
    output_tokens = usage.get('candidatesTokenCount', 0)
    thinking_tokens = usage.get('thoughtsTokenCount', 0) or 0
    total_tokens = usage.get('totalTokenCount', 0)

    # Extract text and log raw response
    text_parts = []
    candidates = data.get('candidates', [])
    print(f"\n{'='*60}")
    thinking_info = f"thinkingLevel={thinking_level}" if thinking_level else f"thinkingBudget={thinking_budget}"
    print(f"GEMINI RAW RESPONSE (REST) — model={model_id} latency={latency}s {thinking_info}")
    print(f"{'='*60}")
    print(f"usageMetadata: prompt={prompt_tokens} output={output_tokens} thinking={thinking_tokens} total={total_tokens}")
    if candidates:
        c = candidates[0]
        print(f"finishReason: {c.get('finishReason')}")
        for j, p in enumerate(c.get('content', {}).get('parts', [])):
            if p.get('thought'):
                print(f"part[{j}] THOUGHT ({len(p.get('text',''))} chars): {p['text'][:500]}")
            elif 'text' in p:
                text_parts.append(p['text'])
                print(f"part[{j}] TEXT ({len(p['text'])} chars): {p['text'][:300]}")
    print(f"{'='*60}\n")

    result_text = ''.join(text_parts).strip()

    debug_info = {
        'model_id': model_id,
        'provider': 'gemini',
        'prompt_tokens': prompt_tokens,
        'output_tokens': output_tokens,
        'thinking_tokens': thinking_tokens,
        'total_tokens': total_tokens,
        'api_latency': latency,
        'thinking_level': thinking_level,
    }

    return result_text, debug_info


def _generate_openai(model_id, system_instruction, temperature, prompt, content_parts):
    client = _get_openai()
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if content_parts:
        content = []
        for part in content_parts:
            if part['type'] == 'text':
                content.append({"type": "text", "text": part['text']})
            elif part['type'] == 'image':
                b64 = base64.b64encode(part['data']).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{part['mime_type']};base64,{b64}", "detail": "low"}
                })
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
    )
    latency = round(time.time() - t0, 2)

    usage = response.usage
    debug_info = {
        'model_id': model_id,
        'provider': 'openai',
        'prompt_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
        'output_tokens': getattr(usage, 'completion_tokens', 0) or 0,
        'thinking_tokens': 0,
        'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
        'api_latency': latency,
    }

    return response.choices[0].message.content.strip(), debug_info


class StreamResult:
    """Wraps a chunk generator for streaming LLM output."""
    def __init__(self, chunk_iter):
        self._iter = chunk_iter
        self.full_text = ""
        self.debug_info = {}

    def __iter__(self):
        for chunk in self._iter:
            if isinstance(chunk, dict):
                # metadata sentinel from the generator
                self.debug_info = chunk
            else:
                self.full_text += chunk
                yield chunk
        # if no metadata was sent, debug_info stays {}


def generate_stream(model_key: str, prompt: str, system_instruction: str = "",
                    temperature: float = 0, content_parts: list = None) -> StreamResult:
    """Streaming variant of generate(). Returns a StreamResult whose iterator yields text chunks.
    After iteration, .full_text and .debug_info are populated.
    For non-Gemini providers, falls back to a single-chunk stream wrapping blocking generate().
    """
    info = MODELS[model_key]
    provider = info['provider']
    model_id = info['id']

    if provider == 'gemini':
        thinking_budget = info.get('thinking_budget')
        thinking_level = info.get('thinking_level')
        gen = _stream_gemini_rest(model_id, system_instruction, temperature,
                                  thinking_level, thinking_budget, prompt, content_parts)
        return StreamResult(gen)
    else:
        # Fallback: call blocking generate(), yield all at once
        def _single_chunk():
            text, dbg = generate(model_key, prompt, system_instruction,
                                 temperature=temperature, content_parts=content_parts)
            yield text
            yield dbg  # metadata sentinel
        return StreamResult(_single_chunk())


def _stream_gemini_rest(model_id, system_instruction, temperature,
                        thinking_level, thinking_budget, prompt, content_parts):
    """Streaming REST call to Gemini. Yields text chunks, then a debug_info dict."""
    import requests as req
    api_key = os.getenv('GEMINI_API_KEY')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:streamGenerateContent?alt=sse&key={api_key}"

    # Build parts (same as non-streaming)
    parts = []
    if content_parts:
        for part in content_parts:
            if part['type'] == 'text':
                parts.append({"text": part['text']})
            elif part['type'] == 'image':
                b64 = base64.b64encode(part['data']).decode('utf-8')
                parts.append({"inline_data": {"mime_type": part['mime_type'], "data": b64}})
    else:
        parts.append({"text": prompt})

    gen_config = {"temperature": temperature}
    if thinking_level:
        gen_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    elif thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": gen_config
    }
    if system_instruction:
        body["system_instruction"] = {"parts": [{"text": system_instruction}]}

    t0 = time.time()
    resp = req.post(url, json=body, timeout=120, stream=True)

    usage = {}
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode('utf-8', errors='replace')
        if not line.startswith('data: '):
            continue
        json_str = line[6:]
        try:
            data = _json.loads(json_str)
        except _json.JSONDecodeError:
            continue

        # Capture usage from every chunk (last one wins with full totals)
        if 'usageMetadata' in data:
            usage = data['usageMetadata']

        # Extract text parts, skip thoughts
        for candidate in data.get('candidates', []):
            for p in candidate.get('content', {}).get('parts', []):
                if p.get('thought'):
                    continue
                text = p.get('text', '')
                if text:
                    yield text

    latency = round(time.time() - t0, 2)
    prompt_tokens = usage.get('promptTokenCount', 0)
    output_tokens = usage.get('candidatesTokenCount', 0)
    thinking_tokens = usage.get('thoughtsTokenCount', 0) or 0
    total_tokens = usage.get('totalTokenCount', 0)

    print(f"GEMINI STREAM done — model={model_id} latency={latency}s tokens=p{prompt_tokens}/o{output_tokens}/t{thinking_tokens}")

    yield {
        'model_id': model_id,
        'provider': 'gemini',
        'prompt_tokens': prompt_tokens,
        'output_tokens': output_tokens,
        'thinking_tokens': thinking_tokens,
        'total_tokens': total_tokens,
        'api_latency': latency,
        'thinking_level': thinking_level,
    }


def _generate_anthropic(model_id, system_instruction, temperature, prompt, content_parts):
    client = _get_anthropic()

    if content_parts:
        content = []
        for part in content_parts:
            if part['type'] == 'text':
                content.append({"type": "text", "text": part['text']})
            elif part['type'] == 'image':
                b64 = base64.b64encode(part['data']).decode('utf-8')
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": part['mime_type'], "data": b64}
                })
    else:
        content = prompt

    kwargs = {
        "model": model_id,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }
    if system_instruction:
        kwargs["system"] = system_instruction

    t0 = time.time()
    response = client.messages.create(**kwargs)
    latency = round(time.time() - t0, 2)

    debug_info = {
        'model_id': model_id,
        'provider': 'anthropic',
        'prompt_tokens': getattr(response.usage, 'input_tokens', 0) or 0,
        'output_tokens': getattr(response.usage, 'output_tokens', 0) or 0,
        'thinking_tokens': 0,
        'total_tokens': (getattr(response.usage, 'input_tokens', 0) or 0) + (getattr(response.usage, 'output_tokens', 0) or 0),
        'api_latency': latency,
    }

    return response.content[0].text.strip(), debug_info
