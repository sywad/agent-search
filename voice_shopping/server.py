"""Voice shopping agent server.

FastAPI server that relays tap-to-talk audio between the browser and a Gemini
Live session, and runs the `search_products` tool (scrape + rank) on the model's
behalf. Run from the repo root so the shared `modules/` package is importable:

    venv/bin/uvicorn voice_shopping.server:app --reload --port 5050
"""
import os
import time
import uuid
import asyncio
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

try:
    from websockets.exceptions import ConnectionClosed
except Exception:  # pragma: no cover - websockets ships with uvicorn[standard]
    class ConnectionClosed(Exception):
        pass

# Gemini Live closes idle/expired sessions with these — treat as a normal end of
# session (client reconnects) rather than a crash.
SESSION_END_ERRORS = (genai_errors.APIError, ConnectionClosed)

from voice_shopping import agent
from voice_shopping import eventlog

load_dotenv()

# Newest general-purpose Live model. Half-cascade audio is more reliable for the
# function-calling we add in Milestone 2 than the native-audio dialog variants.
MODEL = "gemini-3.1-flash-live-preview"

# Gemini Live audio contract: input is 16 kHz PCM16, output is 24 kHz PCM16 mono.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

SYSTEM_PROMPT = """You are a friendly, concise voice shopping assistant for Amazon.
Always speak in English (US), regardless of anything else.
You are talking to the user out loud, so keep replies short and conversational —
one or two sentences, no lists or markdown.

Your job: understand what they want, search, and recommend — all by voice.

Understanding the request:
- Infer the use-case and a sensible scope from what they say. Ask a clarifying
  question ONLY if you genuinely can't search well without it (for example, no
  budget at all for a category whose price varies wildly). Ask at most ONE short
  question, then search. Don't ask about budget for cheap everyday items.

Searching:
- Call `search_products` with the query plus any budget (min_price/max_price) and
  features/preferences they mentioned.
- Search Amazon by default. Only add Walmart or Target to `retailers` when the user
  asks to compare stores or names one. Don't ask which store unless they bring it up.
- About 15 products are shown by default; pass `limit` (up to 30) when the user wants
  more or fewer ("show me lots of options" → higher; "just the best few" → ~3-5).

After results come back:
- Briefly recommend the top one or two picks and why they fit — a sentence or two,
  spoken naturally. If you searched more than one store, say which store a pick is
  from. The cards are on screen, so don't read out every spec or any links.

Follow-ups (important):
- If they want changes ("cheaper", "only Sony", "something smaller", "add Walmart"),
  call `search_products` again with the updated constraints.
- For a quick question about a pick, answer from what you already found.
- For DEEPER detail ("tell me more about the second one", "does it have X", "which
  of these is better for travel"), call `get_product_details` for that item (or for
  each item you're comparing) to get real specs and review highlights, then summarize.
- When you talk about a specific item, call `highlight_product` with its rank so its
  card is highlighted on screen for the user to tap. Refer to items by rank/store.
- To reorder, filter, or trim the cards already shown ("sort by brand", "only Nike",
  "cheapest first", "just the top 3"), call `arrange_results` with the ranks in the
  order to display — you decide the order from what you know about the items. This
  reuses the current results; don't search again just to re-sort."""

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Voice Shopping Agent")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

_client = None


def get_client() -> genai.Client:
    """Lazily build the Gemini client so import never fails on a missing key."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    return _client


def live_config(resume_handle: str = None) -> types.LiveConnectConfig:
    """Audio-out config with hands-free turn detection and transcripts.

    Automatic VAD (the default — we no longer disable it) lets Gemini detect when
    the user starts/stops talking and supports barge-in, so the client just streams
    the mic continuously while the conversation is active. `session_resumption`
    makes the session resumable so follow-ups survive an idle disconnect.
    """
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        # Pin output language — Live otherwise sometimes drifts (e.g. to Spanish).
        speech_config=types.SpeechConfig(language_code="en-US"),
        tools=[agent.tool()],
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


def _lookup(state, rank):
    """Resolve a rank (int/float/str from the model) to a stored result card."""
    results = state.get("results") or {}
    try:
        return results.get(int(rank))
    except (TypeError, ValueError):
        return None


async def handle_tool_call(ws: WebSocket, session, tool_call, state, session_id):
    """Run each requested function and return its result to the Live session.

    Product cards/details are pushed to the browser separately so the UI updates
    while the model composes its spoken summary from the compact result. `state`
    holds the last search's ranked cards so rank references (e.g. "the 2nd one")
    can resolve to a real product.
    """
    responses = []
    for fc in tool_call.function_calls:
        args = dict(fc.args or {})
        if fc.name == "search_products":
            stores = [agent.RETAILER_LABELS[r] for r in agent._normalize_retailers(args.get("retailers"))]
            await ws.send_json({"type": "search_running", "query": args.get("query", ""), "stores": stores})
            t0 = time.time()
            try:
                compact, cards = await agent.run_search(args)
            except Exception as e:
                print(f"search_products error: {e}")
                traceback.print_exc()
                compact, cards = {"error": str(e)}, []
            latency_ms = int((time.time() - t0) * 1000)
            if cards:
                await ws.send_json({"type": "products", "query": args.get("query", ""), "cards": cards})
                state["query"] = args.get("query", "")
                state["results"] = {c["rank"]: c for c in cards}
            else:
                await ws.send_json({"type": "no_products", "query": args.get("query", "")})
            eventlog.log("search", session_id=session_id, query=args.get("query"),
                         retailers=stores, result_count=len(cards), latency_ms=latency_ms,
                         blocked=(len(cards) == 0),
                         payload={"min_price": args.get("min_price"),
                                  "max_price": args.get("max_price"),
                                  "features": args.get("features")})
            result = compact

        elif fc.name == "get_product_details":
            card = _lookup(state, args.get("rank"))
            if not card:
                result = {"error": "No matching product yet — search first, or restate which item."}
            else:
                await ws.send_json({"type": "detail_running", "rank": card["rank"]})
                try:
                    summary = await agent.fetch_details(card, state.get("query", ""))
                except Exception as e:
                    print(f"get_product_details error: {e}")
                    summary = ""
                if summary:
                    await ws.send_json({"type": "product_detail", "rank": card["rank"], "summary": summary})
                eventlog.log("detail", session_id=session_id,
                             payload={"rank": card["rank"], "title": card.get("title", "")[:120],
                                      "store": card.get("store"), "found": bool(summary)})
                result = {
                    "rank": card["rank"],
                    "title": card.get("title", "")[:90],
                    "details": summary or "Couldn't read the detail page; share the card info you have.",
                    "instruction": "Relay these details to the user in a sentence or two, spoken naturally.",
                }

        elif fc.name == "highlight_product":
            card = _lookup(state, args.get("rank"))
            if card:
                await ws.send_json({"type": "highlight_product", "rank": card["rank"]})
                eventlog.log("highlight", session_id=session_id,
                             payload={"rank": card["rank"], "title": card.get("title", "")[:120]})
                result = {"ok": True, "rank": card["rank"]}
            else:
                result = {"error": "No matching product to highlight."}

        elif fc.name == "arrange_results":
            results = state.get("results") or {}
            order = []
            for r in (args.get("order") or []):
                try:
                    ri = int(r)
                except (TypeError, ValueError):
                    continue
                if ri in results and ri not in order:
                    order.append(ri)
            if not order:
                result = {"error": "No current results to arrange — search first."}
            else:
                await ws.send_json({"type": "arrange", "order": order, "title": args.get("title")})
                eventlog.log("arrange", session_id=session_id,
                             payload={"order": order, "title": args.get("title")})
                result = {"ok": True, "shown": len(order)}

        else:
            result = {"error": f"unknown function {fc.name}"}
        responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))

    if responses:
        # The session or browser WS may have dropped during a slow search — don't
        # let a late tool response crash anything.
        try:
            await session.send_tool_response(function_responses=responses)
        except Exception as e:
            print(f"tool response not delivered (session likely closed): {e}")


async def _read_init(ws):
    """Wait for the client's first `init` message: an optional Gemini resume handle
    plus the last search's cards (so server-side rank lookups survive a reconnect).
    Returns (resume_handle, state). Tolerates a missing/late init."""
    import json
    resume_handle = None
    state = {"results": {}, "query": ""}
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=5)
        init = json.loads(raw)
        if init.get("type") == "init":
            resume_handle = init.get("resume_handle") or None
            lr = init.get("last_results") or {}
            cards = lr.get("cards") or []
            if cards:
                state["query"] = lr.get("query", "")
                state["results"] = {c["rank"]: c for c in cards if "rank" in c}
    except (asyncio.TimeoutError, Exception) as e:
        print(f"init not received cleanly ({type(e).__name__}); starting fresh")
    return resume_handle, state


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = uuid.uuid4().hex[:12]
    resume_handle, state = await _read_init(ws)
    eventlog.log("session_start", session_id=session_id,
                 payload={"resumed": bool(resume_handle), "restored": len(state["results"]),
                          "user_agent": ws.headers.get("user-agent")})
    turn = {"user": "", "agent": ""}  # accumulates the current turn's transcripts

    def flush_turn():
        u, a = turn["user"].strip(), turn["agent"].strip()
        if u or a:
            eventlog.log("turn", session_id=session_id, user_text=u or None, agent_text=a or None)
        turn["user"] = turn["agent"] = ""

    try:
        async with get_client().aio.live.connect(model=MODEL, config=live_config(resume_handle)) as session:
            await ws.send_json({"type": "ready", "model": MODEL, "resumed": bool(resume_handle),
                                "restored": len(state["results"])})

            async def browser_to_gemini():
                """Forward mic audio + push-to-talk control messages to Gemini."""
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    if msg.get("bytes") is not None:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=msg["bytes"],
                                mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                            )
                        )
                    # Hands-free: automatic VAD detects turns, so no per-turn control
                    # messages are expected here (the one-time `init` is read before
                    # this loop starts). Any stray text is ignored.

            bg_tasks = set()  # `state` is set above from the client's init message

            async def gemini_to_browser():
                """Forward Gemini audio, transcripts, and turn signals to the browser."""
                while True:
                    async for response in session.receive():
                        if response.session_resumption_update:
                            upd = response.session_resumption_update
                            if upd.resumable and upd.new_handle:
                                # Hand the latest resume handle to the client to replay
                                # on reconnect.
                                await ws.send_json({"type": "resume_handle", "handle": upd.new_handle})
                        if response.tool_call:
                            # Run the search concurrently so a slow scrape doesn't
                            # stall the receive loop (which keeps audio flowing and
                            # the session alive).
                            t = asyncio.create_task(handle_tool_call(ws, session, response.tool_call, state, session_id))
                            bg_tasks.add(t)
                            t.add_done_callback(bg_tasks.discard)
                        sc = response.server_content
                        if response.data:
                            await ws.send_bytes(response.data)
                        if sc:
                            if sc.input_transcription and sc.input_transcription.text:
                                turn["user"] += sc.input_transcription.text
                                await ws.send_json({"type": "user_transcript",
                                                    "text": sc.input_transcription.text})
                            if sc.output_transcription and sc.output_transcription.text:
                                turn["agent"] += sc.output_transcription.text
                                await ws.send_json({"type": "agent_transcript",
                                                    "text": sc.output_transcription.text})
                            if sc.interrupted:
                                await ws.send_json({"type": "interrupted"})
                            if sc.turn_complete:
                                flush_turn()  # log the completed user+agent exchange
                                await ws.send_json({"type": "turn_complete"})

            tasks = [asyncio.create_task(browser_to_gemini()),
                     asyncio.create_task(gemini_to_browser())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect,) + SESSION_END_ERRORS):
                    raise exc

    except WebSocketDisconnect:
        print("Client disconnected")
    except SESSION_END_ERRORS as e:
        # Idle/expired Gemini Live session — normal; the client reconnects.
        print(f"Gemini Live session ended ({type(e).__name__}): {e}")
    except Exception as e:
        print(f"WS error: {e}")
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        flush_turn()  # capture any in-progress exchange before closing
        eventlog.log("session_end", session_id=session_id)
        try:
            await ws.close()
        except Exception:
            pass
