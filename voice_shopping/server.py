"""Voice shopping agent — Milestone 1.

FastAPI server that relays push-to-talk audio between the browser and a Gemini
Live session. No product tools yet — this milestone proves the end-to-end voice
loop: hold the mic button, speak, hear the model reply.

Run from the repo root so the shared `modules/` package stays importable later:

    venv/bin/uvicorn voice_shopping.server:app --reload --port 5050
"""
import os
import asyncio
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

from voice_shopping import agent

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

After results come back:
- Briefly recommend the top one or two picks and why they fit — a sentence or two,
  spoken naturally. If you searched more than one store, say which store a pick is
  from. The cards are on screen, so don't read out every spec or any links.

Follow-ups (important):
- If they want changes ("cheaper", "only Sony", "something smaller", "add Walmart"),
  call `search_products` again with the updated constraints.
- They can ask about a specific pick by its number or store — answer from what you
  already found, no need to search again."""

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


def live_config() -> types.LiveConnectConfig:
    """Audio-out config with manual (push-to-talk) turn control and transcripts."""
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        # Pin output language — Live otherwise sometimes drifts (e.g. to Spanish).
        speech_config=types.SpeechConfig(language_code="en-US"),
        tools=[agent.tool()],
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
        # Push-to-talk: we mark turn boundaries ourselves instead of letting the
        # server guess from silence.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


async def handle_tool_call(ws: WebSocket, session, tool_call):
    """Run each requested function and return its result to the Live session.

    Product cards are pushed to the browser separately so the UI can render them
    while the model composes its spoken summary from the compact result.
    """
    responses = []
    for fc in tool_call.function_calls:
        if fc.name == "search_products":
            args = dict(fc.args or {})
            stores = [agent.RETAILER_LABELS[r] for r in agent._normalize_retailers(args.get("retailers"))]
            await ws.send_json({"type": "search_running", "query": args.get("query", ""), "stores": stores})
            try:
                compact, cards = await agent.run_search(args)
            except Exception as e:
                print(f"search_products error: {e}")
                traceback.print_exc()
                compact, cards = {"error": str(e)}, []
            if cards:
                await ws.send_json({"type": "products", "query": args.get("query", ""), "cards": cards})
            else:
                await ws.send_json({"type": "no_products", "query": args.get("query", "")})
            result = compact
        else:
            result = {"error": f"unknown function {fc.name}"}
        responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))

    if responses:
        await session.send_tool_response(function_responses=responses)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        async with get_client().aio.live.connect(model=MODEL, config=live_config()) as session:
            await ws.send_json({"type": "ready", "model": MODEL})

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
                    elif msg.get("text") is not None:
                        import json
                        event = json.loads(msg["text"])
                        if event.get("type") == "start_turn":
                            await session.send_realtime_input(activity_start=types.ActivityStart())
                        elif event.get("type") == "end_turn":
                            await session.send_realtime_input(activity_end=types.ActivityEnd())

            async def gemini_to_browser():
                """Forward Gemini audio, transcripts, and turn signals to the browser."""
                while True:
                    async for response in session.receive():
                        if response.tool_call:
                            await handle_tool_call(ws, session, response.tool_call)
                        sc = response.server_content
                        if response.data:
                            await ws.send_bytes(response.data)
                        if sc:
                            if sc.input_transcription and sc.input_transcription.text:
                                await ws.send_json({"type": "user_transcript",
                                                    "text": sc.input_transcription.text})
                            if sc.output_transcription and sc.output_transcription.text:
                                await ws.send_json({"type": "agent_transcript",
                                                    "text": sc.output_transcription.text})
                            if sc.interrupted:
                                await ws.send_json({"type": "interrupted"})
                            if sc.turn_complete:
                                await ws.send_json({"type": "turn_complete"})

            tasks = [asyncio.create_task(browser_to_gemini()),
                     asyncio.create_task(gemini_to_browser())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    raise exc

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WS error: {e}")
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
