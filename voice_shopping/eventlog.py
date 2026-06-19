"""Conversation & behavior event logging.

Captures per-session events (utterance turns, searches, detail views, session
lifecycle) to Postgres for later analysis. Resilient by design: if DATABASE_URL
is unset or the DB is unreachable, it degrades to stdout-only and never raises
into the request path. All writes are fire-and-forget so logging never adds
latency to the voice loop.

Set DATABASE_URL to a Postgres connection string to enable persistence.
"""
import os
import json
import asyncio

_pool = None
_enabled = None  # None = not yet tried, True/False afterwards
_init_lock = asyncio.Lock()
_bg = set()  # keep references to fire-and-forget tasks

DDL = """
CREATE TABLE IF NOT EXISTS events (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT,
    ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
    type         TEXT NOT NULL,            -- session_start | turn | search | detail | highlight | session_end
    user_text    TEXT,
    agent_text   TEXT,
    query        TEXT,
    retailers    TEXT[],
    result_count INT,
    latency_ms   INT,
    blocked      BOOLEAN,
    payload      JSONB
);
CREATE INDEX IF NOT EXISTS events_session_idx ON events (session_id);
CREATE INDEX IF NOT EXISTS events_ts_idx ON events (ts);
"""


def _normalize_url(url: str) -> str:
    # Render/Heroku hand out postgres://; asyncpg wants postgresql://.
    if url and url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


async def _get_pool():
    global _pool, _enabled
    if _enabled is False:
        return None
    if _pool is not None:
        return _pool
    async with _init_lock:
        if _pool is not None:
            return _pool
        url = _normalize_url(os.getenv("DATABASE_URL"))
        if not url:
            _enabled = False
            print("eventlog: DATABASE_URL not set — logging to stdout only")
            return None
        try:
            import asyncpg
            _pool = await asyncpg.create_pool(url, min_size=1, max_size=4, command_timeout=10)
            async with _pool.acquire() as conn:
                await conn.execute(DDL)
            _enabled = True
            print("eventlog: connected to Postgres, events table ready")
        except Exception as e:
            _enabled = False
            print(f"eventlog: DB unavailable ({e}); logging to stdout only")
            return None
    return _pool


async def _write(type, session_id, user_text, agent_text, query,
                 retailers, result_count, latency_ms, blocked, payload):
    # Always emit a compact stdout line (shows in Render logs even without a DB).
    brief = {k: v for k, v in {
        "query": query, "results": result_count, "ms": latency_ms, "blocked": blocked,
        "user": (user_text or "")[:80] or None, "agent": (agent_text or "")[:80] or None,
    }.items() if v is not None}
    print(f"[event] {type} session={session_id} {json.dumps(brief)}", flush=True)

    pool = await _get_pool()
    if not pool:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO events
                   (session_id, type, user_text, agent_text, query, retailers,
                    result_count, latency_ms, blocked, payload)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb)""",
                session_id, type, user_text, agent_text, query, retailers,
                result_count, latency_ms, blocked,
                json.dumps(payload) if payload is not None else None,
            )
    except Exception as e:
        print(f"eventlog: insert failed ({e})")


def log(type, session_id=None, user_text=None, agent_text=None, query=None,
        retailers=None, result_count=None, latency_ms=None, blocked=None, payload=None):
    """Fire-and-forget: schedule the write without blocking the caller."""
    try:
        t = asyncio.create_task(_write(
            type, session_id, user_text, agent_text, query, retailers,
            result_count, latency_ms, blocked, payload))
        _bg.add(t)
        t.add_done_callback(_bg.discard)
    except RuntimeError:
        # No running loop (shouldn't happen in the async server) — skip silently.
        pass
