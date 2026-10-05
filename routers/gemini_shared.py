"""
Shared Gemini-calling infrastructure, used by both routers/gold.py and
routers/chatbot.py.

This used to live entirely inside gold.py. It's pulled out here because
gold and the chatbot draw from the SAME actual Google API quota — if
each file tracked cooldowns independently, gold.py could learn a model
is exhausted and chatbot.py would have no idea, wasting a request on a
guaranteed failure (and vice versa). One shared cooldown table means
either feature marking a model exhausted protects the other from
hitting it too.

Two calling strategies are offered:

  - race_gemini_models(...): fires several models concurrently per
    batch, first success wins, used by gold.py. Good fit for an
    occasional background refresh where speed-to-result matters and
    burning a couple of extra requests occasionally is fine.

  - call_gemini_sequential(...): tries models ONE AT A TIME, only
    moving to the next on failure. Used by chatbot.py. A chat can be
    many messages long, and racing 3 models per message would burn
    ~3x the quota per message (cancelling a "losing" concurrent request
    client-side doesn't un-send it — Google's servers already received
    it). Sequential trades a little latency in the rare failure case
    for spending only 1 request per message in the common successful
    case.

Both strategies share the same underlying single-call primitive, the
same cooldown bookkeeping, and the same `async with ... .aio as client`
cleanup pattern (see the note in _call_gemini_once's callers — closing
the async client explicitly, rather than letting Python's garbage
collector do it later, avoids a real "Event loop is closed" error and
connection leak that happens otherwise once asyncio.run() has already
torn down the event loop the cleanup needs).
"""

import asyncio
import os
import re
import time

from .mem_log import log_memory

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Ordered by preference, based on (a) this project's own rate-limit
# dashboard (Google AI Studio -> usage) and (b) Google's official models
# page, which explicitly says: "we are limiting access to the 2.5
# models to users who have actively used them in the past... For any
# new projects, use our latest models: 3.5 Flash-Lite or 3.8 Flash."
# In practice, gemini-2.5-flash and gemini-2.5-flash-lite now return a
# hard 404 "no longer available to new users" for this account — so
# they're excluded entirely rather than kept as a deprioritized last
# resort. Quotas differ a lot between models and change over time —
# re-check https://aistudio.google.com/usage occasionally and adjust
# this list; nothing here is guaranteed to stay accurate indefinitely.
GEMINI_CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",   # 500 req/day on this account — Stable, Google's recommended lite default
    "gemini-3.1-flash-lite",   # 500 req/day — Stable
    "gemini-3.5-flash",        # 20 req/day — Stable
    "gemini-3.6-flash",        # 20 req/day — Stable
    "gemini-3.7-flash",        # 20 req/day — Stable
    "gemini-3.8-flash",        # 20 req/day — Stable, Google's recommended flagship default
    "gemini-3-flash-preview",  # confirmed callable via list_gemini_models.py — a Preview
                                # model, so Google could change/retire it with less notice
                                # than the dated stable releases above, but it's a genuinely
                                # separate quota pool while it's around
]

GEMINI_RACE_BATCH_SIZE = 3          # gold.py: try this many models concurrently per round
GEMINI_RACE_ATTEMPT_TIMEOUT = 15    # seconds to wait for a batch before moving on
GEMINI_SEQUENTIAL_TIMEOUT = 15      # chatbot.py: seconds to wait for a single model before
                                     # moving to the next one

# Per-model cooldowns — once a model reports quota exhaustion
# (429 RESOURCE_EXHAUSTED), there's no point trying it again until its
# quota resets, so it's skipped for a while rather than wasting a
# request on a guaranteed failure. A 404 (model doesn't exist / has
# been deprecated) gets a long cooldown too — that's not transient like
# a quota limit, so retrying it constantly would waste a request on a
# guaranteed failure indefinitely, until someone notices and edits the
# candidate list. This dict is process-wide (module-level), so gold.py
# and chatbot.py both read/write the exact same cooldown state.
_model_cooldown_until = {}  # model name -> unix timestamp
DEFAULT_COOLDOWN_SECONDS = 4 * 60 * 60     # fallback if a 429 doesn't include a retry delay
NOT_FOUND_COOLDOWN_SECONDS = 24 * 60 * 60  # 404s are effectively permanent, not transient


def _mark_cooldown_if_permanent_error(model_name, error_text):
    if any(marker in error_text for marker in ("RESOURCE_EXHAUSTED", "429")):
        match = re.search(r"retry in ([\d.]+)s", error_text)
        cooldown_seconds = float(match.group(1)) if match else DEFAULT_COOLDOWN_SECONDS
        _model_cooldown_until[model_name] = time.time() + cooldown_seconds
    elif any(marker in error_text for marker in ("404", "NOT_FOUND")):
        _model_cooldown_until[model_name] = time.time() + NOT_FOUND_COOLDOWN_SECONDS


def available_models(candidates=None):
    """Returns `candidates` (defaults to GEMINI_CANDIDATE_MODELS) with
    any currently-cooling-down models filtered out."""
    candidates = candidates if candidates is not None else GEMINI_CANDIDATE_MODELS
    now = time.time()
    return [m for m in candidates if _model_cooldown_until.get(m, 0) <= now]


async def _call_gemini_once(aio_client, model, contents, system_instruction=None):
    kwargs = {"model": model, "contents": contents}
    if system_instruction:
        from google.genai import types
        kwargs["config"] = types.GenerateContentConfig(system_instruction=system_instruction)
    response = await aio_client.models.generate_content(**kwargs)
    return response.text.strip() if response and response.text else ""


async def race_gemini_models(contents, models, system_instruction=None):
    """Tries `models` in batches of GEMINI_RACE_BATCH_SIZE, concurrently
    within each batch. Returns (text, model_name) from whichever model
    responds first with a success. Raises if every model in every batch
    fails or times out."""
    from google import genai  # lazy import — only paid for when a Gemini feature is actually hit
    log_memory("gemini_shared: google.genai imported for race")

    last_errors = []

    async with genai.Client(api_key=GEMINI_API_KEY).aio as client:
        for i in range(0, len(models), GEMINI_RACE_BATCH_SIZE):
            batch = models[i:i + GEMINI_RACE_BATCH_SIZE]
            tasks = {
                asyncio.create_task(_call_gemini_once(client, m, contents, system_instruction)): m
                for m in batch
            }

            done, pending = await asyncio.wait(
                tasks.keys(),
                timeout=GEMINI_RACE_ATTEMPT_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # cancel() on an asyncio.Task genuinely interrupts the
            # in-flight call — unlike an abandoned thread, this doesn't
            # leave it running forever.
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            for t in done:
                model_name = tasks[t]
                exc = t.exception()
                if exc is None:
                    return t.result(), model_name
                error_text = str(exc)
                print(f"Gemini race: {model_name} failed: {error_text}")
                _mark_cooldown_if_permanent_error(model_name, error_text)
                last_errors.append(f"{model_name}: {error_text}")

            for model_name in [tasks[t] for t in pending]:
                last_errors.append(f"{model_name}: timed out after {GEMINI_RACE_ATTEMPT_TIMEOUT}s")
                print(f"Gemini race: {model_name} timed out after {GEMINI_RACE_ATTEMPT_TIMEOUT}s")

        raise RuntimeError("All Gemini models failed or timed out: " + "; ".join(last_errors))


async def call_gemini_sequential(contents, models, system_instruction=None):
    """Tries `models` ONE AT A TIME, moving to the next only on failure
    or timeout. Returns (text, model_name) from whichever model
    succeeds first. Raises if every model fails.

    Deliberately not concurrent — see the module docstring for why this
    matters for anything that makes many calls per user interaction
    (like a chat, where a single conversation can be dozens of
    messages), as opposed to gold.py's occasional background refresh."""
    from google import genai  # lazy import
    log_memory("gemini_shared: google.genai imported for sequential")

    last_errors = []

    async with genai.Client(api_key=GEMINI_API_KEY).aio as client:
        for model in models:
            task = asyncio.create_task(_call_gemini_once(client, model, contents, system_instruction))
            try:
                result = await asyncio.wait_for(task, timeout=GEMINI_SEQUENTIAL_TIMEOUT)
                return result, model
            except Exception as e:
                error_text = f"timed out after {GEMINI_SEQUENTIAL_TIMEOUT}s" if isinstance(e, asyncio.TimeoutError) else str(e)
                print(f"Gemini sequential: {model} failed: {error_text}")
                _mark_cooldown_if_permanent_error(model, error_text)
                last_errors.append(f"{model}: {error_text}")
                # asyncio.wait_for cancels the task on timeout automatically;
                # on any other exception the task is already finished.

        raise RuntimeError("All Gemini models failed: " + "; ".join(last_errors))
