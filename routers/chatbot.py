"""
AI Financial Advisor chatbot — ported from the original Flask app's
/chatbot route and get_financial_advice() function, with real changes:

  - Actual multi-turn memory. The original was single-shot: every
    message built one fresh prompt with no prior turns included at all
    (see get_financial_advice() in the old app — it only ever sent the
    system instruction + that one message). Whatever "chat history" the
    old app kept was purely a client-side UI convenience for display,
    never sent back to Gemini as context. This version actually sends
    the running conversation to Gemini on every turn, so it can refer
    back to what was already discussed.

  - No server-side conversation storage. There's no login system on
    this portfolio, so there's no durable identity to attach saved
    conversations to, and no database this project already has running.
    The frontend (localStorage) owns the persisted transcript entirely;
    this endpoint is stateless — it receives the whole conversation so
    far plus the new message, and just returns a reply. Nothing is kept
    in server memory between requests except the per-session rate-limit
    counter below.

  - Sequential model fallback, not racing. gold.py races 3 models
    concurrently per call, which is fine for an occasional background
    refresh. A chat conversation can be dozens of messages long, and
    racing would burn ~3x the daily quota per message (a "losing"
    concurrent request still reaches Google's servers even if cancelled
    client-side — cancellation just stops US from waiting on it). This
    tries one model at a time, only advancing on failure, so a normal
    successful reply costs exactly 1 request.

  - Per-session daily message cap. Every model listed in
    gemini_shared.py draws from the same shared, fairly small daily
    quota that gold.py's insights feature also depends on. Without a
    cap, one visitor having a long chat could exhaust the day's budget
    for every other feature and every other visitor. Capped per
    browser (see CLIENT_ID note below) rather than globally, so one
    heavy user doesn't lock everyone else out, but also can't
    themselves send unlimited messages.
"""

import re
from datetime import date
from typing import List, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .gemini_shared import GEMINI_API_KEY, GEMINI_CANDIDATE_MODELS, available_models, call_gemini_sequential
from .mem_log import log_memory

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

# =====================================================================
# System instruction
#
# Carries over the original app's financial-advisor persona and its
# "no tables" instruction, extended to also steer away from headers and
# code blocks — not because those are inherently bad, but because the
# frontend only parses a small, deliberately limited markdown subset
# (bold text and simple bullet lists — see gold.js's renderLiteMarkdown
# for the pattern this follows). Instructions alone don't perfectly
# guarantee compliance (the model has ignored style instructions
# before), so this is paired with the frontend actually being able to
# render the bold/bullet subset it commonly does produce, rather than
# relying on instructions alone.
# =====================================================================
SYSTEM_INSTRUCTION = (
    "You are a financial advisor embedded in a portfolio demo project. "
    "Provide financial guidance in a structured yet conversational manner. "
    "Use short paragraphs and simple bullet points (lines starting with '-') "
    "where a list genuinely helps. You may use **bold** for emphasis on key "
    "terms or numbers. Do NOT use tables, markdown headers (#), numbered "
    "lists, links, or code blocks — plain paragraphs and simple '-' bullets "
    "only. Keep replies reasonably concise; this is a chat interface, not a "
    "long-form article."
)

# =====================================================================
# Conversation titles — generated in the SAME call as the first reply
#
# The frontend shows a sidebar of separate conversations, each with a
# short title. Asking Gemini for that title in a second request would
# double the quota cost of every new chat, and quota is the scarce
# resource here. Instead, for the first message of a conversation only,
# the model is asked to put the title on the first line of its reply;
# _split_title peels it off before the reply goes back to the frontend.
# If the model ignores the format, title comes back None and the
# frontend falls back to the first few words of the user's message — so
# a formatting slip costs nothing but a slightly less polished title.
# =====================================================================
TITLE_INSTRUCTION = (
    " This is the FIRST message of a new conversation. Begin your response "
    "with a single line in exactly this format: 'TITLE: ' followed by a "
    "concise 4-7 word plain-text title for the conversation's topic (no "
    "quotes, no markdown, no trailing period), then a blank line, then your "
    "normal reply. Never mention or explain the title line."
)

# Tolerant on purpose: models are told "no markdown" but sometimes bold
# the label anyway (**TITLE:** ...), wrap the whole line in bold, or
# prefix it with '#'. If the pattern were strict, those replies would
# both lose their title AND show the raw "**TITLE:** ..." line to the
# visitor — so it accepts that decoration, and may appear within the
# first few lines in case the model adds a short preamble first.
_TITLE_LINE_RE = re.compile(
    r"^[ \t]*[*_#>` \t]*TITLE[ \t]*[:\-–—][ \t]*(.+?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
TITLE_SEARCH_WINDOW_CHARS = 300  # only look near the top of the reply
MAX_TITLE_WORDS = 8
MAX_TITLE_CHARS = 60


def _clean_title(raw):
    """Strips markdown/quote noise and enforces length limits, since the
    model's output is untrusted formatting-wise."""
    title = re.sub(r"[*_`#\"“”]", "", raw)
    title = re.sub(r"\s+", " ", title).strip().strip("'").rstrip(".:;,!-– ")
    title = " ".join(title.split(" ")[:MAX_TITLE_WORDS])[:MAX_TITLE_CHARS].strip()
    return title or None


def _split_title(raw_reply):
    """Returns (title_or_None, reply_without_title_line). If there's no
    TITLE line near the top — or removing it would leave an empty reply
    — returns (None, raw_reply) unchanged rather than risk showing the
    visitor nothing."""
    match = _TITLE_LINE_RE.search(raw_reply, 0, TITLE_SEARCH_WINDOW_CHARS)
    if not match:
        return None, raw_reply
    remaining = (raw_reply[:match.start()] + raw_reply[match.end():]).strip()
    remaining = re.sub(r"\n{3,}", "\n\n", remaining)
    if not remaining:
        return None, raw_reply
    return _clean_title(match.group(1)), remaining

# =====================================================================
# Per-session daily rate limiting
#
# "Session" here means one generated client ID, created once by the
# frontend and stored in localStorage — the same ID the frontend also
# uses as the key under which it stores that visitor's chat history.
# Reusing it for both means the rate limit persists across visits the
# same way the chat history does (a new tab/session via sessionStorage
# would trivially reset it, which defeats the point).
#
# This is in-memory only, like every other cache in this backend — it
# resets on redeploy/restart, which is an acceptable tradeoff given
# Render's disk is ephemeral anyway and this isn't a security boundary,
# just a courtesy limit protecting a small shared quota.
# =====================================================================
MAX_MESSAGES_PER_SESSION_PER_DAY = 18

_session_usage = {}  # client_id -> {"date": "YYYY-MM-DD", "count": int}


def _get_and_increment_usage(client_id):
    """Returns (messages_used_today_after_this_one, limit). Raises
    HTTPException(429) if already at the cap."""
    today = date.today().isoformat()
    entry = _session_usage.get(client_id)

    if not entry or entry["date"] != today:
        entry = {"date": today, "count": 0}

    if entry["count"] >= MAX_MESSAGES_PER_SESSION_PER_DAY:
        _session_usage[client_id] = entry
        raise HTTPException(
            status_code=429,
            detail=(
                f"You've reached today's limit of {MAX_MESSAGES_PER_SESSION_PER_DAY} "
                "messages for this demo — this keeps the shared free API quota "
                "available for other visitors too. Please try again tomorrow."
            ),
        )

    entry["count"] += 1
    _session_usage[client_id] = entry
    return entry["count"], MAX_MESSAGES_PER_SESSION_PER_DAY


# =====================================================================
# Request/response models
# =====================================================================
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    client_id: str = Field(..., min_length=8, max_length=128)
    history: List[ChatTurn] = Field(default_factory=list, max_length=60)
    message: str = Field(..., min_length=1, max_length=2000)


# =====================================================================
# Building the Gemini `contents` list
# =====================================================================
def _build_contents(history, new_message):
    """Converts the frontend's simple {role, text} turns into the
    google-genai SDK's types.Content list, then appends the new user
    message as the final turn."""
    from google.genai import types

    contents = []
    for turn in history:
        gemini_role = "model" if turn.role == "assistant" else "user"
        contents.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=turn.text)]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=new_message)]))
    return contents


# =====================================================================
# Routes
# =====================================================================
@router.post("/message")
async def send_message(payload: ChatRequest):
    """Sends the running conversation (history + new message) to
    Gemini and returns its reply. Stateless — nothing about this
    conversation is stored server-side; the caller is expected to keep
    sending the full history back on every subsequent call (that's
    exactly what the frontend's localStorage-backed history is for)."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not configured.")

    messages_used, limit = _get_and_increment_usage(payload.client_id)

    candidates = available_models(GEMINI_CANDIDATE_MODELS)
    if not candidates:
        raise HTTPException(
            status_code=503,
            detail="Every configured Gemini model is currently on cooldown (daily quota exhausted). Please try again later.",
        )

    contents = _build_contents(payload.history, payload.message)

    # An empty history means this is the first message of a brand-new
    # conversation — the only time a title is needed (see the note above
    # TITLE_INSTRUCTION for why it rides along with the reply).
    is_new_conversation = len(payload.history) == 0
    system_instruction = SYSTEM_INSTRUCTION + (TITLE_INSTRUCTION if is_new_conversation else "")

    try:
        reply_text, model_used = await call_gemini_sequential(contents, candidates, system_instruction=system_instruction)
        log_memory("chatbot: Gemini response received")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Chatbot temporarily unavailable: {e}")

    title = None
    if is_new_conversation:
        title, reply_text = _split_title(reply_text)

    return {
        "success": True,
        "reply": reply_text,
        "title": title,
        "model_used": model_used,
        "messages_used_today": messages_used,
        "messages_limit_per_day": limit,
    }


@router.get("/limit/{client_id}")
def get_usage(client_id: str):
    """Lets the frontend show remaining message count without spending
    one — e.g. on page load, before the visitor sends anything."""
    today = date.today().isoformat()
    entry = _session_usage.get(client_id)
    used = entry["count"] if entry and entry["date"] == today else 0
    return {
        "messages_used_today": used,
        "messages_limit_per_day": MAX_MESSAGES_PER_SESSION_PER_DAY,
        "messages_remaining_today": max(0, MAX_MESSAGES_PER_SESSION_PER_DAY - used),
    }