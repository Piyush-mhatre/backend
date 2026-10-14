import os
import random
import threading
import json
import re
from typing import List

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .mem_log import log_memory, get_memory_stats
from .gemini_shared import GEMINI_API_KEY, available_models, call_gemini_sequential

router = APIRouter(
    prefix="/news",
    tags=["News"]
)


# ============================================================
# Configuration
# ============================================================

# Local tokenizer files (committed directly to git — small, no Release
# needed) instead of AutoTokenizer.from_pretrained("ProsusAI/finbert"),
# so boot never needs a Hugging Face network call at all.
TOKENIZER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "tokenizer"
)

# ONNX Runtime, INT8-quantized (see export_to_onnx.py, run once locally).
# This replaces the old torch-based model entirely — onnxruntime's own
# import footprint is roughly 10x smaller than torch's, which is what
# was causing OOM crashes on Render's 512MB free tier even after every
# quantization-level fix (torch's runtime — libtorch/MKL/OpenMP — costs
# 200-300MB just by existing in the process, independent of model size).
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "finbert_int8.onnx"
)

API_KEY = os.environ.get("NEWS_API_KEY")

# Minimum number of articles we want back from the precise (title-scoped)
# search before we give up on it and fall back to a broader one. Narrow
# keywords ("Reliance Jio", say) can legitimately have few title hits —
# this stops the page coming back empty for anything but the most common
# keywords.
MIN_RESULTS_BEFORE_FALLBACK = 3

# Optional: restrict results to known financial-news domains for even
# tighter relevance. Off by default — a wrong/too-short list can hide
# legitimate results more than it helps. Add domains here if you find
# specific keywords are still pulling in irrelevant sources.
PREFERRED_DOMAINS = []  # e.g. ["reuters.com", "moneycontrol.com", "livemint.com"]

# ============================================================
# Global model state
#
# NOTE: the actual model is built lazily inside load_finbert_model()
# (triggered by start_model_loading(), called at app startup elsewhere).
# Nothing heavy happens here at import time — tokenizer/model/state are
# just declared as None/False until the background thread populates them.
# ============================================================

tokenizer = None
model = None  # an onnxruntime.InferenceSession once loaded
np = None
ort = None
AutoTokenizer = None
_finbert_libs_loaded = False


def _ensure_finbert_libs():
    global np, ort, AutoTokenizer, _finbert_libs_loaded
    if _finbert_libs_loaded:
        return
    import numpy as _np
    import onnxruntime as _ort
    from transformers import AutoTokenizer as _AutoTokenizer

    np, ort, AutoTokenizer = _np, _ort, _AutoTokenizer
    _finbert_libs_loaded = True

model_loading = False
model_ready = False

sentiment_cache = {}

# FinBERT label mapping:
# 0 = positive
# 1 = negative
# 2 = neutral

ID_TO_SENTIMENT = {
    0: "positive",
    1: "negative",
    2: "neutral"
}


# ============================================================
# Load INT8 FinBERT (ONNX Runtime)
# ============================================================

def load_finbert_model():
    global tokenizer
    global model
    global model_loading
    global model_ready

    if model_ready:
        return

    model_loading = True

    print("Loading INT8 FinBERT model...")

    try:
        _ensure_finbert_libs()
        # Load from local files committed to the repo — no Hugging Face
        # network call at boot at all (this used to hit HF for the
        # tokenizer every cold start; now it's just reading small local
        # files, one less thing that can rate-limit or fail).
        print("Loading FinBERT tokenizer (local files)...")

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_DIR
        )
        log_memory("news: FinBERT tokenizer loaded")

        # Load the ONNX Runtime session directly — no torch, no fp32
        # architecture build, no quantize_dynamic call. onnxruntime's own
        # import/runtime footprint is roughly 10x smaller than torch's,
        # which is the actual fix for the OOM crashes: torch's runtime
        # (libtorch/MKL/OpenMP) costs 200-300MB just by existing in the
        # process, independent of model size, and no amount of
        # quantizing the model itself could get under that floor.
        print("Loading pre-quantized INT8 FinBERT ONNX model...")

        session_options = ort.SessionOptions()
        # Free tier has limited CPU too — keep thread pools small so
        # onnxruntime doesn't spin up more worker threads than useful,
        # which costs both memory and contention with FastAPI's own
        # request handling.
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        model = ort.InferenceSession(
            MODEL_PATH,
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )

        model_ready = True
        log_memory("news: ONNX INT8 session initialized")

        print("INT8 FinBERT (ONNX) model loaded successfully!")

    except Exception as e:
        model_ready = False
        print(f"Error loading FinBERT: {e}")

    finally:
        model_loading = False


# ============================================================
# Sentiment prediction
# ============================================================

def predict_sentiment(text):
    global model
    global tokenizer

    if not model_ready or np is None:
        return "neutral"

    if not text:
        return "neutral"

    # Cache repeated text
    if text in sentiment_cache:
        return sentiment_cache[text]

    try:
        encoded = tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np"
        )

        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        logits = model.run(["logits"], ort_inputs)[0]
        prediction = int(np.argmax(logits, axis=1)[0])

        result = ID_TO_SENTIMENT.get(
            prediction,
            "neutral"
        )

        sentiment_cache[text] = result

        return result

    except Exception as e:
        print(f"Sentiment prediction error: {e}")
        return "neutral"


# ============================================================
# Fetch news
#
# RELEVANCE FIX: the original version searched with a bare keyword
# (q=reliance), which NewsAPI matches loosely against title+description+
# content — so "reliance" pulled in anything containing that word in any
# sense ("our reliance on renewable energy"), not just Reliance
# Industries. FinBERT correctly scores that kind of generic, non-financial
# sentence as neutral — so "everything comes back neutral" was actually a
# correct classification of irrelevant input, not a broken model.
#
# Fix, in order of restrictiveness:
#   1. Wrap the keyword in quotes -> NewsAPI treats it as an exact phrase
#      instead of loose keyword matching.
#   2. searchIn=title,description -> an article that mentions the phrase
#      in its title/description is far more likely to actually be about
#      it than one that just contains it somewhere in a long body.
#   3. sortBy=relevancy -> surfaces the best matches first instead of
#      just the most recent.
#   4. If that precise search comes back thin (narrow keyword), fall back
#      to a broader exact-phrase search without the searchIn restriction,
#      so the page doesn't just come back empty.
# ============================================================

def _fetch_articles(keyword, *, search_in=None, use_preferred_domains=False):
    """One NewsAPI call. Returns the raw articles list (may be empty)."""
    params = {
        "q": f'"{keyword}"',   # exact phrase, not loose keyword matching
        "pageSize": 10,
        "sortBy": "relevancy",
        "apiKey": API_KEY,
    }
    if search_in:
        params["searchIn"] = search_in
    if use_preferred_domains and PREFERRED_DOMAINS:
        params["domains"] = ",".join(PREFERRED_DOMAINS)

    response = requests.get(
        "https://newsapi.org/v2/everything",
        params=params,
        timeout=10
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        print(f"NewsAPI error: {data}")
        return []

    return data.get("articles", [])


class ArticleSentimentItem(BaseModel):
    id: int = 0
    title: str = ""
    description: str = ""


class SentimentRequest(BaseModel):
    engine: str = Field(default="gemini")  # "gemini" or "finbert"
    articles: List[ArticleSentimentItem]


async def classify_sentiment_gemini(articles_data):
    """
    Takes a list of article dicts (each having 'title' and 'description').
    Uses Gemini (sequential fallback across GEMINI_CANDIDATE_MODELS) to
    classify each article's financial sentiment into 'positive', 'negative', or 'neutral'.
    Returns (sentiments_list, model_used).
    If Gemini is unavailable or errors out, safely defaults all to 'neutral'.
    """
    if not articles_data:
        return [], "none"

    if not GEMINI_API_KEY:
        print("Gemini sentiment: GEMINI_API_KEY not configured, defaulting to neutral.")
        return ["neutral"] * len(articles_data), "fallback"

    candidates = available_models()
    if not candidates:
        print("Gemini sentiment: All candidate models cooling down, defaulting to neutral.")
        return ["neutral"] * len(articles_data), "fallback"

    lines = []
    for idx, art in enumerate(articles_data):
        title = (art.get("title") or "").strip()
        desc = (art.get("description") or "").strip()[:160]
        lines.append(f"[{idx}] Title: {title}\nDescription: {desc}")

    prompt = (
        "Classify the financial sentiment of each of the following news articles into strictly "
        "one of: 'positive', 'negative', or 'neutral'.\n\n"
        + "\n\n".join(lines) +
        "\n\nRespond ONLY with a valid JSON array of strings corresponding to each article index, for example: "
        '["positive", "neutral", "negative"]. Do not return markdown codeblocks, explanation, or any other text.'
    )

    system_instruction = (
        "You are an expert financial sentiment classifier. Analyze news articles objectively "
        "from an investor's perspective. Output strictly a JSON array of strings: 'positive', 'negative', or 'neutral'."
    )

    try:
        reply_text, model_used = await call_gemini_sequential(
            prompt,
            candidates,
            system_instruction=system_instruction
        )
        log_memory(f"news: Gemini ({model_used}) sentiment completed")

        cleaned = re.sub(r"^```(?:json)?\s*", "", reply_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            sentiments = []
            for item in parsed:
                val = str(item).strip().lower()
                if "pos" in val:
                    sentiments.append("positive")
                elif "neg" in val:
                    sentiments.append("negative")
                else:
                    sentiments.append("neutral")

            while len(sentiments) < len(articles_data):
                sentiments.append("neutral")
            return sentiments[:len(articles_data)], model_used

    except Exception as e:
        print(f"Gemini sentiment classification error: {e}")

    return ["neutral"] * len(articles_data), "fallback"


def fetch_news_articles(keyword):
    """
    Fetches raw financial news from NewsAPI for a keyword.
    Assigns initial 'neutral' sentiment to each article so news
    retrieval is fast, decoupled, and never blocked by model or API latency.
    """
    if not API_KEY:
        raise RuntimeError(
            "NEWS_API_KEY environment variable is not configured."
        )

    try:
        # Precise pass: exact phrase, title+description only, relevancy-sorted.
        articles = _fetch_articles(
            keyword,
            search_in="title,description",
            use_preferred_domains=True,
        )

        if len(articles) < MIN_RESULTS_BEFORE_FALLBACK:
            print(
                f"Only {len(articles)} title/description hits for "
                f"'{keyword}', falling back to a broader exact-phrase search"
            )
            fallback_articles = _fetch_articles(keyword)

            # Merge, preferring the precise hits first and skipping dupes.
            seen_urls = {a.get("url") for a in articles}
            for a in fallback_articles:
                if a.get("url") not in seen_urls:
                    articles.append(a)
                    seen_urls.add(a.get("url"))

        random_news = random.sample(
            articles,
            min(len(articles), 10)
        )

        news = []
        for idx, article in enumerate(random_news):
            description = article.get("description") or ""
            news.append({
                "id": idx,
                "title": article.get("title", "No Title"),
                "sentiment": "neutral",
                "description": description or "No description available.",
                "url": article.get("url", "#"),
                "image_url": article.get("urlToImage") or "https://via.placeholder.com/300"
            })

        return news

    except requests.RequestException as e:
        print(f"NewsAPI request error: {e}")
        raise RuntimeError(f"Failed to fetch news: {e}")
    except Exception as e:
        print(f"News processing error: {e}")
        raise RuntimeError(f"Failed to process news: {e}")


def get_news_with_sentiment(keyword):
    """
    Legacy wrapper: fetches articles and, if FinBERT happens to be already ready,
    predicts sentiment; otherwise leaves sentiment as neutral.
    """
    news = fetch_news_articles(keyword)
    if model_ready:
        for item in news:
            sentiment_text = (item.get("description") or "")[:100]
            if sentiment_text:
                item["sentiment"] = predict_sentiment(sentiment_text)
    return news


# ============================================================
# Routes
# ============================================================

@router.get("/{keyword}")
def get_news(keyword: str):
    """
    Fetch up to 10 financial news articles for a keyword.
    Returns articles immediately so the frontend displays news instantly.
    The frontend then requests sentiment asynchronously without blocking news display.
    """
    if not keyword.strip():
        raise HTTPException(
            status_code=400,
            detail="Keyword cannot be empty."
        )

    # Note: We intentionally do NOT call start_model_loading() unconditionally here!
    # FinBERT will only load if explicitly requested by the user and memory headroom allows.
    try:
        news = fetch_news_articles(keyword.strip())
        log_memory(f"news: /news/{keyword.strip()} completed ({len(news)} articles)")
        return {
            "keyword": keyword.strip(),
            "count": len(news),
            "articles": news
        }

    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/sentiment")
async def analyze_sentiment(payload: SentimentRequest):
    """
    Asynchronously classifies sentiment for a list of articles using either
    Gemini (default, lightweight cloud AI) or FinBERT (local INT8 model, only
    if memory headroom is sufficient).
    """
    engine = (payload.engine or "gemini").lower().strip()
    articles = payload.articles

    if not articles:
        return {"success": True, "engine": engine, "sentiments": []}

    if engine == "finbert":
        mem_stats = get_memory_stats(threshold_mb=150.0)
        can_use = model_ready or mem_stats["can_load_finbert"]

        if not can_use:
            return {
                "success": False,
                "error": mem_stats["reason"],
                "engine": "finbert",
                "model_used": "none",
                "sentiments": [{"id": a.id, "sentiment": "neutral"} for a in articles]
            }

        # If memory headroom is available and model is not ready yet, load it now
        if not model_ready:
            load_finbert_model()

        results = []
        for a in articles:
            text = (a.description or a.title or "")[:100]
            sentiment = predict_sentiment(text)
            results.append({"id": a.id, "sentiment": sentiment})

        return {
            "success": True,
            "engine": "finbert",
            "model_used": "ProsusAI/finbert (ONNX INT8)",
            "sentiments": results
        }

    else:  # Gemini (default)
        articles_data = [{"title": a.title, "description": a.description} for a in articles]
        sentiments_list, model_used = await classify_sentiment_gemini(articles_data)
        results = [
            {"id": a.id, "sentiment": sentiments_list[i]}
            for i, a in enumerate(articles)
        ]
        return {
            "success": True,
            "engine": "gemini",
            "model_used": model_used,
            "sentiments": results
        }


# ============================================================
# Model status
# ============================================================

@router.get("/status/model")
def model_status():
    mem_stats = get_memory_stats(threshold_mb=150.0)
    can_use = model_ready or mem_stats["can_load_finbert"]
    return {
        "model": "ProsusAI/finbert (ONNX INT8)",
        "quantization": "INT8",
        "ready": model_ready,
        "loading": model_loading,
        "default_engine": "gemini",
        "can_load_finbert": can_use,
        "memory": mem_stats,
        "reason": "FinBERT model ready in memory" if model_ready else mem_stats["reason"]
    }


# ============================================================
# Startup loader (memory-guarded)
# ============================================================

def start_model_loading():
    """
    Safely start loading FinBERT only if memory headroom is >= 150MB.
    Prevents Render free tier OOM crashes.
    """
    global model_loading

    if model_ready or model_loading:
        return

    mem_stats = get_memory_stats(threshold_mb=150.0)
    if not mem_stats["can_load_finbert"]:
        print(f"FinBERT load skipped: {mem_stats['reason']}")
        return

    threading.Thread(
        target=load_finbert_model,
        daemon=True
    ).start()

