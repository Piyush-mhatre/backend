import os
import random
import threading

import numpy as np
import requests
import onnxruntime as ort
from fastapi import APIRouter, HTTPException
from transformers import AutoTokenizer

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
        # Load from local files committed to the repo — no Hugging Face
        # network call at boot at all (this used to hit HF for the
        # tokenizer every cold start; now it's just reading small local
        # files, one less thing that can rate-limit or fail).
        print("Loading FinBERT tokenizer (local files)...")

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_DIR
        )

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

    if not model_ready:
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


def get_news_with_sentiment(keyword):

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

        # Fallback: precise search was too narrow (few/no hits) — retry
        # with the same exact-phrase matching but without the title/
        # description restriction, so a legitimate niche keyword still
        # returns something instead of an empty page.
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

        for article in random_news:

            description = article.get(
                "description"
            ) or ""

            # Match the behavior of the old application:
            # only the first 100 characters are sent to the model.
            sentiment_text = description[:100]

            if sentiment_text:
                sentiment = predict_sentiment(
                    sentiment_text
                )
            else:
                sentiment = "neutral"

            news.append({
                "title": article.get(
                    "title",
                    "No Title"
                ),

                "sentiment": sentiment,

                "description": (
                    description
                    or "No description available."
                ),

                "url": article.get(
                    "url",
                    "#"
                ),

                "image_url": article.get(
                    "urlToImage"
                ) or "https://via.placeholder.com/300"
            })

        return news

    except requests.RequestException as e:

        print(f"NewsAPI request error: {e}")

        raise RuntimeError(
            f"Failed to fetch news: {e}"
        )

    except Exception as e:

        print(f"News processing error: {e}")

        raise RuntimeError(
            f"Failed to process news: {e}"
        )


# ============================================================
# Routes
# ============================================================

@router.get("/{keyword}")
def get_news(keyword: str):

    """
    Fetch up to 10 financial news articles for a keyword
    and classify each article using INT8 FinBERT.
    """

    if not keyword.strip():
        raise HTTPException(
            status_code=400,
            detail="Keyword cannot be empty."
        )

    try:

        news = get_news_with_sentiment(
            keyword.strip()
        )

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


# ============================================================
# Model status
# ============================================================

@router.get("/status/model")
def model_status():

    return {
        "model": "ProsusAI/finbert (ONNX INT8)",
        "quantization": "INT8",
        "ready": model_ready,
        "loading": model_loading
    }


# ============================================================
# Startup loader
# ============================================================

def start_model_loading():

    global model_loading

    if model_ready or model_loading:
        return

    threading.Thread(
        target=load_finbert_model,
        daemon=True
    ).start()
