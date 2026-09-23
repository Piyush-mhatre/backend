"""
Gold price + Gemini AI insights — ported from the original Flask app's
/get_gold_data and /get_gemini_insights routes.

Changes from the original, and why:

  - No hardcoded API key. The original had a real Gemini key committed
    directly in source (os.environ["API_KEY"] = "AIzaSy..."). This reads
    GEMINI_API_KEY from the environment instead, following the same
    pattern as every other feature in this backend (see main.py's
    "ADDING A NEW FEATURE LATER" note) — set it in Render's dashboard,
    never in code.

  - New Gemini SDK. `google.generativeai` (used in the original) is
    Google's deprecated legacy SDK. This uses the current `google-genai`
    SDK instead. The model string is "gemini-flash-latest" — Google's
    "-latest" aliases track whatever their current recommended model
    for that tier is, so this doesn't silently break every few months
    the way a hardcoded dated model name has been (the Gemini line has
    gone through several deprecation waves in the time since the
    original app was written).

  - In-memory caching instead of on-disk JSON cache files. Render's
    disk is ephemeral (wiped on every redeploy/restart anyway), so the
    original's file-based cache-with-atomic-rename logic bought nothing
    in this environment — it's replaced with the same simple in-memory
    TTL dict pattern already used for stock history in
    stock_analysis.py.

  - No forex-python. The original tried yfinance, then forex-python,
    then ECB XML as three fallback sources for the USD->INR rate.
    forex-python is a fairly unmaintained package for what's really
    just a "nice to have" middle fallback here, so it's dropped —
    yfinance first, ECB XML (triangulated through EUR) as the fallback,
    same as before minus that one dependency.

  - Lazy imports for yfinance and google-genai, matching the pattern in
    stock_analysis.py — neither loads into memory until a /gold route
    is actually hit, not at app boot.
"""

import os
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/gold", tags=["Gold"])

# =====================================================================
# Lazy-loaded heavy dependency: yfinance
# =====================================================================
yf = None
_yf_loaded = False


def _ensure_yfinance():
    global yf, _yf_loaded
    if _yf_loaded:
        return
    import yfinance as _yf
    yf = _yf
    _yf_loaded = True


# =====================================================================
# Config
# =====================================================================
GOLD_TICKER = "GC=F"               # COMEX Gold futures — primary source
GOLD_TICKER_FALLBACK = "XAUUSD=X"  # spot gold/USD — used if GC=F comes back empty

GRAMS_PER_TROY_OUNCE = 31.1035

GOLD_CACHE_TTL_SECONDS = 60 * 60      # 1 hour, matches the original app
CURRENCY_CACHE_TTL_SECONDS = 60 * 60  # 1 hour

# Last-resort static fallback if every live USD->INR source fails AND
# there's no cached rate to fall back to either. Deliberately NOT
# presented as a live rate to the frontend — see rate_is_estimated below.
# Update this occasionally; it only matters on a cold start where every
# other source has also failed.
USD_INR_EMERGENCY_FALLBACK = 88.0

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-flash-latest"

# =====================================================================
# In-memory caches (see module docstring for why not on-disk)
# =====================================================================
_gold_cache = {"data": None, "fetched_at": 0}
_currency_cache = {"rate": None, "fetched_at": 0}
_insights_cache = {"result": None}
_insights_loading = False


def _retry(fn, description, attempts=3, backoff_seconds=2):
    """Call fn() up to `attempts` times with a short backoff. Yahoo
    Finance and similar free sources intermittently rate-limit or
    return thin/empty data — most of these are transient (see the same
    pattern already used for stock.history() in stock_analysis.py)."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            result = fn()
            if result is not None:
                return result
            last_error = f"{description} returned no data"
        except Exception as e:
            last_error = str(e)
        if attempt < attempts:
            time.sleep(backoff_seconds * attempt)
    print(f"{description} failed after {attempts} attempts: {last_error}")
    return None


# =====================================================================
# USD -> INR exchange rate
# =====================================================================
def _get_ecb_usd_inr_rate():
    """ECB publishes EUR->X rates, not USD->INR directly, so this
    triangulates: (EUR->INR) / (EUR->USD) = USD->INR."""
    response = requests.get(
        "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml",
        timeout=5
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"ns": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}

    eur_to = {}
    for cube in root.findall(".//ns:Cube[@currency]", ns):
        eur_to[cube.attrib["currency"]] = float(cube.attrib["rate"])

    if "INR" in eur_to and "USD" in eur_to:
        return eur_to["INR"] / eur_to["USD"]
    return None


def get_usd_inr_rate():
    """Returns (rate, is_estimated)."""
    now = time.time()
    if _currency_cache["rate"] and (now - _currency_cache["fetched_at"] < CURRENCY_CACHE_TTL_SECONDS):
        return _currency_cache["rate"], False

    _ensure_yfinance()

    def _from_yfinance():
        hist = yf.Ticker("INR=X").history(period="1d")
        if hist.empty:
            return None
        return float(hist.iloc[-1]["Close"])

    rate = _retry(_from_yfinance, "USD/INR via yfinance")

    if rate is None:
        rate = _retry(_get_ecb_usd_inr_rate, "USD/INR via ECB XML (triangulated)")

    is_estimated = False
    if rate is None or not (50 < rate < 110):
        rate = _currency_cache["rate"] or USD_INR_EMERGENCY_FALLBACK
        is_estimated = True

    _currency_cache["rate"] = rate
    _currency_cache["fetched_at"] = now
    return rate, is_estimated


# =====================================================================
# Gold price
# =====================================================================
def _fetch_gold_history():
    _ensure_yfinance()

    def _try_ticker(symbol):
        data = yf.Ticker(symbol).history(period="30d")
        return None if data.empty else data

    data = _retry(lambda: _try_ticker(GOLD_TICKER), f"gold history ({GOLD_TICKER})")
    if data is None:
        data = _retry(lambda: _try_ticker(GOLD_TICKER_FALLBACK), f"gold history ({GOLD_TICKER_FALLBACK})")
    return data


def get_gold_data(force_refresh=False):
    """Returns (data_dict, success_bool, is_fresh_bool). is_fresh tells
    the caller whether this actually hit Yahoo Finance just now, or
    served the cached result — used so /price doesn't re-trigger a
    Gemini call on every single hit within the cache window (see the
    /price route below)."""
    now = time.time()
    if not force_refresh and _gold_cache["data"] and (now - _gold_cache["fetched_at"] < GOLD_CACHE_TTL_SECONDS):
        return _gold_cache["data"], True, False

    data = _fetch_gold_history()

    # Need at least 8 rows so the "7 trading days ago" comparison below
    # (data.iloc[-7]) is safe.
    if data is None or len(data) < 8:
        if _gold_cache["data"]:
            return _gold_cache["data"], True, False
        return {"success": False, "error": "Gold price data unavailable from any source right now."}, False, False

    current_price_per_ounce = float(data.iloc[-1]["Close"])

    # Sanity bound — wide enough to not reject legitimate prices as gold
    # rises over time, just enough to catch obviously broken data (e.g.
    # a stray 0 or a misparsed value).
    if not (100 < current_price_per_ounce < 10000):
        if _gold_cache["data"]:
            return _gold_cache["data"], True, False
        return {"success": False, "error": "Retrieved gold price is outside a plausible range."}, False, False

    usd_inr_rate, rate_is_estimated = get_usd_inr_rate()

    current_price_per_gram = current_price_per_ounce / GRAMS_PER_TROY_OUNCE
    historical_prices = data["Close"] / GRAMS_PER_TROY_OUNCE

    response_data = {
        "success": True,
        "usd_inr_rate": round(usd_inr_rate, 2),
        "rate_is_estimated": rate_is_estimated,
        "current_price": {
            "per_ounce_usd": round(current_price_per_ounce, 2),
            "per_gram_usd": round(current_price_per_gram, 2),
            "per_gram_inr": round(current_price_per_gram * usd_inr_rate, 2),
        },
        "karat_prices": {
            f"{k}K": {
                "USD": round(current_price_per_gram * (k / 24), 2),
                "INR": round(current_price_per_gram * usd_inr_rate * (k / 24), 2),
            }
            for k in [24, 22, 18, 14, 10]
        },
        "plot_data": {
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "prices": [round(p, 2) for p in historical_prices.tolist()],
        },
        "recommendation": "Positive" if data.iloc[-1]["Close"] > data.iloc[-7]["Close"] else "Negative",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    _gold_cache["data"] = response_data
    _gold_cache["fetched_at"] = now
    return response_data, True, True


# =====================================================================
# Gemini AI insights
# =====================================================================
def _generate_gemini_insights(gold_data):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

    from google import genai  # lazy import — see module docstring

    client = genai.Client(api_key=GEMINI_API_KEY)

    dates = gold_data.get("plot_data", {}).get("dates", [])
    date_range = f"{dates[0]} to {dates[-1]}" if dates else "the last 30 days"

    prompt = f"""You are a financial expert. Analyze the following gold price data:

Current Price: ${gold_data.get('current_price', {}).get('per_ounce_usd')} per ounce (${gold_data.get('current_price', {}).get('per_gram_usd')} per gram)
Recent Trend: {"Up" if gold_data.get("recommendation") == "Positive" else "Down"}
Date Range: {date_range}

Please provide:
1. A brief analysis of the current gold price trend
2. Potential factors influencing this movement
3. A short-term outlook (next 1-2 weeks)

Keep it concise (150-200 words) and avoid tables."""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip() if response and response.text else "No insights available."


def trigger_insights_update(gold_data):
    """Fire-and-forget background refresh — mirrors the FinBERT loading
    pattern in news.py (daemon thread + a loading flag), so a slow or
    failed Gemini call never blocks the /gold/price response that
    triggered it."""
    global _insights_loading

    if _insights_loading:
        return
    _insights_loading = True

    def _update():
        global _insights_loading
        try:
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

            # Gemini occasionally returns a 503 "high demand, try again
            # later" — that's an explicit signal it's transient, worth
            # a couple of retries before treating it as a real failure
            # (same reasoning as the yfinance retries above).
            insights_text = None
            last_error = None
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    insights_text = _generate_gemini_insights(gold_data)
                    break
                except Exception as e:
                    last_error = str(e)
                    is_transient = any(marker in last_error for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
                    print(f"Gemini insights attempt {attempt} failed: {last_error}")
                    if attempt < attempts and is_transient:
                        time.sleep(4 * attempt)
                        continue
                    raise

            _insights_cache["result"] = {
                "success": True,
                "insights": insights_text,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gold_price_at_analysis": gold_data.get("current_price", {}).get("per_ounce_usd"),
            }
        except Exception as e:
            print(f"Error updating Gemini insights: {e}")
            _insights_cache["result"] = {
                "success": False,
                "error": str(e),
                "last_attempt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        finally:
            _insights_loading = False

    threading.Thread(target=_update, daemon=True).start()


# =====================================================================
# Routes
# =====================================================================
@router.get("/price")
def gold_price(refresh: bool = Query(False, description="Force a fresh fetch, bypassing the 1-hour cache")):
    """Current gold price (per ounce/gram, USD+INR), karat breakdown,
    30-day plot data, and a simple trend recommendation.

    Auto-triggers a background Gemini insights refresh only when this
    call actually hit Yahoo Finance (a fresh fetch, or the very first
    call ever) — NOT on every hit that's served from the 1-hour price
    cache. Otherwise every repeat page load within that hour would
    silently re-trigger a new Gemini call for no reason. Poll
    /gold/insights separately to pick up the result once it's ready;
    use /gold/insights/refresh to manually regenerate insights without
    touching Yahoo Finance at all (e.g. to retry after a Gemini 503)."""
    data, success, is_fresh = get_gold_data(force_refresh=refresh)

    if success and data.get("success"):
        if is_fresh or _insights_cache["result"] is None:
            trigger_insights_update(data)

    return data


@router.get("/insights")
def gold_insights():
    """Latest AI-generated commentary on the gold price trend. Returns
    a 'processing' status until the background thread kicked off by
    /gold/price (or /gold/insights/refresh) finishes."""
    result = _insights_cache["result"]
    if not result:
        return {
            "success": False,
            "error": "No insights available yet.",
            "status": "processing" if _insights_loading else "not_started",
        }
    return result


@router.post("/insights/refresh")
def gold_insights_refresh():
    """Manually regenerate AI insights using whatever gold price data is
    currently cached — does NOT call Yahoo Finance at all, so this is
    safe to retry as many times as needed (e.g. after a Gemini 503)
    without adding any load to the yfinance/Yahoo side, which already
    has its own rate-limiting problems independent of this feature."""
    if not _gold_cache["data"]:
        raise HTTPException(
            status_code=400,
            detail="No gold price data cached yet — call /gold/price first."
        )
    trigger_insights_update(_gold_cache["data"])
    return {"success": True, "status": "processing"}
