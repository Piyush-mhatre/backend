# Portfolio Backend API — Explainable AI Financial Advisor

FastAPI backend powering the **Explainable AI Financial Advisor** portfolio by **Piyush Mhatre**. Deployed on Render's free tier, designed with lazy imports, in-memory TTL caches, and zero unnecessary startup cost.

---

## Tech Stack

| Layer | Library / Tool |
| :--- | :--- |
| Framework | FastAPI + Uvicorn (Python 3.11) |
| Validation | Pydantic v2 |
| NLP / Sentiment | ONNX Runtime · INT8-quantized ProsusAI FinBERT · HuggingFace Tokenizers |
| Generative AI | Google GenAI SDK (`google-genai`) |
| Market Data | Yahoo Finance (`yfinance`) · ECB XML · Financial Modeling Prep (FMP) |
| ML / Forecasting | Scikit-Learn · Optional Facebook Prophet · Linear Volatility engine |
| Visualization | Plotly (charts serialized to JSON for the frontend) |

---

## Project Structure

```
backend/
├── main.py                     # App entry point, CORS, /calculate, startup hooks
├── requirements.txt
├── runtime.txt                 # python-3.11.9
├── models/
│   ├── finbert_int8.onnx       # INT8-quantized FinBERT (~105 MB, downloaded at build)
│   ├── tokenizer/              # Local tokenizer files committed to git (~1-2 MB)
│   └── export_to_onnx.py       # One-time local script to produce the .onnx file
└── routers/
    ├── mem_log.py              # Memory diagnostic helper (Linux/Render only)
    ├── gemini_shared.py        # Shared Gemini quota management for gold + chatbot
    ├── stock_analysis.py       # /analyze — Piotroski, forecast, candlestick, sector
    ├── news.py                 # /news/{keyword} — FinBERT sentiment on NewsAPI articles
    ├── gold.py                 # /gold/price · /gold/insights — live gold + Gemini commentary
    ├── chatbot.py              # /chatbot/message — multi-turn financial advisor
    └── fin_invest.py           # /investment/recommend — Decision Tree recommender
```

---

## Features

### `POST /calculate` — Investment Plan Projection (`main.py`)
Pure Python compound-growth projection of 13 asset categories (Mutual Funds, PPF, NPS, Fixed Deposits, REITs, Bonds, Gold, Equity, SIP) over 30 years. No external API. No heavy libraries. Runs instantly from a cold start.

---

### `POST /analyze` — Stock Analysis (`routers/stock_analysis.py`)

All four sub-analyses share one `yf.Ticker` fetch to avoid redundant Yahoo Finance calls.

| Sub-analysis | What it does |
| :--- | :--- |
| **Piotroski F-Score** | 9-point financial health check (ROA, Operating CF, Leverage, Current Ratio, Share Dilution, Gross Margin, Asset Turnover) with plain-English explanations per criterion |
| **Forecast** | Dual-engine, toggled by `FORECAST_ENGINE` env var: `trend` (default) = fast linear regression + 95% volatility band; `prophet` = Facebook Prophet seasonality model |
| **Candlestick + Info** | 5-year Plotly candlestick chart; company profile from FMP API + yfinance fallback |
| **Sector Performance** | Stock's % change vs. sector ETF benchmarks, top/bottom sector ranking |

**Memory note:** `numpy`, `pandas`, `yfinance`, `plotly` are **lazy-imported** on the first `/analyze` request, not at boot. The `_ensure_heavy_libs()` guard prevents re-importing on subsequent calls.

---

### `GET /news/{keyword}` — News Sentiment Analysis (`routers/news.py`)

1. Calls NewsAPI with exact-phrase matching (`"keyword"`) scoped to `title,description`, sorted by relevance.
2. Falls back to a broader exact-phrase search if fewer than 3 articles match the precise pass.
3. Classifies each article's description (first 100 chars) with the INT8 FinBERT ONNX model.
4. Returns up to 10 articles with `positive` / `negative` / `neutral` labels.

**Memory note:** FinBERT replaced PyTorch (~250 MB baseline) with ONNX Runtime (~35 MB runtime footprint). The `InferenceSession` is pre-warmed on a **background daemon thread at startup** via `start_model_loading()`, so the first real request doesn't pay the load cost.

---

### `GET /gold/price` · `GET /gold/insights` · `POST /gold/insights/refresh` — Gold Rates (`routers/gold.py`)

- **Price**: Fetches COMEX `GC=F` (fallback: `XAUUSD=X`) via yfinance. Triangulates USD→INR via ECB XML (yfinance `INR=X` first, ECB as fallback). Returns per-gram/ounce prices in USD + INR, 5 karat breakdowns (10K–24K), and 30-day Plotly chart data. Results are cached in-memory for 1 hour.
- **Insights**: On each fresh price fetch (or manual `/refresh`), fires a background thread that races candidate Gemini models concurrently (first success wins) to generate 150–200 word AI market commentary. Cached separately — polling `/insights` picks it up once ready.
- **`/refresh`**: Regenerates Gemini insights from cached gold data without touching Yahoo Finance.

---

### `POST /chatbot/message` · `GET /chatbot/limit/{client_id}` — Financial Chatbot (`routers/chatbot.py`)

- **Stateless multi-turn**: The caller sends the full conversation history on every request; nothing is stored server-side between requests.
- **Conversation titles**: First message of a new conversation asks Gemini to prepend a `TITLE: …` line; `_split_title()` peels it off before the reply reaches the frontend, saving a separate title-generation API call.
- **Per-session daily cap**: 18 messages/day per `client_id` (a UUID stored in browser `localStorage`). Enforced in-memory; resets on redeploy/restart.
- **Sequential model fallback**: Tries Gemini models one at a time (not concurrently) to conserve daily quota — a multi-turn chat would burn 3× quota per message if racing.

---

### `POST /investment/recommend` — Investment Recommender (`routers/fin_invest.py`)

Decision Tree classifier (scikit-learn) over a 14-option dataset matching user inputs:

| Input | Options |
| :--- | :--- |
| `risk` | `low` · `medium` · `high` |
| `tax` | `yes` · `no` |
| `liquidity` | `low` · `medium` · `high` |
| `duration` | 1–40 years |

Returns the best match + up to 3 alternatives, each with per-factor explanations (risk match, tax benefit, liquidity fit, horizon check) — including honest mismatches. Scikit-learn and numpy are **lazy-imported** on first request. The fitted model (`_clf`) is cached in memory and reused for all subsequent calls.

---

### `routers/gemini_shared.py` — Shared Gemini Infrastructure

Both `gold.py` and `chatbot.py` draw from the same Google AI daily quota. This module centralises:

- **Candidate model list**: `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.5-flash`, … (ordered by preference)
- **Per-model cooldown tracking**: A 429 quota-exhaustion marks a model as unavailable for 4 hours (or however long the API says). A 404 marks it for 24 hours. Both features read and write the same `_model_cooldown_until` dict.
- **`race_gemini_models()`**: Fires models in concurrent batches of 3 — first success wins. Used by `gold.py` for occasional insights refreshes.
- **`call_gemini_sequential()`**: Tries models one at a time. Used by `chatbot.py` to avoid burning 3× quota per chat message.

---

## Memory Diagnostics — `routers/mem_log.py`

`log_memory(label)` uses `resource.getrusage(RUSAGE_SELF).ru_maxrss` to print the **process peak RSS high-water mark** (in MB) to stdout at labelled checkpoints throughout the app. Look for `[memory]` lines in Render's log stream.

> **Why peak RSS, not current usage?** On Linux, `ru_maxrss` is a one-way ratchet — it only ever goes up for the life of the process. C-extension libraries like numpy, pandas, onnxruntime, and plotly release Python objects but the OS may not immediately reclaim the underlying pages. Peak RSS therefore tells you the worst this process has consumed *so far*, making it ideal for tracing which feature pushes you towards Render's 512 MB ceiling.

> **Windows dev note:** The `resource` module is Linux-only. On Windows the helper prints a safe no-op message and returns immediately — your local dev server will not crash.

### Checkpoint map

| Label | Where | What it tells you |
| :--- | :--- | :--- |
| `main.py: App boot & routers mounted` | Startup | Baseline RSS before any request |
| `main.py: FinBERT prewarm kicked off` | Startup | RSS just as FinBERT background thread starts |
| `main.py: Prophet pre-warm finished` | Startup (if `FORECAST_ENGINE=prophet`) | Cost of importing Prophet's Stan backend |
| `main.py: after /calculate` | `/calculate` | Pure-Python route — should show no growth |
| `news: FinBERT tokenizer loaded` | Background thread | Cost of loading local tokenizer files |
| `news: ONNX INT8 session initialized` | Background thread | Cost of loading the 105 MB ONNX model into RAM |
| `news: sentiment batch classification finished` | `/news/{keyword}` | Inference cost on up to 10 articles |
| `news: /news/{keyword} completed` | `/news/{keyword}` | Total cost of the news route end-to-end |
| `gold: yfinance imported` | First `/gold/price` | One-time cost of importing yfinance |
| `gold: price & exchange data processed` | `/gold/price` | After Yahoo fetch + ECB triangulation |
| `gemini_shared: google.genai imported for race` | Background insights thread | One-time cost of importing google-genai |
| `gold: Gemini insights generated` | Background thread | After Gemini responds |
| `gold: /gold/price completed` | `/gold/price` | Total route cost |
| `gemini_shared: google.genai imported for sequential` | First `/chatbot/message` | One-time import cost |
| `chatbot: Gemini response received` | `/chatbot/message` | After each Gemini reply |
| `stock_analysis: heavy libs imported` | First `/analyze` | numpy + pandas + yfinance + plotly all at once |
| `stock_analysis: price history fetched for {ticker}` | `/analyze` | After yfinance history pull |
| `stock_analysis: Prophet model fit complete` | `/analyze` (prophet only) | After fitting the Prophet model |
| `stock_analysis: /analyze full pipeline completed` | `/analyze` | After all 4 sub-analyses |
| `fin_invest: Scikit-learn imported & DecisionTree fitted` | First `/investment/recommend` | One-time sklearn + numpy import + fit |
| `fin_invest: /investment/recommend completed` | `/investment/recommend` | Per-request cost |
