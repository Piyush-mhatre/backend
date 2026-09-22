"""
Portfolio backend API — starts with just the /calculate endpoint that
powers the "Explainable AI Financial Advisor" project's finplan demo.

This file intentionally does NOT use any API keys. calculate_investments()
below is pure Python arithmetic (compound-growth projection), ported
directly from the original Flask app with no behavior changes. Future
features (gold prices, the chatbot, news sentiment, stock analysis) will
each add their own route + their own environment variable for whatever
external key they need — see the "ADDING A NEW FEATURE LATER" note at
the bottom of this file for the pattern to follow.
"""

import os
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from routers import stock_analysis
from routers import news
from routers import gold

app = FastAPI(title="Piyush Mhatre — Portfolio Backend", version="0.2.0")
app.include_router(stock_analysis.router)
app.include_router(news.router)
app.include_router(gold.router)

# =====================================================================
# CORS — only these origins are allowed to call this API from a browser.
# Add your real Vercel domain(s) here. Keep localhost entries for local
# development; remove them later if you want, they're harmless either way
# since this API has no secrets behind it for now.
# =====================================================================
ALLOWED_ORIGINS = [
    "https://piyushsmportfolio.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:5500",   # VS Code "Live Server" default, if you use it
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =====================================================================
# Request validation — Pydantic rejects malformed requests automatically
# (e.g. negative income, missing fields) before your code ever runs.
# =====================================================================
class FinPlanRequest(BaseModel):
    income: float = Field(..., gt=0, description="Annual income in INR")
    age: int = Field(..., ge=18, le=90)
    salaryGrowth: float = Field(..., ge=0, le=50, description="Percent, e.g. 8 for 8%")
    investmentPercentage: float = Field(..., gt=0, le=100)
    riskProfile: str = Field(default="mid")  # accepted for future use — see note below


# =====================================================================
# Core calculation — ported as-is from the original Flask app.
# Pure Python, no external calls, no API keys touched.
#
# NOTE: riskProfile is accepted from the frontend but not yet used to
# change the allocation percentages below (this matches the original
# app's behavior — it wasn't wired up there either). If you want "Low
# risk" vs "High risk" to actually produce different numbers, that's a
# real follow-up: branch the `allocation` dict on payload.riskProfile.
# =====================================================================
def calculate_investments(annual_income, age, salary_growth, investment_percentage):
    """Project investment growth across 13 categories over 30 years."""
    years = range(age, age + 30)

    salary_growth = salary_growth / 100
    investment_percentage = investment_percentage / 100

    emergency_fund = min(annual_income * 0.5, 1000000)

    investments = {
        'Emergency Fund': [emergency_fund] * len(years),
        'Mutual Funds - Large Cap': [],
        'Mutual Funds - Mid Cap': [],
        'Mutual Funds - Small Cap': [],
        'PPF': [],
        'NPS': [],
        'Fixed Deposits': [],
        'REIT (Real Estate)': [],
        'Corporate Bonds': [],
        'Gold': [],
        'Equity Stocks': [],
        'Government Bonds': [],
        'SIP': []
    }

    risk_factors = {
        'Emergency Fund': 1,
        'Mutual Funds - Large Cap': 2,
        'Mutual Funds - Mid Cap': 2,
        'Mutual Funds - Small Cap': 2,
        'PPF': 1,
        'NPS': 1,
        'Fixed Deposits': 1,
        'REIT (Real Estate)': 2,
        'Corporate Bonds': 1,
        'Gold': 2,
        'Equity Stocks': 3,
        'Government Bonds': 1,
        'SIP': 2
    }

    growth_rates = {
        'Emergency Fund': 0,
        'Mutual Funds - Large Cap': 0.1501,
        'Mutual Funds - Mid Cap': 0.2259,
        'Mutual Funds - Small Cap': 0.2681,
        'PPF': 0.071,
        'NPS': 0.105,
        'Fixed Deposits': 0.078,
        'REIT (Real Estate)': 0.12,
        'Corporate Bonds': 0.09,
        'Gold': 0.105,
        'Equity Stocks': 0.1067,
        'Government Bonds': 0.0667,
        'SIP': 0.15
    }

    allocation = {
        'Mutual Funds - Large Cap': 0.15,
        'Mutual Funds - Mid Cap': 0.10,
        'Mutual Funds - Small Cap': 0.05,
        'PPF': 0.10,
        'NPS': 0.10,
        'Fixed Deposits': 0.10,
        'REIT (Real Estate)': 0.10,
        'Corporate Bonds': 0.08,
        'Gold': 0.07,
        'Equity Stocks': 0.05,
        'Government Bonds': 0.05,
        'SIP': 0.05
    }

    cumulative_investment = 0
    annual_investments = {inv_type: [] for inv_type in allocation.keys()}

    for year in range(len(years)):
        if year > 0:
            annual_income = annual_income * (1 + salary_growth)

        annual_savings = annual_income * investment_percentage
        cumulative_investment += annual_savings

        for investment_type, alloc_percentage in allocation.items():
            current_investment = annual_savings * alloc_percentage

            if year > 0:
                previous_value = investments[investment_type][year - 1]
                growth_rate = growth_rates[investment_type]
                current_value = previous_value * (1 + growth_rate) + current_investment
            else:
                current_value = current_investment

            if investment_type == 'REIT (Real Estate)' and year < 3:
                current_value = 0

            investments[investment_type].append(current_value)
            annual_investments[investment_type].append(current_investment)

    return {
        'years': list(years),
        'investments': investments,
        'risk_factors': risk_factors,
        'annual_investments': annual_investments,
        'growth_rates': growth_rates
    }


# =====================================================================
# ROUTES
# =====================================================================
@app.get("/")
def root():
    """Simple root route so visiting the base URL doesn't 404 — also
    useful as the target for an uptime-pinger (see backend README)."""
    return {"status": "ok", "service": "piyush-portfolio-backend"}


@app.on_event("startup")
def prewarm_forecast_engine():
    """Equivalent of the original app's 'start loading the model as soon
    as the landing page is opened' behavior — adapted for how this is
    actually deployed. On Render's free tier, the process only *starts*
    when something wakes it from a cold sleep — and that wake-up is
    exactly what the portfolio's warmup.js ping triggers the moment a
    visitor lands on any page. So "on server startup" here already lines
    up with "as soon as the portfolio is visited," without needing any
    separate mechanism.

    This only does anything if FORECAST_ENGINE=prophet is set — Prophet's
    own import is the expensive part (it initializes its Stan/cmdstanpy
    backend), so doing that import once here, in a background thread,
    means it's already paid for by the time a real /analyze request
    arrives, instead of the first visitor eating that cost.
    """
    if os.environ.get("FORECAST_ENGINE", "trend").lower() != "prophet":
        return  # lightweight engine active — nothing heavy to pre-load

    def _warm():
        try:
            from prophet import Prophet  # noqa: F401 — import side-effect is the point
            print("Prophet pre-imported at startup (FORECAST_ENGINE=prophet)")
        except Exception as e:
            print(f"Prophet pre-warm failed: {e}")

    threading.Thread(target=_warm, daemon=True).start()


@app.on_event("startup")
def prewarm_finbert():
    """Kicks off the INT8 FinBERT load in a background thread as soon as
    the app starts. news.start_model_loading() is safe to call exactly
    once here — it spins up load_finbert_model() on its own daemon
    thread and is a no-op if called again while already loading/loaded."""
    news.start_model_loading()


@app.get("/health")
def health():
    """Dedicated health-check route for uptime monitoring services."""
    return {"status": "healthy"}


@app.post("/calculate")
def calculate(payload: FinPlanRequest):
    """Powers the finplan demo: projects a 30-year investment plan from
    income, age, salary growth %, and investment %."""
    try:
        result = calculate_investments(
            payload.income,
            payload.age,
            payload.salaryGrowth,
            payload.investmentPercentage,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# ADDING A NEW FEATURE LATER (gold / chatbot / news / stock-analysis)
# Follow this pattern for each one:
#
#   1. Add the real key to Render's dashboard → Environment → new
#      variable (e.g. GEMINI_API_KEY). Never write the real value here.
#   2. Read it in code with: os.environ["GEMINI_API_KEY"]
#      (use os.environ.get(...) with NO hardcoded fallback value —
#      a fallback default defeats the whole point, see the security
#      note in the backend README).
#   3. Add a new Pydantic request model + route, same shape as
#      /calculate above.
#   4. The frontend only ever calls YOUR backend route — it never sees
#      the real key, because the key never leaves this server.
# =====================================================================
