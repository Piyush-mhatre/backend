"""
Investment Recommender — ported from the old Flask app's create_dataset(),
train_model() and get_recommendation() (landing.py).

Kept faithful: same 14-option dataset, same encoding, same Decision Tree,
same way of picking up to 3 alternatives. Changes from the original:

  - pandas is gone. DecisionTreeClassifier only needs numpy arrays, so the
    dataset is a plain list of dicts. The only new dependency is scikit-learn.
  - scikit-learn/numpy are imported lazily, on the first request, so they
    cost nothing at app boot (same pattern as gold.py / stock_analysis.py).
  - The model is trained once and cached in _clf, never per request.
  - Every option returned (the pick AND the alternatives) carries real
    per-factor reasons, including honest mismatches — the old app returned
    bare data with no explanation.

Honest framing: the tree has 14 samples and 14 classes (one per option), so
it behaves like a nearest-match lookup over 4 features rather than a model
that generalizes from real-world data. Fine for a demo; the frontend says so.
No external APIs are called, so no caching/cooldown machinery is needed.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/investment", tags=["Investment Recommender"])

_COLUMNS = ("name", "return_rate", "risk_level", "tax_benefits", "liquidity", "min_duration", "description")
_ROWS = [
    ("Mutual Funds (Large Cap)", 15.01, "medium", "no", "high", 3, "Large cap mutual funds investing in established companies"),
    ("Mutual Funds (Mid Cap)", 22.59, "medium", "no", "high", 3, "Mid cap mutual funds with higher growth potential"),
    ("Mutual Funds (Small Cap)", 26.81, "medium", "no", "high", 3, "Small cap mutual funds with highest growth potential but volatility"),
    ("PPF", 7.1, "low", "yes", "medium", 15, "Public Provident Fund - government backed long-term savings"),
    ("NPS", 10.5, "low", "yes", "low", 20, "National Pension System - retirement focused investment"),
    ("Fixed Deposits", 7.8, "low", "yes", "low", 1, "Fixed deposits with guaranteed returns from banks"),
    ("REIT", 22.5, "medium", "no", "high", 3, "Real Estate Investment Trust with property based returns"),
    ("Corporate Bonds", 9.0, "low", "no", "medium", 3, "Corporate bonds with fixed interest rates"),
    ("Gold", 10.5, "medium", "no", "high", 1, "Gold investments (physical, ETF, or bonds)"),
    ("Equity Stocks (Large Cap)", 7.0, "high", "no", "high", 1, "Large cap stocks of established companies"),
    ("Equity Stocks (Mid Cap)", 10.28, "high", "no", "high", 1, "Mid cap stocks with growth potential"),
    ("Equity Stocks (Small Cap)", 14.74, "high", "no", "high", 1, "Small cap stocks with highest risk/reward"),
    ("Government Bonds", 6.67, "low", "yes", "medium", 5, "Government bonds with sovereign guarantee"),
    ("SIP", 15.0, "medium", "no", "high", 1, "Systematic Investment Plan for regular investments in mutual funds"),
]
OPTIONS = [dict(zip(_COLUMNS, row)) for row in _ROWS]

RISK = {"low": 0, "medium": 1, "high": 2}
LIQUIDITY = {"low": 0, "medium": 1, "high": 2}
TAX = {"no": 0, "yes": 1}

_clf = None  # trained lazily on first request, then reused


def _features(risk, tax, liquidity, duration):
    return [RISK[risk], TAX[tax], LIQUIDITY[liquidity], duration]


def _get_clf():
    global _clf
    if _clf is None:
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier

        X = np.array([_features(o["risk_level"], o["tax_benefits"], o["liquidity"], o["min_duration"]) for o in OPTIONS])
        y = np.arange(len(OPTIONS))  # target = row index, exactly like the original
        _clf = DecisionTreeClassifier(random_state=42).fit(X, y)
    return _clf


class RecommendRequest(BaseModel):
    risk: Literal["low", "medium", "high"] = "medium"
    tax: Literal["yes", "no"] = "no"
    liquidity: Literal["low", "medium", "high"] = "medium"
    duration: int = Field(5, ge=1, le=40, description="Investment horizon in years")


def _fallback_alternatives(excluded, tax_code, risk_code, need):
    """Original fallback order: same tax-benefit match -> same risk match -> next rows."""
    rest = [i for i in range(len(OPTIONS)) if i not in excluded]
    tax_matches = [i for i in rest if TAX[OPTIONS[i]["tax_benefits"]] == tax_code]
    if tax_matches:
        pool = [i for i in tax_matches if RISK[OPTIONS[i]["risk_level"]] == risk_code] or tax_matches
    else:
        pool = [i for i in rest if RISK[OPTIONS[i]["risk_level"]] == risk_code] or rest
    return pool[:need]


def _explain(opt, req):
    """Per-factor reasons (match True/False) for one option against the user's profile."""
    reasons = []

    same = opt["risk_level"] == req.risk
    reasons.append({"factor": "Risk", "match": same, "text": (
        f"Risk level is {opt['risk_level']}, matching your {req.risk} risk tolerance." if same
        else f"Its risk level is {opt['risk_level']}, whereas you chose {req.risk} risk.")})

    has_tax = opt["tax_benefits"] == "yes"
    if req.tax == "yes":
        reasons.append({"factor": "Tax", "match": has_tax, "text": (
            "Offers tax benefits, as you wanted." if has_tax
            else "It has no tax benefits, whereas you wanted them.")})
    else:
        reasons.append({"factor": "Tax", "match": True, "text": (
            "Comes with tax benefits, a bonus since you didn't require them." if has_tax
            else "No tax benefits, which is fine since you didn't require them.")})

    same = opt["liquidity"] == req.liquidity
    reasons.append({"factor": "Liquidity", "match": same, "text": (
        f"Liquidity is {opt['liquidity']}, matching your preference." if same
        else f"Its liquidity is {opt['liquidity']}, whereas you preferred {req.liquidity} liquidity.")})

    fits = opt["min_duration"] <= req.duration
    reasons.append({"factor": "Horizon", "match": fits, "text": (
        f"Suggested minimum holding period is {opt['min_duration']} year(s), which fits your {req.duration}-year horizon." if fits
        else f"It needs at least {opt['min_duration']} years, longer than your {req.duration}-year horizon.")})

    return {**opt, "reasons": reasons, "matched": sum(r["match"] for r in reasons)}


@router.post("/recommend")
def recommend(payload: RecommendRequest):
    """Returns the best-matching option plus up to 3 alternatives, each with reasons."""
    try:
        clf = _get_clf()
        x = [_features(payload.risk, payload.tax, payload.liquidity, payload.duration)]
        best = int(clf.predict(x)[0])

        ranked = sorted(enumerate(clf.predict_proba(x)[0]), key=lambda p: p[1], reverse=True)
        alt_idx = [int(i) for i, prob in ranked[1:4] if prob > 0]
        if len(alt_idx) < 3:
            alt_idx += _fallback_alternatives(
                alt_idx + [best], TAX[payload.tax], RISK[payload.risk], 3 - len(alt_idx))

        return {
            "profile": {
                "risk_tolerance": payload.risk.capitalize(),
                "tax_benefits": payload.tax.capitalize(),
                "liquidity_preference": payload.liquidity.capitalize(),
                "investment_horizon": payload.duration,
            },
            "recommendation": _explain(OPTIONS[best], payload),
            "alternatives": [_explain(OPTIONS[i], payload) for i in alt_idx[:3]],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommender error: {e}")
