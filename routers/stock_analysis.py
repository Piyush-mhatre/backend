"""
Stock analysis feature: Piotroski F-Score, price forecast, candlestick +
company info, and sector performance — ported from the original Flask app.

RAM MANAGEMENT (read this before deploying):
This module never imports `torch` (that's only needed for the chatbot/news
BERT model, in a different feature entirely) — that alone avoids the single
biggest memory risk.

The forecast itself can run on one of two engines, controlled by the
FORECAST_ENGINE environment variable in Render:

  FORECAST_ENGINE=trend    (default, safe)
    A lightweight linear-trend + volatility-band projection. Cheap in
    memory and CPU, no heavy dependency. Good default until you've
    confirmed Prophet fits comfortably in Render's free 512MB.

  FORECAST_ENGINE=prophet  (heavier, opt-in)
    The original Facebook Prophet model. Genuinely useful, but Prophet's
    own documentation and community reports suggest it wants meaningfully
    more memory than 512MB to run comfortably. Try it, watch Render's
    memory graph after a few real requests, and switch back to "trend"
    instantly (just change the env var, no redeploy needed) if it's
    hovering near the limit or getting OOM-killed.

Both engines return the exact same response shape, so the frontend never
needs to know or care which one produced a given forecast.
"""

import os
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# How much price history to pull for the candlestick chart and the trend/
# Prophet forecast. The original app fetched candlestick data all the way
# back to 2001 for every ticker — for an old stock that's ~6,000 daily
# candles, which is a real memory/payload cost for no benefit on a demo
# site. 5 years keeps the chart meaningful while cutting that dramatically.
HISTORY_PERIOD = "5y"


# =====================================================================
# Piotroski F-Score
# =====================================================================
def safe_financial_value(df, metric, index=0):
    try:
        value = df.loc[metric].iloc[index]
        if pd.isna(value) or value is None:
            return None, f"Missing value for {metric}"
        return float(value), None
    except (KeyError, IndexError, AttributeError) as e:
        return None, f"Unable to fetch {metric}: {str(e)}"


def piotroski_score_detailed(stock):
    try:
        bs = stock.balance_sheet
        inc = stock.financials
        cf = stock.cash_flow

        scores = {}
        missing_data = []
        explanations = {}

        def calculate_ratio(numerator, denominator, metric_name):
            if numerator is None or denominator is None or denominator == 0:
                missing_data.append(metric_name)
                return None
            return numerator / denominator

        net_income, _ = safe_financial_value(inc, "Net Income")
        total_assets, _ = safe_financial_value(bs, "Total Assets")

        if net_income is not None and total_assets is not None:
            roa = net_income / total_assets
            scores["ROA"] = 1 if net_income > 0 else 0
            explanations["ROA"] = (
                f"Return on Assets (ROA) is {'positive' if net_income > 0 else 'negative'} "
                f"at {roa:.2%}. This {'indicates' if net_income > 0 else 'does not indicate'} "
                f"profitability relative to assets."
            )
        else:
            scores["ROA"] = 0
            missing_data.append("Net Income or Total Assets")
            explanations["ROA"] = "Could not calculate ROA due to missing data."

        op_cash_flow, _ = safe_financial_value(cf, "Operating Cash Flow")
        if op_cash_flow is not None:
            scores["Operating CF"] = 1 if op_cash_flow > 0 else 0
            explanations["Operating CF"] = (
                f"Operating Cash Flow is {'positive' if op_cash_flow > 0 else 'negative'} "
                f"at ${op_cash_flow/1e9:.2f}B."
            )
        else:
            scores["Operating CF"] = 0
            missing_data.append("Operating Cash Flow")
            explanations["Operating CF"] = "Could not evaluate Operating Cash Flow due to missing data."

        net_income_prev, _ = safe_financial_value(inc, "Net Income", 1)
        total_assets_prev, _ = safe_financial_value(bs, "Total Assets", 1)
        roa_current = calculate_ratio(net_income, total_assets, "Current ROA")
        roa_prev = calculate_ratio(net_income_prev, total_assets_prev, "Previous ROA")

        if roa_current is not None and roa_prev is not None:
            scores["ROA Change"] = 1 if roa_current > roa_prev else 0
            explanations["ROA Change"] = (
                f"ROA {'improved' if roa_current > roa_prev else 'declined'} "
                f"from {roa_prev:.2%} to {roa_current:.2%}."
            )
        else:
            scores["ROA Change"] = 0
            explanations["ROA Change"] = "Could not calculate ROA change due to missing data."

        if op_cash_flow is not None and net_income is not None:
            scores["Quality of Earnings"] = 1 if op_cash_flow > net_income else 0
            explanations["Quality of Earnings"] = (
                f"Operating Cash Flow ${op_cash_flow/1e9:.2f}B is "
                f"{'greater' if op_cash_flow > net_income else 'less'} than Net Income ${net_income/1e9:.2f}B."
            )
        else:
            scores["Quality of Earnings"] = 0
            explanations["Quality of Earnings"] = "Could not evaluate earnings quality due to missing data."

        ltd_current, _ = safe_financial_value(bs, "Long Term Debt")
        ltd_prev, _ = safe_financial_value(bs, "Long Term Debt", 1)
        if ltd_current is not None and ltd_prev is not None:
            scores["Leverage"] = 1 if ltd_current <= ltd_prev else 0
            explanations["Leverage"] = (
                f"Long-term debt {'decreased or remained stable' if ltd_current <= ltd_prev else 'increased'} "
                f"from ${ltd_prev/1e9:.2f}B to ${ltd_current/1e9:.2f}B."
            )
        else:
            scores["Leverage"] = 0
            missing_data.append("Long Term Debt")
            explanations["Leverage"] = "Could not evaluate leverage change due to missing data."

        ca_current, _ = safe_financial_value(bs, "Current Assets")
        ca_prev, _ = safe_financial_value(bs, "Current Assets", 1)
        cl_current, _ = safe_financial_value(bs, "Current Liabilities")
        cl_prev, _ = safe_financial_value(bs, "Current Liabilities", 1)
        curr_ratio_current = calculate_ratio(ca_current, cl_current, "Current Ratio")
        curr_ratio_prev = calculate_ratio(ca_prev, cl_prev, "Previous Current Ratio")

        if curr_ratio_current is not None and curr_ratio_prev is not None:
            scores["Current Ratio"] = 1 if curr_ratio_current > curr_ratio_prev else 0
            explanations["Current Ratio"] = (
                f"Current ratio {'improved' if curr_ratio_current > curr_ratio_prev else 'declined'} "
                f"from {curr_ratio_prev:.2f} to {curr_ratio_current:.2f}."
            )
        else:
            scores["Current Ratio"] = 0
            explanations["Current Ratio"] = "Could not calculate current ratio change due to missing data."

        shares_current, _ = safe_financial_value(bs, "Share Issued")
        shares_prev, _ = safe_financial_value(bs, "Share Issued", 1)
        if shares_current is not None and shares_prev is not None:
            scores["Share Dilution"] = 1 if shares_current <= shares_prev else 0
            explanations["Share Dilution"] = (
                f"Number of shares {'decreased or remained stable' if shares_current <= shares_prev else 'increased'} "
                f"from {shares_prev:.0f} to {shares_current:.0f}."
            )
        else:
            scores["Share Dilution"] = 0
            missing_data.append("Share Information")
            explanations["Share Dilution"] = "Could not evaluate share dilution due to missing data."

        gp_current, _ = safe_financial_value(inc, "Gross Profit")
        gp_prev, _ = safe_financial_value(inc, "Gross Profit", 1)
        rev_current, _ = safe_financial_value(inc, "Total Revenue")
        rev_prev, _ = safe_financial_value(inc, "Total Revenue", 1)
        gm_current = calculate_ratio(gp_current, rev_current, "Current Gross Margin")
        gm_prev = calculate_ratio(gp_prev, rev_prev, "Previous Gross Margin")

        if gm_current is not None and gm_prev is not None:
            scores["Gross Margin"] = 1 if gm_current > gm_prev else 0
            explanations["Gross Margin"] = (
                f"Gross margin {'improved' if gm_current > gm_prev else 'declined'} "
                f"from {gm_prev:.2%} to {gm_current:.2%}."
            )
        else:
            scores["Gross Margin"] = 0
            explanations["Gross Margin"] = "Could not calculate gross margin change due to missing data."

        at_current = calculate_ratio(rev_current, total_assets, "Current Asset Turnover")
        at_prev = calculate_ratio(rev_prev, total_assets_prev, "Previous Asset Turnover")
        if at_current is not None and at_prev is not None:
            scores["Asset Turnover"] = 1 if at_current > at_prev else 0
            explanations["Asset Turnover"] = (
                f"Asset turnover {'improved' if at_current > at_prev else 'declined'} "
                f"from {at_prev:.2f} to {at_current:.2f}."
            )
        else:
            scores["Asset Turnover"] = 0
            explanations["Asset Turnover"] = "Could not calculate asset turnover change due to missing data."

        return scores, missing_data, explanations, {
            "net_income": net_income,
            "op_cash_flow": op_cash_flow,
            "total_assets": total_assets,
            "revenue": rev_current,
        }
    except Exception as e:
        traceback.print_exc()
        return None, [f"Error in calculation: {str(e)}"], None, None


def analyze_piotroski(stock, ticker):
    try:
        scores, missing_data, explanations, metrics = piotroski_score_detailed(stock)
        if scores is None:
            return {"success": False, "error": "Failed to retrieve financial data"}

        total_score = sum(scores.values())
        if total_score >= 8:
            overall_interpretation = "Excellent financial health. Strong candidate for investment consideration."
        elif total_score >= 6:
            overall_interpretation = "Good financial health. Potential investment opportunity."
        elif total_score >= 4:
            overall_interpretation = "Average financial health. Further investigation recommended."
        elif total_score >= 2:
            overall_interpretation = "Below average financial health. Caution advised."
        else:
            overall_interpretation = "Poor financial health. High risk investment."

        score_details = [
            {"criterion": c, "score": s, "explanation": explanations.get(c, "No explanation available")}
            for c, s in scores.items()
        ]

        formatted_metrics = {}
        if metrics:
            if metrics["net_income"] is not None:
                formatted_metrics["Net Income"] = f"${metrics['net_income']/1e9:.2f}B"
            if metrics["op_cash_flow"] is not None:
                formatted_metrics["Operating Cash Flow"] = f"${metrics['op_cash_flow']/1e9:.2f}B"
            if metrics["total_assets"] is not None:
                formatted_metrics["Total Assets"] = f"${metrics['total_assets']/1e9:.2f}B"
            if metrics["revenue"] is not None:
                formatted_metrics["Total Revenue"] = f"${metrics['revenue']/1e9:.2f}B"

        return {
            "success": True,
            "ticker": ticker,
            "total_score": total_score,
            "interpretation": overall_interpretation,
            "score_details": score_details,
            "missing_data": missing_data,
            "metrics": formatted_metrics,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# =====================================================================
# FORECAST — two interchangeable engines, same response shape
# =====================================================================
def _lightweight_trend_forecast(history_df, company_name, ticker, currency):
    """Cheap fallback: linear trend + a volatility-based uncertainty band.
    No heavy dependency, near-instant, safe on any RAM budget."""
    dfx = history_df.reset_index()
    dfx["ds"] = pd.to_datetime(dfx[dfx.columns[0]]).dt.tz_localize(None)
    dfx["y"] = dfx["Close"].values

    x = np.arange(len(dfx))
    slope, intercept = np.polyfit(x, dfx["y"].values, 1)

    future_periods = 365
    future_x = np.arange(len(dfx), len(dfx) + future_periods)
    future_dates = pd.date_range(dfx["ds"].iloc[-1] + pd.Timedelta(days=1), periods=future_periods)
    future_values = slope * future_x + intercept

    daily_returns = dfx["y"].pct_change().dropna()
    daily_vol = daily_returns.std()
    days_out = np.arange(1, future_periods + 1)
    band_width = dfx["y"].iloc[-1] * daily_vol * np.sqrt(days_out) * 1.96  # ~95% band
    future_upper = future_values + band_width
    future_lower = future_values - band_width

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=dfx["ds"], y=dfx["y"], name="Historical Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=future_dates, y=future_values, name="Forecast (trend)", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=future_dates, y=future_upper, fill=None, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_dates, y=future_lower, fill="tonexty", mode="lines", line=dict(width=0), name="~95% band", fillcolor="rgba(0, 176, 0, 0.2)", hoverinfo="skip"))
    fig.update_layout(
        title=f"{company_name} ({ticker}) Stock Price Forecast (lightweight trend model)",
        xaxis_title="Date", yaxis_title=f"Price ({currency})",
        hovermode="x unified", legend=dict(x=0.01, y=0.99), template="plotly_white",
    )

    last_price = float(dfx["y"].iloc[-1])
    forecast_end_price = float(future_values[-1])
    price_change = forecast_end_price - last_price
    percent_change = (price_change / last_price) * 100
    confidence_width = float(future_upper[-1] - future_lower[-1])
    relative_uncertainty = (confidence_width / forecast_end_price) * 100

    return {
        "success": True,
        "ticker": ticker,
        "company_name": company_name,
        "currency": currency,
        "engine": "trend",
        "last_price": last_price,
        "forecast_end_price": forecast_end_price,
        "price_change": price_change,
        "percent_change": percent_change,
        "chart": json.loads(plotly.io.to_json(fig)),
        "explanation": {
            "summary": f"A simple linear trend projects a {'positive' if price_change > 0 else 'negative'} "
                       f"movement of {abs(price_change):.2f} {currency} ({percent_change:.2f}%) over the next year. "
                       f"This is a lightweight statistical projection, not a seasonality-aware model.",
            "trend": f"The underlying linear trend is {'upward' if price_change > 0 else 'downward'}.",
            "uncertainty": f"The ~95% band width is {confidence_width:.2f} {currency} "
                            f"({relative_uncertainty:.2f}% of the predicted price), based on historical daily volatility.",
            "seasonality": [],
        },
        "forecast_date_range": {
            "start": str(future_dates[0].date()),
            "end": str(future_dates[-1].date()),
        },
    }


def _prophet_forecast(history_df, company_name, ticker, currency):
    """Heavier, opt-in engine. Imports Prophet lazily (only when actually
    selected) and creates a FRESH Prophet() instance per call — Prophet
    models can only be fit once; reusing one instance across requests
    raises 'Prophet object can only be fit once. Instantiate a new object.'"""
    from prophet import Prophet  # lazy import — only pulled in if this engine is active

    dfx = pd.DataFrame()
    dfx["ds"] = pd.to_datetime(history_df.index).tz_localize(None)
    dfx["y"] = history_df["Close"].values

    model = Prophet(daily_seasonality=True)  # new instance every call, on purpose
    model.fit(dfx)

    future_periods = 365
    future_forecast = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future_forecast)

    historical_end_date = dfx["ds"].iloc[-1]
    future_dates = forecast["ds"][forecast["ds"] > historical_end_date]
    future_values = forecast["yhat"][forecast["ds"] > historical_end_date]
    future_upper = forecast["yhat_upper"][forecast["ds"] > historical_end_date]
    future_lower = forecast["yhat_lower"][forecast["ds"] > historical_end_date]

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=dfx["ds"], y=dfx["y"], name="Historical Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=future_dates, y=future_values, name="Forecast", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=future_dates, y=future_upper, fill=None, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_dates, y=future_lower, fill="tonexty", mode="lines", line=dict(width=0), name="95% Confidence", fillcolor="rgba(0, 176, 0, 0.2)", hoverinfo="skip"))
    fig.update_layout(
        title=f"{company_name} ({ticker}) Stock Price Forecast",
        xaxis_title="Date", yaxis_title=f"Price ({currency})",
        hovermode="x unified", legend=dict(x=0.01, y=0.99), template="plotly_white",
    )

    last_price = float(dfx["y"].iloc[-1])
    forecast_end_price = float(future_values.iloc[-1])
    price_change = forecast_end_price - last_price
    percent_change = (price_change / last_price) * 100
    confidence_width = float(future_upper.iloc[-1] - future_lower.iloc[-1])
    relative_uncertainty = (confidence_width / forecast_end_price) * 100

    components = forecast[["trend", "yearly", "weekly", "daily"]]
    seasonality_notes = []
    if "yearly" in components.columns:
        seasonality_notes.append("Yearly seasonality detected in the price pattern.")
    if "weekly" in components.columns:
        seasonality_notes.append("Weekly patterns are present with price variations across different days of the week.")
    if "daily" in components.columns:
        seasonality_notes.append("Daily price fluctuations show consistent patterns within trading days.")

    return {
        "success": True,
        "ticker": ticker,
        "company_name": company_name,
        "currency": currency,
        "engine": "prophet",
        "last_price": last_price,
        "forecast_end_price": forecast_end_price,
        "price_change": price_change,
        "percent_change": percent_change,
        "chart": json.loads(plotly.io.to_json(fig)),
        "explanation": {
            "summary": f"The forecast predicts a {'positive' if price_change > 0 else 'negative'} price movement "
                       f"of {abs(price_change):.2f} {currency} ({percent_change:.2f}%) over the next year.",
            "trend": f"The underlying trend is {'upward' if price_change > 0 else 'downward'}.",
            "uncertainty": f"The forecast has a 95% confidence interval width of {confidence_width:.2f} {currency} "
                            f"({relative_uncertainty:.2f}% of predicted price).",
            "seasonality": seasonality_notes,
        },
        "forecast_date_range": {
            "start": str(future_dates.iloc[0].date()),
            "end": str(future_dates.iloc[-1].date()),
        },
    }


def forecast_stock(stock, history_df, ticker):
    try:
        stock_info = stock.info
        company_name = stock_info.get("shortName", ticker)
        currency = stock_info.get("currency", "USD")

        engine = os.environ.get("FORECAST_ENGINE", "trend").lower()
        if engine == "prophet":
            return _prophet_forecast(history_df, company_name, ticker, currency)
        return _lightweight_trend_forecast(history_df, company_name, ticker, currency)
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# =====================================================================
# Candlestick + company info
# =====================================================================
def analyze_candlestick_and_info(stock, history_df, ticker):
    try:
        stock_info = stock.info
        company_name = stock_info.get("shortName", ticker)
        currency = stock_info.get("currency", "USD")
        historical_data = history_df

        fig = go.Figure(data=[go.Candlestick(
            x=historical_data.index,
            open=historical_data["Open"], high=historical_data["High"],
            low=historical_data["Low"], close=historical_data["Close"],
            name="Candlestick",
        )])
        fig.update_layout(
            title=f"{company_name} ({ticker}) Candlestick Chart (last {HISTORY_PERIOD})",
            xaxis_title="Date", yaxis_title=f"Price ({currency})",
            xaxis_rangeslider_visible=False, template="plotly_white",
        )

        filtered_info = {
            "Country": stock_info.get("country", "N/A"),
            "Website": stock_info.get("website", "N/A"),
            "Industry": stock_info.get("industry", "N/A"),
            "Business Summary": stock_info.get("longBusinessSummary", "N/A"),
            "Recommendation": stock_info.get("recommendationKey", "N/A").upper(),
        }

        latest_price = historical_data.iloc[-1].Close if not historical_data.empty else None

        if not historical_data.empty and len(historical_data) > 30:
            short_term_return = ((historical_data.iloc[-1].Close / historical_data.iloc[-30].Close) - 1) * 100
            medium_term_idx = -180 if len(historical_data) >= 180 else 0
            medium_term_return = ((historical_data.iloc[-1].Close / historical_data.iloc[medium_term_idx].Close) - 1) * 100
            long_term_idx = -252 if len(historical_data) >= 252 else 0
            long_term_return = ((historical_data.iloc[-1].Close / historical_data.iloc[long_term_idx].Close) - 1) * 100

            recent_returns = historical_data.iloc[-30:].Close.pct_change().dropna()
            volatility = recent_returns.std() * 100

            recent_closes = historical_data.iloc[-10:].Close
            price_patterns = []
            if all(recent_closes.iloc[i] <= recent_closes.iloc[i + 1] for i in range(len(recent_closes) - 3, len(recent_closes) - 1)):
                price_patterns.append("Recent short-term uptrend detected in the last few days")
            if all(recent_closes.iloc[i] >= recent_closes.iloc[i + 1] for i in range(len(recent_closes) - 3, len(recent_closes) - 1)):
                price_patterns.append("Recent short-term downtrend detected in the last few days")
            if volatility > 3:
                price_patterns.append(f"High price volatility detected (daily std dev: {volatility:.2f}%)")

            year_high = historical_data.iloc[-252:].High.max() if len(historical_data) >= 252 else historical_data.High.max()
            year_low = historical_data.iloc[-252:].Low.min() if len(historical_data) >= 252 else historical_data.Low.min()
            if latest_price > 0.95 * year_high:
                price_patterns.append("Price is near 52-week high, which may indicate strong buying momentum")
            if latest_price < 1.05 * year_low:
                price_patterns.append("Price is near 52-week low, which may indicate potential value or ongoing concerns")

            recent_volume = historical_data.iloc[-30:].Volume
            avg_volume = recent_volume.mean()
            latest_volume = recent_volume.iloc[-1]
            if latest_volume > 1.5 * avg_volume:
                volume_analysis = f"Trading volume is significantly above average ({latest_volume/avg_volume:.1f}x), indicating strong market interest"
            elif latest_volume < 0.5 * avg_volume:
                volume_analysis = f"Trading volume is significantly below average ({latest_volume/avg_volume:.1f}x), indicating lower market interest"
            else:
                volume_analysis = "Trading volume is in line with recent average"
        else:
            short_term_return = medium_term_return = long_term_return = volatility = None
            price_patterns = ["Insufficient historical data for pattern detection"]
            volume_analysis = "Insufficient data for volume analysis"

        chart_explanations = {
            "short_term": f"In the last 30 days, the stock price has {'increased' if short_term_return and short_term_return > 0 else 'decreased'} by {abs(short_term_return):.2f}%" if short_term_return is not None else "Short-term trend data unavailable",
            "medium_term": f"Over the last ~6 months, the stock price has {'increased' if medium_term_return and medium_term_return > 0 else 'decreased'} by {abs(medium_term_return):.2f}%" if medium_term_return is not None else "Medium-term trend data unavailable",
            "long_term": f"Over the last year, the stock price has {'increased' if long_term_return and long_term_return > 0 else 'decreased'} by {abs(long_term_return):.2f}%" if long_term_return is not None else "Long-term trend data unavailable",
            "volatility": f"The stock's recent volatility is {volatility:.2f}%, which is {'high' if volatility > 3 else 'moderate' if volatility > 1.5 else 'low'}" if volatility is not None else "Volatility data unavailable",
            "patterns": price_patterns,
            "volume": volume_analysis,
        }

        candlestick_explanation = (
            "Candlestick charts provide rich visual information about price movements:\n\n"
            "1. Each candle represents a time period (usually a day)\n"
            "2. The body (colored part) shows opening and closing prices\n"
            "3. If the candle is green/white, the price closed higher than it opened (bullish)\n"
            "4. If the candle is red/black, the price closed lower than it opened (bearish)\n"
            "5. The wicks/shadows (lines extending from the body) show the high and low prices"
        )

        if filtered_info["Business Summary"] != "N/A":
            filtered_info["Business Summary"] = filtered_info["Business Summary"].replace("\n", " ")

        recommendation = filtered_info["Recommendation"]
        if recommendation in ("BUY", "STRONG_BUY"):
            recommendation_explanation = "Analysts generally expect the stock to outperform the market and recommend purchasing shares."
        elif recommendation in ("SELL", "STRONG_SELL"):
            recommendation_explanation = "Analysts generally expect the stock to underperform the market and recommend selling shares."
        elif recommendation in ("HOLD", "NEUTRAL"):
            recommendation_explanation = "Analysts recommend maintaining current positions without buying or selling additional shares."
        else:
            recommendation_explanation = "No clear analyst consensus is available for this stock."

        return {
            "success": True,
            "ticker": ticker,
            "company_name": company_name,
            "currency": currency,
            "current_price": float(latest_price) if latest_price is not None else None,
            "company_info": filtered_info,
            "recommendation_explanation": recommendation_explanation,
            "chart": json.loads(plotly.io.to_json(fig)),
            "chart_analysis": chart_explanations,
            "candlestick_guide": candlestick_explanation,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# =====================================================================
# Sector performance (Financial Modeling Prep)
# =====================================================================
def analyze_sector_performance(stock, ticker):
    try:
        stock_info = stock.info
        sector = stock_info.get("sector", "N/A")
        industry = stock_info.get("industry", "N/A")
        company_name = stock_info.get("shortName", ticker)

        if sector == "N/A":
            return {"success": False, "error": f"Sector information not available for {ticker}"}

        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            return {"success": False, "error": "FMP_API_KEY is not configured on the server"}

        # Corrected endpoint — FMP's current docs show this path (singular
        # "sector-performance", no "/stock/" prefix); the original app's
        # URL used an older/incorrect path.
        url = f"https://financialmodelingprep.com/api/v3/sector-performance?apikey={api_key}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            sector_data = response.json()

        except requests.exceptions.HTTPError as e:
            print(f"FMP HTTP error: {response.status_code}")
            return {
                "success": False,
                "error": f"FMP API request failed with status {response.status_code}."
            }

        except requests.exceptions.RequestException as e:
            print(f"FMP request error: {type(e).__name__}")
            return {
                "success": False,
                "error": "Failed to connect to the sector performance service."
            }

        # Defensive parsing: handle either a flat list response or a dict
        # wrapping the list under "sectorPerformance" — FMP's exact
        # response shape has shifted across API versions.
        if isinstance(sector_data, list):
            raw_entries = sector_data
        elif isinstance(sector_data, dict):
            raw_entries = sector_data.get("sectorPerformance", [])
        else:
            raw_entries = []

        sector_performance = None
        sector_performances = []
        for entry in raw_entries:
            try:
                sector_name = entry.get("sector")
                raw_change = entry.get("changesPercentage", entry.get("changePercentage"))
                if raw_change is None:
                    continue
                change_percentage = float(str(raw_change).replace("%", ""))
                sector_performances.append({"sector": sector_name, "change_percentage": change_percentage})
                if sector_name == sector:
                    sector_performance = {"sector": sector_name, "change_percentage": change_percentage}
            except (KeyError, ValueError, AttributeError):
                continue

        if not sector_performance:
            return {"success": False, "error": f"No performance data found for the {sector} sector."}

        explanation = f"The {sector} sector, which {company_name} operates in, has "
        if sector_performance["change_percentage"] > 0:
            explanation += f"increased by {sector_performance['change_percentage']:.2f}% recently. "
        elif sector_performance["change_percentage"] < 0:
            explanation += f"decreased by {abs(sector_performance['change_percentage']):.2f}% recently. "
        else:
            explanation += "shown no significant change recently. "

        if sector_performances:
            sorted_sectors = sorted(sector_performances, key=lambda x: x["change_percentage"], reverse=True)
            top_sector = sorted_sectors[0]
            bottom_sector = sorted_sectors[-1]
            explanation += (
                f"The best performing sector is {top_sector['sector']} with a "
                f"{top_sector['change_percentage']:.2f}% change, while the worst performing sector is "
                f"{bottom_sector['sector']} with a {bottom_sector['change_percentage']:.2f}% change."
            )
        else:
            top_sector = bottom_sector = None

        return {
            "success": True,
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "sector_performance": sector_performance,
            "sector_comparison": {"top_sector": top_sector, "bottom_sector": bottom_sector},
            "explanation": explanation,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# =====================================================================
# ROUTE
# =====================================================================
class StockAnalyzeRequest(BaseModel):
    ticker: str


@router.post("/analyze")
def analyze(payload: StockAnalyzeRequest):
    ticker = payload.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="No ticker provided")

    try:
        # Fetch the ticker and its price history ONCE, share across all
        # four analyses below — the original app fetched these separately
        # in each function (4x the network calls and redundant memory for
        # the same data). This is the main real RAM/latency win available
        # here beyond the forecast-engine toggle.
        stock = yf.Ticker(ticker)
        history_df = stock.history(period=HISTORY_PERIOD)

        if history_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")

        piotroski_results = analyze_piotroski(stock, ticker)
        forecast_results = forecast_stock(stock, history_df, ticker)
        candlestick_results = analyze_candlestick_and_info(stock, history_df, ticker)
        sector_results = analyze_sector_performance(stock, ticker)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "piotroski": piotroski_results,
        "prophet": forecast_results,  # key name kept as "prophet" for frontend compatibility, regardless of engine used
        "candlestick": candlestick_results,
        "sector": sector_results,
    }
