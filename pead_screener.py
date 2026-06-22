"""
PEAD Screener: Surprise + Recency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One email per day, 7 AM ET on weekdays. Two sections, zero overlap:

  BUY TODAY       -- yesterday's reporters scored on SUE + 52wk-high recency.
                     These are the trades you'd enter at today's open.
  REPORTING TODAY -- stocks reporting today, pre-enriched with recency ratios.
                     These become tomorrow's BUY TODAY candidates.

Each day's email is self-contained. No 3-day window, no repeated tickers.

Academic basis:
  - SUE + recency captures 74% of PEAD profits (Alpha Architect)
  - Drift strongest when surprise is cash-flow / revenue driven
  - Survives in small/mid cap + low analyst coverage names

Dependencies:  pip install requests yfinance pandas

Usage:
  python pead_screener.py              # prints report to stdout
  python pead_screener.py --email      # sends email (requires env vars)
"""

import os
import sys
import json
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ── CONFIG ──────────────────────────────────────────────────────────

FINNHUB_API_KEY = os.environ.get(
    "FINNHUB_API_KEY", "d85n75pr01qitd92s31gd85n75pr01qitd92s320"
)

EMAIL_FROM     = os.environ.get("EMAIL_FROM", "")
EMAIL_TO       = os.environ.get("EMAIL_TO", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
SMTP_SERVER    = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT      = int(os.environ.get("SMTP_PORT", "587"))

# Signal thresholds -- tune these
RECENCY_THRESHOLD = 0.85     # price / 52wk high (0.85 = within 15%)
SUE_THRESHOLD     = 0.10     # |SUE| minimum (10% beat or miss)
MIN_MARKET_CAP    = 50_000_000   # $50M floor to avoid untradeable microcaps

NASDAQ_HEADERS = {
    "authority": "api.nasdaq.com",
    "accept": "application/json, text/plain, */*",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "origin": "https://www.nasdaq.com",
    "referer": "https://www.nasdaq.com/",
}


# ── DATA FETCHERS ───────────────────────────────────────────────────

def fetch_nasdaq_earnings(date_str: str) -> list[dict]:
    """Earnings calendar from Nasdaq public API. No key needed."""
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={date_str}"
    try:
        resp = requests.get(url, headers=NASDAQ_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        return data.get("rows", []) or []
    except Exception as e:
        print(f"  [nasdaq] {date_str} error: {e}")
        return []


def fetch_finnhub_earnings(from_date: str, to_date: str) -> list[dict]:
    """Earnings actuals from Finnhub free tier."""
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {"from": from_date, "to": to_date, "token": FINNHUB_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("earningsCalendar", [])
    except Exception as e:
        print(f"  [finnhub] error: {e}")
        return []


def enrich_with_yfinance(symbols: list[str]) -> dict:
    """
    Batch-fetch recency ratio + metadata from yfinance.
    Rate-limited with small sleep to avoid throttling.
    """
    enriched = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        try:
            tk = yf.Ticker(sym)
            info = tk.info or {}

            high_52 = info.get("fiftyTwoWeekHigh")
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            recency = None
            if high_52 and price and high_52 > 0:
                recency = round(price / high_52, 4)

            enriched[sym] = {
                "price": price,
                "52wk_high": high_52,
                "recency": recency,
                "avg_volume": info.get("averageVolume"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector"),
                "short_pct": info.get("shortPercentOfFloat"),
            }
        except Exception as e:
            print(f"  [yfinance] {sym}: {e}")
            enriched[sym] = {}

        # Progress + rate limit
        if (i + 1) % 20 == 0:
            print(f"  enriched {i + 1}/{total}")
            time.sleep(0.5)

    return enriched


# ── SIGNAL LOGIC ────────────────────────────────────────────────────

def compute_sue(actual: float, estimate: float) -> float | None:
    """
    Standardized Unexpected Earnings.
    (actual - estimate) / |estimate|
    Falls back to raw diff if estimate ~ 0.
    """
    if actual is None or estimate is None:
        return None
    if abs(estimate) < 0.01:
        return actual - estimate
    return (actual - estimate) / abs(estimate)


def score_signals(finnhub_earnings: list[dict], yf_data: dict) -> list[dict]:
    """Score each reported stock on SUE + recency."""
    signals = []
    for e in finnhub_earnings:
        sym = e.get("symbol", "")
        actual = e.get("epsActual")
        estimate = e.get("epsEstimate")

        if actual is None or estimate is None:
            continue

        sue = compute_sue(actual, estimate)
        if sue is None:
            continue

        yf_info = yf_data.get(sym, {})
        recency = yf_info.get("recency")
        mkt_cap = yf_info.get("market_cap")

        # Revenue surprise (secondary confirmation signal)
        rev_actual = e.get("revenueActual")
        rev_est = e.get("revenueEstimate")
        rev_surprise = None
        if rev_actual and rev_est and rev_est > 0:
            rev_surprise = round((rev_actual - rev_est) / rev_est, 4)

        signals.append({
            "symbol": sym,
            "eps_actual": actual,
            "eps_estimate": estimate,
            "sue": round(sue, 4),
            "rev_surprise": rev_surprise,
            "recency": recency,
            "price": yf_info.get("price"),
            "52wk_high": yf_info.get("52wk_high"),
            "market_cap": mkt_cap,
            "avg_volume": yf_info.get("avg_volume"),
            "sector": yf_info.get("sector"),
            "short_pct": yf_info.get("short_pct"),
        })

    return signals


def filter_signals(signals: list[dict]) -> list[dict]:
    """
    Keep stocks where:
      - |SUE| > threshold  (meaningful surprise)
      - recency > threshold (near 52wk high -- momentum confirmation)
      - market cap > floor  (tradeable)
    """
    passed = []
    for s in signals:
        sue = s.get("sue")
        recency = s.get("recency")
        mkt_cap = s.get("market_cap")

        if sue is None or abs(sue) < SUE_THRESHOLD:
            continue
        if recency is not None and recency < RECENCY_THRESHOLD:
            continue
        if mkt_cap is not None and mkt_cap < MIN_MARKET_CAP:
            continue

        passed.append(s)

    # Strongest beats first
    passed.sort(key=lambda x: abs(x["sue"]), reverse=True)
    return passed


# ── OUTPUT ──────────────────────────────────────────────────────────

def fmt_cap(mc):
    if not mc:
        return "N/A"
    if mc >= 1e9:
        return f"${mc / 1e9:.1f}B"
    return f"${mc / 1e6:.0f}M"


def format_report(
    signals: list[dict],
    reporting_today: list[dict],
    buy_date: str,
    report_date: str,
) -> str:
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"PEAD SCREENER -- {now}")
    lines.append("=" * 70)

    # ── BUY TODAY ──
    lines.append(
        f"\nBUY TODAY  (reported {buy_date}, entering drift window)"
        f"\n  Filters: SUE > {SUE_THRESHOLD}  |  recency > {RECENCY_THRESHOLD}"
    )
    lines.append("-" * 70)

    if not signals:
        lines.append("  No signals today.")
    else:
        for s in signals:
            tag = "BEAT" if s["sue"] > 0 else "MISS"
            rec = f"{s['recency']:.2f}" if s["recency"] else "N/A"
            rev = f"{s['rev_surprise']:+.1%}" if s["rev_surprise"] is not None else "N/A"
            cap = fmt_cap(s["market_cap"])

            lines.append(
                f"  {s['symbol']:8s} {tag}  "
                f"SUE={s['sue']:+.2f}  "
                f"EPS={s['eps_actual']} vs {s['eps_estimate']}  "
                f"Rev={rev}  Recency={rec}  Cap={cap}"
            )
            extras = []
            if s.get("sector"):
                extras.append(f"Sector: {s['sector']}")
            if s.get("avg_volume"):
                extras.append(f"AvgVol: {s['avg_volume']:,}")
            if s.get("short_pct"):
                extras.append(f"Short: {s['short_pct']:.1%}")
            if extras:
                lines.append(f"           {'  |  '.join(extras)}")

    # ── REPORTING TODAY ──
    lines.append(
        f"\nREPORTING TODAY  ({report_date})"
        f"\n  Sorted by recency -- tomorrow's potential signals"
    )
    lines.append("-" * 70)

    if not reporting_today:
        lines.append("  No earnings scheduled today.")
    else:
        for u in reporting_today:
            rec = f"{u['recency']:.2f}" if u.get("recency") else " N/A"
            cap = fmt_cap(u.get("market_cap"))
            name = u.get("name", "")[:28]
            timing = u.get("time", "")
            # Clean up Nasdaq's timing labels
            if "pre-market" in timing.lower():
                timing = "pre"
            elif "after" in timing.lower():
                timing = "post"
            else:
                timing = "tbd"
            lines.append(
                f"  {u['symbol']:8s} {name:28s}  "
                f"Est={u.get('epsForecast', 'N/A'):>8s}  "
                f"Recency={rec}  {cap:>8s}  [{timing}]"
            )

    # ── FOOTER ──
    lines.append(f"\n{'=' * 70}")
    lines.append(
        f"Config: SUE>{SUE_THRESHOLD}  Recency>{RECENCY_THRESHOLD}  "
        f"MinCap>{fmt_cap(MIN_MARKET_CAP)}"
    )
    lines.append("Sources: Nasdaq (calendar) + Finnhub (actuals) + yfinance (price)")

    return "\n".join(lines)


def send_email(subject: str, body: str):
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD]):
        print("\n[email not configured -- printing report]\n")
        print(body)
        return

    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"Email sent to {EMAIL_TO}")
    except Exception as e:
        print(f"Email error: {e}")
        print(body)


# ── MAIN ────────────────────────────────────────────────────────────

def main():
    today = datetime.today()
    today_str = today.strftime("%Y-%m-%d")
    send = "--email" in sys.argv

    # ── Weekly lookback: Mon–Thu of the current week ──
    # Friday run: today is Friday (weekday=4), so look back 4 days
    lookback_days = 4
    week_start = today - timedelta(days=lookback_days)
    week_start_str = week_start.strftime("%Y-%m-%d")
    yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # ── 1. BUY TODAY: score the whole week's reporters ──
    print(f"=== WEEKLY BUY LIST (reported {week_start_str} to {yesterday_str}) ===")
    finnhub_data = fetch_finnhub_earnings(week_start_str, yesterday_str)
    print(f"  Finnhub: {len(finnhub_data)} records")

    scan_symbols = list({e["symbol"] for e in finnhub_data if e.get("symbol")})
    print(f"  Enriching {len(scan_symbols)} symbols via yfinance...")
    yf_scan = enrich_with_yfinance(scan_symbols)

    all_signals = score_signals(finnhub_data, yf_scan)
    signals = filter_signals(all_signals)
    print(f"  {len(all_signals)} scored -> {len(signals)} passed filters")

    # ── 2. REPORTING TODAY: Friday's earnings calendar ──
    print(f"\n=== REPORTING TODAY ({today_str}) ===")
    today_earnings = fetch_nasdaq_earnings(today_str)
    print(f"  Nasdaq: {len(today_earnings)} reports")

    today_symbols = list({r["symbol"] for r in today_earnings if r.get("symbol")})
    print(f"  Enriching {len(today_symbols)} symbols via yfinance...")
    yf_today = enrich_with_yfinance(today_symbols)

    reporting_today = []
    for r in today_earnings:
        sym = r.get("symbol", "")
        info = yf_today.get(sym, {})
        reporting_today.append({
            **r,
            "recency": info.get("recency"),
            "market_cap": info.get("market_cap"),
        })

    reporting_today.sort(key=lambda x: x.get("recency") or 0, reverse=True)

    # ── 3. Report ──
    report = format_report(signals, reporting_today, f"{week_start_str} to {yesterday_str}", today_str)
    subject = (
        f"PEAD Weekly {today_str} "
        f"-- {len(signals)} signal{'s' if len(signals) != 1 else ''}, "
        f"{len(reporting_today)} reporting today"
    )

    if send:
        send_email(subject, report)
    else:
        print(f"\n{report}")

if __name__ == "__main__":
    main()
