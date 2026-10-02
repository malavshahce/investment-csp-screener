"""Scanning engine for the cash-secured put screener: data fetching, option math,
per-ticker scan, market regime and the capital-aware trade-plan builder."""
import io
import json
import logging
import time
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ---------------- Constants ----------------

LEVERAGE = {
    "SOXL": 3, "SOXS": 3, "TQQQ": 3, "SQQQ": 3, "SPXL": 3,
    "SPXS": 3, "TNA": 3, "TZA": 3, "LABU": 3, "LABD": 3,
    "QLD": 2, "SSO": 2, "UVXY": 2,
}

MIN_SANE_IV = 0.05   # Yahoo IV below this is almost always a bad quote
MAX_SANE_IV = 5.0
MAX_WORKERS = 6      # keep modest to avoid Yahoo rate limiting
MAX_TICKERS = 4000   # safety cap per scan

# Label shown in the UI -> column the candidates are ranked by
RANK_OPTIONS = {
    "Score": "Score",
    "Edge $ (premium vs fair value)": "Edge $",
    "Win probability": "Win % (cons.)",
    "Return per day": "Return/Day %",
    "Premium $": "Premium $",
}

# ---------------- Ticker universes ----------------

LIQUID_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "NFLX",
    "COST", "JPM", "BAC", "WFC", "C", "GS", "MS", "XOM", "CVX", "PFE", "MRK", "JNJ",
    "KO", "PEP", "WMT", "HD", "DIS", "BA", "INTC", "MU", "QCOM", "CRM", "ORCL", "ADBE",
    "UBER", "PLTR", "COIN", "SOFI", "F", "GM", "T", "VZ", "CSCO", "IBM", "PYPL", "SHOP",
    "SNOW", "ABNB", "NKE", "SBUX", "MCD", "LLY", "UNH", "V", "MA", "CAT", "DE", "LMT",
    "RTX", "HOOD", "RIVN", "NIO", "BABA", "SMCI", "ARM", "TSM", "ASML", "BE", "MARA",
    "RIOT", "AAL", "DAL", "CCL", "NCLH", "LCID", "SNAP", "PINS", "ROKU", "DKNG",
]
INDEX_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU",
    "GLD", "SLV", "TLT", "HYG", "EEM", "EFA", "SMH", "ARKK", "KRE", "XBI",
]
LEVERAGED_ETFS = list(LEVERAGE)

UNIVERSE_FALLBACK = {"S&P 500": LIQUID_STOCKS, "Nasdaq-100": LIQUID_STOCKS}


def _wiki_tables(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=20).read().decode()
    return pd.read_html(io.StringIO(html))


@st.cache_data(ttl=86400)
def load_index_universe(name):
    """S&P 500 / Nasdaq-100 constituents from Wikipedia; falls back to a curated list."""
    try:
        if name == "S&P 500":
            cols, tables = ["Symbol"], _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        else:
            cols, tables = ["Ticker", "Symbol"], _wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            for c in cols:
                if c in t.columns and len(t) > 50:
                    return [str(x).strip().upper().replace(".", "-") for x in t[c].dropna()]
    except Exception:
        pass
    return UNIVERSE_FALLBACK[name]


UNIVERSES = {
    "Most-liquid stocks (~80)": lambda: LIQUID_STOCKS,
    "Index & sector ETFs": lambda: INDEX_ETFS,
    "Leveraged ETFs": lambda: LEVERAGED_ETFS,
    "Nasdaq-100": lambda: load_index_universe("Nasdaq-100"),
    "S&P 500 (slow)": lambda: load_index_universe("S&P 500"),
}


# ---------------- Whole-market universe ----------------

MARKET_UNIVERSE = "Entire US options market (pre-filtered)"
_WEB_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}


def _fetch(url, timeout=40):
    req = urllib.request.Request(url, headers=_WEB_HEADERS)
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def _norm_symbol(sym):
    return str(sym).strip().upper().replace("/", "-").replace(".", "-")


@st.cache_data(ttl=21600, show_spinner=False)
def load_optionable_symbols():
    """Every symbol with listed options, from Cboe's official symbol directory."""
    try:
        df = pd.read_csv(io.StringIO(_fetch("https://www.cboe.com/us/options/symboldir/equity_index_options/?download=csv")))
        df.columns = [c.strip() for c in df.columns]
        return sorted({_norm_symbol(x) for x in df["Stock Symbol"].dropna()})
    except Exception:
        return []


@st.cache_data(ttl=21600, show_spinner=False)
def load_stock_table():
    """Last price and volume for all US-listed stocks (Nasdaq screener) in a single request."""
    try:
        rows = json.loads(_fetch("https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"))["data"]["rows"]
        df = pd.DataFrame(rows)
        return pd.DataFrame({
            "sym": df["symbol"].map(_norm_symbol),
            "price": pd.to_numeric(df["lastsale"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"),
            "volume": pd.to_numeric(df["volume"].astype(str).str.replace(",", ""), errors="coerce"),
        }).dropna(subset=["price"])
    except Exception:
        return pd.DataFrame(columns=["sym", "price", "volume"])


def build_market_universe(max_contract_cost, min_price, min_volume, max_n):
    """Pre-filter the whole optionable market without touching Yahoo: has options, price in range,
    affordable for your cash, and enough share volume. Returns (most-liquid-first tickers, info dict)."""
    opt = set(load_optionable_symbols())
    tbl = load_stock_table()
    info = {"optionable": len(opt), "priced": 0, "passed": 0, "scanning": 0, "error": None}
    if not opt or tbl.empty:
        info["error"] = "Could not download the market lists (Cboe / Nasdaq); using your other universes only."
        return [], info
    m = tbl[tbl["sym"].isin(opt)]
    info["priced"] = len(m)
    # A put within ~20% of spot costs ~0.8 x spot x 100, so skip stocks whose such strikes are unaffordable.
    f = m[(m["price"] >= min_price) & (m["price"] * 80 <= max_contract_cost) & (m["volume"].fillna(0) >= min_volume)]
    info["passed"] = len(f)
    f = f.assign(dollar_vol=f["price"] * f["volume"].fillna(0)).sort_values("dollar_vol", ascending=False).head(max_n)
    info["scanning"] = len(f)
    return f["sym"].tolist(), info


# ---------------- Data helpers ----------------


def with_retry(fn, attempts=3, delay=1.0):
    """Call fn(), retrying on failure with a growing delay. Re-raises the last error."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


@st.cache_data(ttl=3600)
def get_risk_free_rate():
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]
        return irx / 100
    except Exception:
        return 0.045


def get_dividend_yield(tk):
    try:
        dy = tk.info.get("dividendYield", 0) or 0
        if dy > 1:
            dy = dy / 100
        return min(max(dy, 0), 0.15)
    except Exception:
        return 0.0


def historical_volatility(close_prices, lookback_days=30):
    if len(close_prices) < lookback_days + 1:
        return np.nan
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    return log_returns[-lookback_days:].std() * np.sqrt(252)


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    val = (100 - 100 / (1 + rs)).iloc[-1]
    return float(val) if pd.notna(val) else np.nan


def trend_label(spot, sma50, sma200):
    if pd.isna(sma200):
        return "Mixed"
    if spot > sma50 > sma200:
        return "Uptrend"
    if spot < sma50 < sma200:
        return "Downtrend"
    return "Mixed"


def ny_now():
    """Current time in New York (options expire on New York dates, whatever your local clock says)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def ny_today():
    return ny_now().date()


def intraday_years():
    """Time left until today's 4:00 PM New York close, in years (floored at 10 minutes), for same-day options."""
    now = ny_now()
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    minutes = max((close - now).total_seconds() / 60, 10)
    return minutes / (365 * 24 * 60)


def days_to_expiry(expiry_str):
    exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    return (exp_date - ny_today()).days


def _to_date(d):
    if isinstance(d, datetime):
        return d.date()
    return d if isinstance(d, date) else None


def get_next_earnings_date(tk):
    try:
        cal = tk.calendar
        earnings_dates = cal.get("Earnings Date") if cal else None
        if earnings_dates:
            future = [d for d in map(_to_date, earnings_dates) if d and d >= date.today()]
            return min(future) if future else None
    except Exception:
        pass
    try:
        ed = tk.get_earnings_dates(limit=8)
        if ed is not None and not ed.empty:
            idx_dates = [d.date() if hasattr(d, "date") else d for d in ed.index]
            future = [d for d in idx_dates if d >= date.today()]
            if future:
                return min(future)
    except Exception:
        pass
    return None


def get_ex_div_date(tk):
    """Next ex-dividend date: from the calendar, else from the company info (epoch seconds). None if past/unknown."""
    try:
        cal = tk.calendar
        d = _to_date(cal.get("Ex-Dividend Date")) if cal else None
        if d and d >= date.today():
            return d
    except Exception:
        pass
    try:
        ts = tk.info.get("exDividendDate")
        if ts:
            d = datetime.fromtimestamp(ts).date()
            if d >= date.today():
                return d
    except Exception:
        pass
    return None


# ---------------- Option math ----------------


def _d1_d2(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_put_metrics(S, K, T, r, q, sigma):
    if T <= 0 or sigma is None or sigma <= 0 or np.isnan(sigma):
        return np.nan, np.nan
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    return delta, norm.cdf(d2)


def bs_put_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma is None or sigma <= 0 or np.isnan(sigma):
        return np.nan
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol_put(price, S, K, T, r, q):
    """Back out the implied volatility that makes the Black-Scholes put price equal `price`. NaN if impossible."""
    if price <= 0 or T <= 0:
        return np.nan
    intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    if price <= intrinsic + 1e-6:
        return np.nan
    try:
        return brentq(lambda sig: bs_put_price(S, K, T, r, q, sig) - price, 0.01, 5.0, xtol=1e-6)
    except Exception:
        return np.nan


def put_pnl_at_drop(spot, K, premium, drop_pct):
    """Per-share P&L at expiry if the stock finishes drop_pct% below today's price."""
    final = spot * (1 - drop_pct / 100)
    return premium - max(K - final, 0)


# ---------------- Scan ----------------


@dataclass
class ScanParams:
    risk_free_rate: float
    min_dte: int
    max_dte: int
    win_prob_min: float
    win_prob_max: float
    min_oi: int
    max_spread_pct: float
    top_n: int
    min_otm_pct: float = 0.0
    min_hist_win: float = 0.0
    min_premium_usd: float = 0.0
    max_contract_cost: float = float("inf")
    exclude_earnings: bool = False
    skip_downtrend: bool = False
    below_sma: bool = False
    require_edge: bool = False
    allow_stale: bool = False   # use last-trade prices when there is no live bid/ask (market closed)
    show_all: bool = False      # never skip a put for liquidity / probability / volatility / spread / premium
    ext_prices: object = None   # {symbol: (price, session, pct)}: pre-market / after-hours stock prices to use as "spot"
    rank_by: str = "Score"


def relax_params(p):
    """Looser liquidity / premium / probability filters, used when the strict scan finds (almost) nothing.
    Risk filters (earnings, downtrend, affordability) are never relaxed."""
    from dataclasses import replace
    return replace(
        p,
        min_premium_usd=min(p.min_premium_usd, 5),
        min_oi=min(p.min_oi, 100),
        max_spread_pct=max(p.max_spread_pct, 25),
        win_prob_min=max(50, p.win_prob_min - 7),
        win_prob_max=max(p.win_prob_max, 99),
        min_hist_win=min(p.min_hist_win, 60),
    )


def widen_params(p):
    """Last-resort search: relaxed liquidity/probability AND earnings/downtrend/history filters off, longer window.
    Used only when the stricter passes found almost nothing; rows are labeled so the extra risk is visible."""
    from dataclasses import replace
    r = relax_params(p)
    return replace(r, exclude_earnings=False, skip_downtrend=False, below_sma=False, require_edge=False,
                   min_hist_win=0, min_otm_pct=0.0, min_premium_usd=min(p.min_premium_usd, 5),
                   win_prob_min=max(50, p.win_prob_min - 12), min_dte=max(1, p.min_dte - 2), max_dte=p.max_dte + 10)


RELAXED_NOTE = ("min premium $5, min open interest 100, max spread 25%, win probability 7 points lower, "
                "min historical win 60%")


def scan_ticker_for_csp(ticker_symbol, p):
    """Scan one ticker. Always fetches live data (no caching). Safe to run in a
    worker thread: it does not touch Streamlit."""
    tk = yf.Ticker(ticker_symbol)
    counts = {"puts_in_window": 0, "otm_priced": 0, "affordable": 0, "premium_ok": 0, "liquidity": 0,
              "spread": 0, "sane_iv": 0, "win_band": 0, "hist_ok": 0, "downtrend_skip": 0,
              "too_expensive": 0}

    try:
        hist = with_retry(lambda: tk.history(period="3y"))
        closes = hist["Close"].dropna()
        spot = float(closes.iloc[-1])
    except Exception:
        return pd.DataFrame(), None, counts

    # Pre-market / after-hours: use the stock's latest extended-hours price as spot (options themselves don't trade there)
    prev_close, ext_session, ext_pct = spot, "", np.nan
    ext = (p.ext_prices or {}).get(ticker_symbol)
    if ext:
        spot, ext_session, ext_pct = float(ext[0]), ext[1], float(ext[2])

    hv = historical_volatility(closes, lookback_days=30)
    close_arr = closes.to_numpy(dtype=float)
    low_52w = closes.tail(252).min()
    sma_50 = closes.tail(50).mean()
    sma_200 = closes.tail(200).mean() if len(closes) >= 200 else np.nan
    rsi14 = compute_rsi(closes)
    trend = trend_label(spot, sma_50, sma_200)
    leverage_multiplier = LEVERAGE.get(ticker_symbol, 1)

    # Puts within ~20% of spot cost ~0.8 x spot x 100 of cash; if even those are unaffordable, don't fetch chains.
    if spot * 80 > p.max_contract_cost:
        counts["too_expensive"] = 1
        return pd.DataFrame(), spot, counts

    if p.skip_downtrend and trend == "Downtrend":
        counts["downtrend_skip"] = 1
        return pd.DataFrame(), spot, counts

    lazy = {}  # fetched only when a chain in the window actually needs them

    def lazy_get(key, fn):
        if key not in lazy:
            lazy[key] = fn(tk)
        return lazy[key]

    sorted_returns = {}

    def hist_win_pct(h, ratio):
        """Share of past h-day windows in which the stock finished at or above `ratio` x the start."""
        if h not in sorted_returns:
            sorted_returns[h] = np.sort(close_arr[h:] / close_arr[:-h]) if len(close_arr) > h + 60 else None
        arr = sorted_returns[h]
        if arr is None:
            return np.nan
        return 100.0 * (1 - np.searchsorted(arr, ratio, side="left") / len(arr))

    try:
        expirations = with_retry(lambda: tk.options)
    except Exception:
        return pd.DataFrame(), spot, counts

    max_strike = spot * (1 - p.min_otm_pct / 100)
    if p.below_sma:
        max_strike = min(max_strike, sma_50)

    candidates = []
    market_open_now = us_market_status()[0]

    for exp in expirations:
        dte = days_to_expiry(exp)
        if dte < p.min_dte or dte > p.max_dte or dte < 0:
            continue

        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        next_earnings = lazy_get("earnings", get_next_earnings_date)
        earnings_in_window = next_earnings is not None and date.today() <= next_earnings <= exp_date
        if p.exclude_earnings and earnings_in_window:
            continue

        try:
            chain = with_retry(lambda: tk.option_chain(exp))
        except Exception:
            continue

        puts = chain.puts
        if puts.empty:
            continue

        counts["puts_in_window"] += len(puts)

        T = dte / 365.0 if dte > 0 else intraday_years()
        hist_days = max(1, int(round(dte * 252 / 365)))
        ex_div = lazy_get("exdiv", get_ex_div_date)
        exdiv_in_window = ex_div is not None and date.today() <= ex_div <= exp_date

        for row in puts.itertuples(index=False):
            K = row.strike
            iv = row.impliedVolatility
            bid = getattr(row, "bid", 0) or 0
            ask = getattr(row, "ask", 0) or 0
            oi = getattr(row, "openInterest", 0)
            oi = 0 if pd.isna(oi) else oi
            vol = getattr(row, "volume", 0)
            vol = 0 if pd.isna(vol) else vol

            iv = float(iv) if iv is not None and not pd.isna(iv) else np.nan
            last = getattr(row, "lastPrice", 0) or 0
            stale = False
            if (bid <= 0 or ask <= 0) and (p.allow_stale or p.show_all) and last > 0:
                bid = ask = float(last)   # no live quote: show the last trade, clearly labeled
                stale = True

            # A put needs some price to be shown; otherwise there is nothing to display.
            if bid <= 0 or ask <= 0 or K > max_strike:
                continue
            premium = bid
            counts["otm_priced"] += 1

            if not p.show_all:
                if K * 100 > p.max_contract_cost:
                    continue
                counts["affordable"] += 1

                if premium * 100 < p.min_premium_usd:
                    continue
                counts["premium_ok"] += 1

                # Yahoo often reports openInterest as 0/blank for active contracts, so fall back to volume.
                liquidity = oi if oi > 0 else vol
                if liquidity < p.min_oi:
                    continue
                counts["liquidity"] += 1

            spread_pct = (ask - bid) / ask * 100
            if not p.show_all and spread_pct > p.max_spread_pct:
                continue
            counts["spread"] += 1

            dividend_yield = lazy_get("dividend", get_dividend_yield)
            if stale:
                # Yahoo's after-hours implied volatilities are placeholders (e.g. exactly 12.5% / 25%), so
                # derive IV from the last-trade price instead.
                iv = implied_vol_put(premium, spot, K, T, p.risk_free_rate, dividend_yield)
            iv_ok = not np.isnan(iv) and iv > 0
            if not p.show_all and (not iv_ok or iv < MIN_SANE_IV or iv > MAX_SANE_IV):
                continue
            counts["sane_iv"] += 1

            if iv_ok:
                delta, win_prob = bs_put_metrics(spot, K, T, p.risk_free_rate, dividend_yield, iv)
            else:
                delta = win_prob = np.nan
            if np.isnan(win_prob) and not p.show_all:
                continue

            win_prob_pct = win_prob * 100
            if not p.show_all and (win_prob_pct < p.win_prob_min or win_prob_pct > p.win_prob_max):
                continue
            counts["win_band"] += 1

            hist_win = hist_win_pct(hist_days, K / spot)
            if not p.show_all and p.min_hist_win > 0 and not np.isnan(hist_win) and hist_win < p.min_hist_win:
                continue
            counts["hist_ok"] += 1
            known = [x for x in (win_prob_pct, hist_win) if not np.isnan(x)]
            cons_win = min(known) if known else np.nan

            # Edge: how much richer the market price is than a fair price at recent realized vol.
            fair = bs_put_price(spot, K, T, p.risk_free_rate, dividend_yield, hv) if hv and hv >= 0.05 else np.nan
            edge_usd = (premium - fair) * 100 if not np.isnan(fair) else np.nan

            if p.require_edge and (np.isnan(edge_usd) or edge_usd <= 0):
                continue

            net_capital = K - premium
            yield_pct = (premium / net_capital) * 100
            annualized_yield = yield_pct * (365 / max(dte, 1))
            score = (cons_win / 100) * annualized_yield / leverage_multiplier
            iv_hv_ratio = (iv / hv) if hv and not np.isnan(hv) and hv > 0 else np.nan
            otm_pct = (spot - K) / spot * 100
            expected_move = spot * iv * np.sqrt(T)
            sigma_distance = (spot - K) / expected_move if expected_move > 0 else np.nan

            candidates.append({
                "Ticker": ticker_symbol,
                "Spot Price": round(spot, 2),
                "Prev Close": round(prev_close, 2),
                "Session": ext_session,
                "Ext %": round(ext_pct, 2) if not np.isnan(ext_pct) else np.nan,
                "Expiration": exp,
                "DTE": dte,
                "Strike": K,
                "% OTM": round(otm_pct, 1),
                "Distance (σ)": round(sigma_distance, 2),
                "Premium (Bid)": round(premium, 2),
                "Mid": round((bid + ask) / 2, 2),
                "Premium $": int(round(premium * 100)),
                "IV %": round(iv * 100, 1),
                "HV % (30d)": round(hv * 100, 1) if not np.isnan(hv) else np.nan,
                "IV/HV": round(iv_hv_ratio, 2) if not np.isnan(iv_hv_ratio) else np.nan,
                "Delta": round(delta, 3),
                "Est. Win Prob %": round(win_prob_pct, 1),
                "Hist. Win %": round(hist_win, 1) if not np.isnan(hist_win) else np.nan,
                "Win % (cons.)": round(cons_win, 1),
                "Edge $": round(edge_usd, 0) if not np.isnan(edge_usd) else np.nan,
                "Yield %": round(yield_pct, 2),
                "Return/Day %": round(yield_pct / max(dte, 1), 3),
                "Annualized Yield %": round(annualized_yield, 1),
                "Score": round(score, 2),
                "Lev": leverage_multiplier,
                "Breakeven": round(K - premium, 2),
                "Capital Req. $": int(round(K * 100)),
                "P&L @ -10% $": int(round(put_pnl_at_drop(spot, K, premium, 10) * 100)),
                "P&L @ -20% $": int(round(put_pnl_at_drop(spot, K, premium, 20) * 100)),
                "Exit Target ($)": f"${premium * 0.3:.2f}–${premium * 0.5:.2f}",
                "Next Earnings": next_earnings.strftime("%Y-%m-%d") if next_earnings else "—",
                "Earnings Alert": "⚠️ In Window" if earnings_in_window else "",
                "Ex-Div Alert": "Ex-div in window" if exdiv_in_window else "",
                "Quote": ("" if not stale else "Last trade (market closed)" if not market_open_now else "No live bid (last trade)"),
                "Trend": trend,
                "RSI": round(rsi14, 0) if not np.isnan(rsi14) else np.nan,
                "Open Interest": int(oi),
                "Volume": int(vol),
                "Spread %": round(spread_pct, 1),
                "50D SMA": round(sma_50, 2),
                "200D SMA": round(sma_200, 2) if not np.isnan(sma_200) else np.nan,
                "52W Low": round(low_52w, 2),
            })

    if not candidates:
        return pd.DataFrame(), spot, counts

    rank_col = RANK_OPTIONS.get(p.rank_by, "Score")
    df = pd.DataFrame(candidates).sort_values(by=rank_col, ascending=False, na_position="last")
    return df.head(p.top_n).reset_index(drop=True), spot, counts


def empty_reason(ticker, spot, counts, p):
    if spot is None:
        return f"{ticker}: could not fetch price/options data (Yahoo may be rate-limiting; try again shortly)."
    if counts.get("too_expensive"):
        return (f"{ticker}: too expensive for your cash — at ${spot:.2f}, puts near the current price need "
                f"about ${spot * 80:,.0f}+ per contract, above your ${p.max_contract_cost:,.0f} limit.")
    if counts.get("downtrend_skip"):
        return f"{ticker}: skipped — stock is in a downtrend (price < 50D SMA < 200D SMA)."
    if counts["puts_in_window"] == 0:
        return (f"{ticker}: no option chains found in the {p.min_dte}-{p.max_dte} day window "
                f"(spot: ${spot:.2f}), or every expiration was excluded by the earnings filter.")
    if counts["otm_priced"] and counts["affordable"] == 0:
        return (f"{ticker}: too expensive for your cash — every OTM put needs more than "
                f"${p.max_contract_cost:,.0f} per contract (spot: ${spot:.2f}).")
    stages = [
        ("puts_in_window", "puts in the window", None),
        ("otm_priced", "OTM with a live bid/ask", None),
        ("affordable", "within your deployable cash", "raise your cash or lower the reserve"),
        ("premium_ok", "paying at least the minimum premium", "lower 'Min premium per contract'"),
        ("liquidity", "with enough open interest/volume", "lower 'Min open interest'"),
        ("spread", "with a tight enough spread", "raise 'Max spread %'"),
        ("sane_iv", "with a sane implied volatility", None),
        ("win_band", "inside your win-probability band", "widen the 'Win probability %' range"),
        ("hist_ok", "meeting the historical-win floor", "lower 'Min historical win %'"),
    ]
    funnel = " → ".join(f"{counts[k]} {label}" for k, label, _ in stages)
    worst, worst_loss, prev = None, 0, counts["puts_in_window"]
    for k, label, tip in stages[1:]:
        loss = prev - counts[k]
        if tip and loss > worst_loss:
            worst, worst_loss = tip, loss
        prev = counts[k]
    hint = f" Biggest drop: try to {worst}." if worst else ""
    if counts["hist_ok"] and p.require_edge:
        hint = " Everything left had non-positive edge — untick 'Require positive edge'."
    return f"{ticker}: no candidates (spot: ${spot:.2f}). {funnel}.{hint}"


# ---------------- Market hours ----------------


def us_market_status():
    """(is_open, message) for regular US trading hours (9:30-16:00 New York time, Mon-Fri; holidays not checked)."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return True, ""
    minutes = now.hour * 60 + now.minute
    if now.weekday() < 5 and 9 * 60 + 30 <= minutes < 16 * 60:
        return True, ""
    return False, (f"The US market is closed (it is {now:%a %H:%M} in New York). Yahoo shows no live bids on options "
                   "outside trading hours, so scans can come back empty or use stale prices. Scan between about "
                   "9:45 AM and 4:00 PM New York time (Mon-Fri) for reliable results.")


# ---------------- Market regime ----------------


def get_market_regime():
    """VIX level and SPY trend, with plain-language sizing guidance."""
    out = {"vix": None, "spy": None, "spy_trend": None, "level": "unknown", "advice": ""}
    try:
        out["vix"] = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
    except Exception:
        pass
    try:
        spy = yf.Ticker("SPY").history(period="1y")["Close"].dropna()
        out["spy"] = float(spy.iloc[-1])
        out["spy_trend"] = trend_label(spy.iloc[-1], spy.tail(50).mean(),
                                       spy.tail(200).mean() if len(spy) >= 200 else np.nan)
    except Exception:
        pass

    vix, trend = out["vix"], out["spy_trend"]
    if vix is None:
        out["advice"] = "Could not read VIX; size conservatively."
        return out
    if vix < 15:
        out["level"], msg = "calm", "Calm market: premiums are thin. Stay selective and don't stretch for yield."
    elif vix < 20:
        out["level"], msg = "normal", "Normal volatility: standard sizing is reasonable."
    elif vix < 30:
        out["level"], msg = "elevated", "Elevated fear: richer premiums, but moves are bigger. Consider smaller size and wider distance."
    else:
        out["level"], msg = "high", "High fear: crashes and gaps are more likely. Cut size sharply and prefer liquid index/large-cap names."
    if trend == "Downtrend":
        msg += " The S&P 500 is in a downtrend, so favour strikes well below spot and fewer positions."
    out["advice"] = msg
    return out


# ---------------- Capital-aware trade plan ----------------


def build_plan(summary, capital, reserve_pct, max_pos_pct, max_positions, max_lev_pct, rank_col="Score",
               max_contracts=1):
    """Greedy basket, best-ranked first, one position per ticker, spread across up to `max_positions` tickers.
    `max_contracts` caps contracts per ticker (1 = one contract each, for maximum diversification; None = as many
    as an equal share of your cash allows). Positions that fit an equal share of your deployable cash go first;
    bigger single contracts are only added last, cheapest first. Returns (plan DataFrame, stats dict)."""
    budget = capital * (1 - reserve_pct / 100)
    pos_cap = capital * max_pos_pct / 100
    lev_cap = capital * max_lev_pct / 100
    alloc = min(pos_cap, budget / max(max_positions, 1))

    state = {"used": 0.0, "lev_used": 0.0}
    rows, seen = [], set()
    ranked = summary.sort_values(rank_col, ascending=False, na_position="last")

    def try_add(r, allow_oversize):
        cost = float(r["Capital Req. $"])
        avail = budget - state["used"]
        if r["Lev"] > 1:
            avail = min(avail, lev_cap - state["lev_used"])
        if cost <= 0 or cost > avail:
            return
        if cost > alloc and not allow_oversize:
            return  # bigger than an equal share: only as a last-pass fallback
        n = int(min(alloc, avail) // cost) if max_contracts != 1 else 1
        if max_contracts:
            n = min(n, max_contracts)
        n = max(n, 1)
        seen.add(r["Ticker"])
        state["used"] += n * cost
        if r["Lev"] > 1:
            state["lev_used"] += n * cost
        rows.append({
            "Ticker": r["Ticker"], "Expiration": r["Expiration"], "DTE": int(r["DTE"]),
            "Strike": r["Strike"], "Contracts": n, "Sell limit (mid)": r["Mid"],
            "Premium $": int(r["Premium $"]) * n, "Cash secured $": int(cost) * n,
            "Win % (cons.)": r["Win % (cons.)"], "Edge $": r["Edge $"] * n if pd.notna(r["Edge $"]) else np.nan,
            "P&L @ -10% $": int(r["P&L @ -10% $"]) * n, "P&L @ -20% $": int(r["P&L @ -20% $"]) * n,
            "Breakeven": r["Breakeven"], "Earnings Alert": r["Earnings Alert"], "Trend": r["Trend"],
            "Lev": int(r["Lev"]), "% of cash": round(n * cost / capital * 100, 1),
            "Size flag": "Above your preferred per-position max" if n * cost > pos_cap else "",
            "Relaxed": r.get("Relaxed", ""), "Quote": r.get("Quote", ""),
        })

    # Pass 1: candidates that fit an equal share of your cash, best-ranked first.
    # Pass 2: bigger single contracts, cheapest first, so leftover cash buys as many positions as possible.
    by_cost = ranked.sort_values("Capital Req. $", kind="stable")
    for allow_oversize, order in ((False, ranked), (True, by_cost)):
        for _, r in order.iterrows():
            if len(rows) >= max_positions:
                break
            if r["Ticker"] not in seen:
                try_add(r, allow_oversize)

    used = state["used"]
    plan = pd.DataFrame(rows)
    others = ranked[~ranked["Ticker"].isin(seen)].drop_duplicates("Ticker")
    left = budget - used
    if len(rows) >= max_positions:
        why = f"it reached your max of {max_positions} positions"
    elif others.empty:
        why = "every ticker that had candidates is already in the plan"
    elif left < float(others["Capital Req. $"].min()):
        why = f"the remaining candidates each need more cash than the ${left:,.0f} still available"
    else:
        why = "the remaining candidates were blocked by your leveraged-ETF cap"

    stats = {
        "deployed": used,
        "deployed_pct": used / capital * 100 if capital else 0,
        "premium": float(plan["Premium $"].sum()) if not plan.empty else 0.0,
        "worst_20": float(plan["P&L @ -20% $"].sum()) if not plan.empty else 0.0,
        "cash_left": capital - used,
        "positions": len(plan),
        "stop_reason": why,
        "others": others,
    }
    stats["return_pct"] = stats["premium"] / used * 100 if used else 0.0
    return plan, stats
