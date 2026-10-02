"""Market data layer (yfinance). Every cached function returns the time it was actually fetched so the UI
can show an honest 'updated at' stamp. Caches are short so data stays fresh; the header's Refresh button
clears them."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import calc
import engine as E
import symbols as SYM

QUOTE_TTL = 60

INDICES = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "Russell 2000": "^RUT", "VIX": "^VIX"}
MACRO = {"10-yr yield": "^TNX", "13-wk T-bill": "^IRX", "Gold": "GC=F", "Oil (WTI)": "CL=F",
         "Bitcoin": "BTC-USD", "US Dollar": "DX-Y.NYB"}
SECTORS = {"Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health Care": "XLV",
           "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Industrials": "XLI",
           "Utilities": "XLU", "Materials": "XLB", "Real Estate": "XLRE", "Communication": "XLC"}

PERIODS = {  # label -> (history period, interval)
    "1D": ("1d", "5m"), "5D": ("5d", "15m"), "1M": ("1mo", "1d"), "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"), "5Y": ("5y", "1wk"), "Max": ("max", "1mo"),
}


def _now():
    return datetime.now()


# ---------------- Quotes ----------------


def _extract(raw, sym):
    """Pull one symbol's OHLCV frame out of a (possibly multi-level) yf.download result."""
    if raw is None or raw.empty:
        return None
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if sym in raw.columns.get_level_values(0):
                return raw[sym]
            if sym in raw.columns.get_level_values(1):
                return raw.xs(sym, axis=1, level=1)
            return None
        return raw
    except Exception:
        return None


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def get_quotes(symbols):
    """Latest price and day / 5-day change for a tuple of symbols in one request.
    Returns (DataFrame, fetched_at)."""
    symbols = tuple(dict.fromkeys(s.upper() for s in symbols))
    rows = []
    try:
        raw = E.with_retry(lambda: yf.download(list(symbols), period="1mo", progress=False, group_by="ticker",
                                               threads=True, auto_adjust=True))
    except Exception:
        raw = None
    for s in symbols:
        sub = _extract(raw, s)
        if sub is None or "Close" not in sub:
            continue
        close = sub["Close"].dropna()
        if len(close) < 2:
            continue
        last, prev = float(close.iloc[-1]), float(close.iloc[-2])
        five = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])
        vol = sub["Volume"].dropna()
        rows.append({
            "Symbol": s, "Price": last, "Prev Close": prev, "Change": last - prev,
            "Day %": (last / prev - 1) * 100, "5D %": (last / five - 1) * 100,
            "1M %": (last / float(close.iloc[0]) - 1) * 100,
            "Volume": float(vol.iloc[-1]) if len(vol) else np.nan,
            "Trend (1M)": [float(x) for x in close.tail(22)],
        })
    return pd.DataFrame(rows), _now()


def resolve_symbols(symbols):
    """Find a live quote for each symbol. A plain symbol with no quote (say RY when you hold the Toronto listing) is
    matched to its Canadian listing by trying .TO, .V, .CN and .NE. Returns ({symbol: resolved or None}, quotes)."""
    symbols = tuple(dict.fromkeys(symbols))
    quotes, _ = get_quotes(symbols)
    have = set(quotes["Symbol"]) if not quotes.empty else set()
    missing = [s for s in symbols if s not in have]
    resolved = {s: (s if s in have else None) for s in symbols}
    cands = {s + suf: s for s in missing if not SYM.suffix_of(s) for suf in SYM.CANADA_SUFFIXES}
    if cands:
        extra, _ = get_quotes(tuple(cands))
        got = set(extra["Symbol"]) if not extra.empty else set()
        for s in missing:
            for suf in SYM.CANADA_SUFFIXES:
                if s + suf in got:
                    resolved[s] = s + suf
                    break
        quotes = pd.concat([quotes, extra], ignore_index=True)
    return resolved, quotes


@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol, period_label="1Y"):
    """Price history for the Lookup chart. Returns (DataFrame, fetched_at)."""
    period, interval = PERIODS[period_label]
    tk = yf.Ticker(symbol)
    df = E.with_retry(lambda: tk.history(period=period, interval=interval, auto_adjust=True))
    return df, _now()


@st.cache_data(ttl=600, show_spinner=False)
def get_price_matrix(symbols, period="1y"):
    """Daily closes for several symbols (one column each). Returns (DataFrame, fetched_at)."""
    symbols = tuple(dict.fromkeys(s.upper() for s in symbols))
    raw = E.with_retry(lambda: yf.download(list(symbols), period=period, progress=False, group_by="ticker",
                                           threads=True, auto_adjust=True))
    cols = {}
    for s in symbols:
        sub = _extract(raw, s)
        if sub is not None and "Close" in sub:
            cols[s] = sub["Close"]
    df = pd.DataFrame(cols).dropna(how="all").ffill()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df, _now()


# ---------------- Company data ----------------

_PROFILE_KEYS = [
    "shortName", "longName", "quoteType", "sector", "industry", "country", "currency", "marketCap",
    "trailingPE", "forwardPE", "priceToBook", "dividendYield", "dividendRate", "trailingAnnualDividendRate",
    "payoutRatio", "exDividendDate", "fiveYearAvgDividendYield", "beta", "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow", "fiftyDayAverage", "twoHundredDayAverage", "targetMeanPrice", "targetHighPrice",
    "targetLowPrice", "recommendationKey", "numberOfAnalystOpinions", "earningsGrowth", "revenueGrowth",
    "profitMargins", "returnOnEquity", "debtToEquity", "longBusinessSummary", "website",
    "fullTimeEmployees", "averageVolume", "category", "totalAssets", "annualReportExpenseRatio",
]


def _fetch_profile(symbol):
    """Uncached company / fund facts (safe to call from worker threads). Empty dict if Yahoo has nothing."""
    try:
        info = E.with_retry(lambda: yf.Ticker(symbol).info)
    except Exception:
        return {}
    return {k: info.get(k) for k in _PROFILE_KEYS if info.get(k) is not None}


@st.cache_data(ttl=900, show_spinner=False)
def get_profile(symbol):
    """Company / fund facts. Returns (dict, fetched_at). The dict is empty if Yahoo has nothing."""
    return _fetch_profile(symbol), _now()


@st.cache_data(ttl=900, show_spinner=False)
def get_profiles(symbols):
    """Facts for many symbols, fetched in parallel. Returns ({symbol: dict}, fetched_at)."""
    symbols = tuple(dict.fromkeys(symbols))
    with ThreadPoolExecutor(max_workers=6) as pool:
        profs = list(pool.map(_fetch_profile, symbols))
    return dict(zip(symbols, profs)), _now()


def annual_dividend_rate(profile):
    """Best available forward annual dividend per share from a profile dict."""
    for k in ("dividendRate", "trailingAnnualDividendRate"):
        v = profile.get(k)
        if v:
            return float(v)
    return 0.0


def forward_dividend(symbol, profile):
    """Annual dividend per share. Uses the profile's rate; for funds/ETFs (where Yahoo omits it) falls back to
    the sum of dividends actually paid over the last 12 months."""
    rate = annual_dividend_rate(profile)
    if rate:
        return rate
    div, _ = get_dividend_history(symbol)
    stats = dividend_stats(div)
    return stats["ttm"] if stats.get("has_dividends") else 0.0


def dividend_yield_pct(profile, price=None):
    """Yield in percent. yfinance reports dividendYield as a percent in recent versions; derive it from the
    rate and price when possible so the number can't be off by 100x."""
    rate = annual_dividend_rate(profile)
    if rate and price:
        return rate / price * 100
    dy = profile.get("dividendYield") or 0
    return float(dy) if dy and dy > 0.5 else float(dy) * 100 if dy else 0.0


@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol, limit=8):
    """Recent headlines. Returns (list of dicts, fetched_at)."""
    out = []
    try:
        items = E.with_retry(lambda: yf.Ticker(symbol).news) or []
    except Exception:
        items = []
    for it in items[:limit]:
        c = it.get("content", it)
        title = c.get("title")
        if not title:
            continue
        link = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                or c.get("link"))
        when = c.get("pubDate")
        if when:
            try:
                when = pd.to_datetime(when).to_pydatetime().replace(tzinfo=None)
            except Exception:
                when = None
        elif c.get("providerPublishTime"):
            when = datetime.fromtimestamp(c["providerPublishTime"])
        out.append({"title": title, "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher", ""),
                    "link": link, "published": when, "summary": c.get("summary") or ""})
    return out, _now()


@st.cache_data(ttl=3600, show_spinner=False)
def get_dividend_history(symbol):
    """Every dividend Yahoo knows about. Returns (DataFrame[Date, Dividend], fetched_at)."""
    try:
        s = E.with_retry(lambda: yf.Ticker(symbol).dividends)
    except Exception:
        s = pd.Series(dtype=float)
    if s is None or s.empty:
        return pd.DataFrame(columns=["Date", "Dividend"]), _now()
    idx = s.index.tz_localize(None) if s.index.tz is not None else s.index
    return pd.DataFrame({"Date": idx, "Dividend": s.values}), _now()


def dividend_stats(div_df, price=None):
    """Trailing-12-month income, payout frequency, growth rate and increase streak from dividend history."""
    if div_df.empty:
        return {"has_dividends": False}
    now = pd.Timestamp.now()
    last12 = div_df[div_df["Date"] > now - pd.Timedelta(days=365)]
    ttm = float(last12["Dividend"].sum())
    count = len(last12)
    by_year = div_df.groupby(div_df["Date"].dt.year)["Dividend"].sum()
    full = by_year[by_year.index < now.year]
    growth = float("nan")
    if len(full) >= 2:
        span = full.tail(6)
        growth = calc.cagr(float(span.iloc[0]), float(span.iloc[-1]), len(span) - 1)
    streak = 0
    vals = list(full.values)
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] > vals[i - 1] * 1.0001:
            streak += 1
        else:
            break
    return {
        "has_dividends": True, "ttm": ttm, "count": count, "frequency": calc.freq_from_count(count),
        "last": float(div_df["Dividend"].iloc[-1]), "last_date": div_df["Date"].iloc[-1],
        "growth_cagr": growth, "streak": streak, "yield_pct": ttm / price * 100 if price else float("nan"),
        "by_year": by_year,
    }


# ---------------- Search ----------------


@st.cache_data(ttl=300, show_spinner=False)
def search_symbols(query, limit=8):
    """Company / ticker search. Returns a list of {symbol, name, type, exchange}."""
    q = (query or "").strip()
    if not q:
        return []
    try:
        hits = E.with_retry(lambda: yf.Search(q, max_results=limit, news_count=0).quotes, attempts=2)
    except Exception:
        return []
    keep = {"EQUITY", "ETF", "INDEX", "MUTUALFUND", "CRYPTOCURRENCY", "CURRENCY", "FUTURE"}
    out = []
    for h in hits:
        if h.get("quoteType") in keep and h.get("symbol"):
            out.append({"symbol": h["symbol"], "name": h.get("shortname") or h.get("longname") or "",
                        "type": h.get("quoteType"), "exchange": h.get("exchDisp") or ""})
    return out


# ---------------- Market regime & movers ----------------


@st.cache_data(ttl=300, show_spinner=False)
def get_regime():
    return E.get_market_regime(), _now()


@st.cache_data(ttl=900, show_spinner=False)
def get_events(symbols):
    """Next earnings date and ex-dividend date for each symbol (slow, so cached longer).
    Returns (DataFrame, fetched_at)."""
    def one(sym):
        tk = yf.Ticker(sym)
        return {"Symbol": sym, "Next Earnings": E.get_next_earnings_date(tk), "Ex-Dividend": E.get_ex_div_date(tk)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, symbols))
    return pd.DataFrame(rows), _now()


@st.cache_data(ttl=300, show_spinner=False)
def get_same_day_expiries(symbols):
    """Which of these symbols have options expiring today (New York date)?  Returns (DataFrame, fetched_at).
    Columns: Symbol, Price, Day %, Expiries this week (count). Empty on weekends/holidays."""
    today = E.ny_today()
    today_iso = today.isoformat()

    def one(sym):
        try:
            opts = yf.Ticker(sym).options
        except Exception:
            return None
        if today_iso not in opts:
            return None
        week = sum(1 for o in opts if 0 <= (datetime.strptime(o, "%Y-%m-%d").date() - today).days <= 7)
        return {"Symbol": sym, "Expiries this week": week}

    with ThreadPoolExecutor(max_workers=8) as pool:
        hits = [h for h in pool.map(one, symbols) if h]
    if not hits:
        return pd.DataFrame(columns=["Symbol", "Price", "Day %", "Expiries this week"]), _now()
    quotes, _ = get_quotes(tuple(h["Symbol"] for h in hits))
    q = quotes.set_index("Symbol") if not quotes.empty else pd.DataFrame()
    rows = []
    for h in hits:
        s = h["Symbol"]
        rows.append({"Symbol": s, "Price": float(q.loc[s, "Price"]) if s in q.index else float("nan"),
                     "Day %": float(q.loc[s, "Day %"]) if s in q.index else float("nan"),
                     "Expiries this week": h["Expiries this week"]})
    df = pd.DataFrame(rows)
    ext, _ = get_extended_prices(tuple(df["Symbol"]))
    if not ext.empty:
        e = ext.set_index("Symbol")
        df["Pre/after-hours price"] = df["Symbol"].map(e["Ext Price"])
        df["Pre/after-hours %"] = df["Symbol"].map(e["Ext %"])
    return df, _now()


@st.cache_data(ttl=60, show_spinner=False)
def get_extended_prices(symbols):
    """Pre-market (4:00-9:30 AM New York) and after-hours (4:00-8:00 PM) stock prices. Options do not trade in these
    sessions, but the stock does, and a gap changes how safe a strike is. Returns (DataFrame, fetched_at) with
    Symbol, Ext Price, Session, Ext Time, Close, Ext %; empty during regular hours or when nothing has traded."""
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    symbols = tuple(dict.fromkeys(s.upper() for s in symbols))
    rows = []
    if symbols and not E.us_market_status()[0]:
        quotes, _ = get_quotes(symbols)                       # last regular-session close for each symbol
        closes = quotes.set_index("Symbol")["Price"] if not quotes.empty else pd.Series(dtype=float)
        today = E.ny_today()
        for i in range(0, len(symbols), 120):                 # chunks keep each download small and fast
            chunk = symbols[i:i + 120]
            try:
                raw = E.with_retry(lambda: yf.download(list(chunk), period="1d", interval="1m", prepost=True,
                                                       progress=False, group_by="ticker", threads=True,
                                                       auto_adjust=True), attempts=2)
            except Exception:
                continue
            for sym in chunk:
                sub = _extract(raw, sym)
                if sub is None or "Close" not in sub or sym not in closes.index:
                    continue
                c = sub["Close"].dropna()
                if c.empty:
                    continue
                t = c.index[-1].tz_convert(ny) if c.index.tz is not None else c.index[-1]
                if t.date() != today:
                    continue
                hhmm = t.hour * 60 + t.minute
                session = "Pre-market" if hhmm < 9 * 60 + 30 else "After hours" if hhmm >= 16 * 60 else None
                if not session:
                    continue
                price, close = float(c.iloc[-1]), float(closes[sym])
                rows.append({"Symbol": sym, "Ext Price": price, "Session": session, "Ext Time": f"{t:%H:%M}",
                             "Close": close, "Ext %": (price / close - 1) * 100})
    cols = ["Symbol", "Ext Price", "Session", "Ext Time", "Close", "Ext %"]
    return pd.DataFrame(rows, columns=cols), _now()


def clear_live_caches():
    """Drop every cached live dataset so the next render downloads fresh data."""
    for fn in (get_quotes, get_history, get_price_matrix, get_profile, get_profiles, get_news, get_dividend_history,
               get_regime, get_events, search_symbols, get_same_day_expiries, get_extended_prices):
        fn.clear()


def rsi_and_trend(symbol):
    """RSI(14) and trend label from ~1y of history (reuses the scanner's indicators)."""
    df, _ = get_history(symbol, "1Y")
    close = df["Close"].dropna() if not df.empty else pd.Series(dtype=float)
    if len(close) < 60:
        return np.nan, "—", close
    sma50 = close.tail(50).mean()
    sma200 = close.tail(200).mean() if len(close) >= 200 else np.nan
    return E.compute_rsi(close), E.trend_label(close.iloc[-1], sma50, sma200), close
