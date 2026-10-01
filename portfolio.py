"""Private portfolio + watchlist storage and valuation. Files live next to the app, are git-ignored,
and are only shown behind the app password."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import markets as MK

BASE = Path(__file__).parent
PORTFOLIO_PATH = Path(os.environ.get("CSP_PORTFOLIO_PATH") or BASE / "portfolio.csv")
WATCHLIST_PATH = Path(os.environ.get("CSP_WATCHLIST_PATH") or BASE / "watchlist.csv")

HOLDING_COLUMNS = ["ticker", "shares", "avg_cost", "account", "notes"]
WATCH_COLUMNS = ["ticker", "alert_above", "alert_below", "note"]

# Common broker export column names -> our names
_ALIASES = {
    "ticker": ["ticker", "symbol", "instrument", "security"],
    "shares": ["shares", "quantity", "qty", "units", "position"],
    "avg_cost": ["avg_cost", "average cost", "avg cost", "average price", "avg price", "cost basis per share",
                 "price paid", "unit cost", "average cost basis"],
    "account": ["account", "account name", "portfolio"],
    "notes": ["notes", "note", "description"],
}


def _load(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[columns]


def load_holdings():
    df = _load(PORTFOLIO_PATH, HOLDING_COLUMNS)
    df["account"] = df["account"].fillna("").astype(str)
    df["notes"] = df["notes"].fillna("").astype(str)
    return df


def save_holdings(df):
    df = df.copy().dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    df = df[df["ticker"] != ""]
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce").fillna(0.0)
    df[HOLDING_COLUMNS].to_csv(PORTFOLIO_PATH, index=False)
    return df[HOLDING_COLUMNS]


def normalize_import(df):
    """Map a broker CSV (any common column naming) onto our holding columns. Returns (df, missing list)."""
    lower = {c.lower().strip(): c for c in df.columns}
    out, missing = pd.DataFrame(), []
    for target, names in _ALIASES.items():
        src = next((lower[n] for n in names if n in lower), None)
        if src is None:
            if target in ("ticker", "shares"):
                missing.append(target)
            out[target] = "" if target in ("account", "notes") else np.nan
        else:
            out[target] = df[src]
    for c in ("shares", "avg_cost"):
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    return out[HOLDING_COLUMNS], missing


def load_watchlist():
    return _load(WATCHLIST_PATH, WATCH_COLUMNS).assign(note=lambda d: d["note"].fillna("").astype(str))


def save_watchlist(df):
    df = df.copy().dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"] != ""].drop_duplicates("ticker")
    df[WATCH_COLUMNS].to_csv(WATCHLIST_PATH, index=False)
    return df[WATCH_COLUMNS]


# ---------------- Valuation ----------------


def value_holdings(holdings):
    """Join holdings with live quotes and company facts. Returns (DataFrame, fetched_at)."""
    h = holdings[(holdings["shares"] > 0) & holdings["ticker"].notna()].copy()
    if h.empty:
        return pd.DataFrame(), None
    symbols = tuple(h["ticker"].tolist())
    quotes, fetched = MK.get_quotes(symbols)
    q = quotes.set_index("Symbol") if not quotes.empty else pd.DataFrame()

    profiles, _ = MK.get_profiles(symbols)

    rows = []
    for _, r in h.iterrows():
        s = r["ticker"]
        if s not in q.index:
            rows.append({"Ticker": s, "Name": "(no quote)", "Shares": r["shares"], "Avg Cost": r["avg_cost"],
                         "Account": r["account"]})
            continue
        price = float(q.loc[s, "Price"])
        prof = profiles.get(s, {})
        rate = MK.forward_dividend(s, prof)
        value = r["shares"] * price
        cost = r["shares"] * r["avg_cost"]
        sector = prof.get("sector") or ("ETF / Fund" if prof.get("quoteType") in ("ETF", "MUTUALFUND") else "Other")
        rows.append({
            "Ticker": s, "Name": prof.get("shortName") or prof.get("longName") or s, "Account": r["account"],
            "Shares": r["shares"], "Avg Cost": r["avg_cost"], "Price": price,
            "Value": value, "Cost": cost, "Gain $": value - cost,
            "Gain %": (value / cost - 1) * 100 if cost else np.nan,
            "Day $": r["shares"] * float(q.loc[s, "Change"]), "Day %": float(q.loc[s, "Day %"]),
            "Div/Share": rate, "Annual Div $": r["shares"] * rate,
            "Yield %": rate / price * 100 if price else 0.0,
            "Yield on Cost %": rate / r["avg_cost"] * 100 if r["avg_cost"] else np.nan,
            "Beta": prof.get("beta", np.nan), "Sector": sector, "Rating": prof.get("recommendationKey", ""),
        })
    df = pd.DataFrame(rows)
    if "Value" in df and df["Value"].notna().any():
        df["Weight %"] = df["Value"] / df["Value"].sum() * 100
    return df, fetched


def summarize(df):
    """Portfolio-level totals from a valued-holdings frame."""
    d = df.dropna(subset=["Value"])
    if d.empty:
        return {}
    total, cost = d["Value"].sum(), d["Cost"].sum()
    beta_rows = d.dropna(subset=["Beta"])
    return {
        "value": total, "cost": cost, "gain": total - cost, "gain_pct": (total / cost - 1) * 100 if cost else np.nan,
        "day": d["Day $"].sum(), "day_pct": d["Day $"].sum() / (total - d["Day $"].sum()) * 100 if total else 0.0,
        "annual_div": d["Annual Div $"].sum(), "yield_pct": d["Annual Div $"].sum() / total * 100 if total else 0.0,
        "yoc_pct": d["Annual Div $"].sum() / cost * 100 if cost else np.nan,
        "beta": float((beta_rows["Beta"] * beta_rows["Value"]).sum() / beta_rows["Value"].sum()) if len(beta_rows) else np.nan,
        "positions": len(d),
        "payers": int((d["Annual Div $"] > 0).sum()),
    }


def insights(df):
    """Plain-language concentration / risk notes."""
    d = df.dropna(subset=["Value"])
    notes = []
    if d.empty:
        return notes
    top = d.sort_values("Value", ascending=False).iloc[0]
    if top["Weight %"] > 25:
        notes.append(f"{top['Ticker']} is {top['Weight %']:.0f}% of your portfolio. One stock above ~25% makes results "
                     "depend heavily on it.")
    sectors = d.groupby("Sector")["Value"].sum() / d["Value"].sum() * 100
    if sectors.max() > 40:
        notes.append(f"{sectors.idxmax()} makes up {sectors.max():.0f}% of the portfolio. Consider spreading across sectors.")
    if len(d) < 8:
        notes.append(f"Only {len(d)} positions. Most investors diversify across at least 8-10 holdings or use broad ETFs.")
    big_loss = d[d["Gain %"] < -20]
    if not big_loss.empty:
        notes.append("Down more than 20% vs your cost: " + ", ".join(big_loss["Ticker"]) + ".")
    payers = d[d["Annual Div $"] > 0]
    if payers.empty:
        notes.append("None of your holdings pay a dividend, so the dividend and DRIP projections will be empty.")
    return notes
