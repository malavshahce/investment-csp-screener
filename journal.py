"""Local trade journal for cash-secured puts, stored as a CSV next to the app.
Tracks open positions, marks them to live prices, suggests actions and measures real results."""
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

JOURNAL_PATH = Path(os.environ.get("CSP_JOURNAL_PATH") or Path(__file__).parent / "trade_journal.csv")

COLUMNS = ["id", "opened", "ticker", "expiration", "strike", "contracts", "premium",
           "est_win_pct", "status", "close_price", "closed", "notes"]
STATUSES = ["open", "closed", "expired", "assigned"]


def load_journal():
    if not JOURNAL_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(JOURNAL_PATH)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["notes"] = df["notes"].fillna("").astype(str)
    df["closed"] = df["closed"].fillna("").astype(str)
    return df[COLUMNS]


def save_journal(df):
    df = df.copy()
    df = df.dropna(subset=["ticker", "strike"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    next_id = int(pd.to_numeric(df["id"], errors="coerce").max() + 1) if df["id"].notna().any() else 1
    for i in df.index[df["id"].isna()]:
        df.at[i, "id"] = next_id
        next_id += 1
    df["id"] = df["id"].astype(int)
    df["opened"] = df["opened"].fillna(date.today().isoformat())
    df["status"] = df["status"].where(df["status"].isin(STATUSES), "open")
    df["contracts"] = df["contracts"].fillna(1).astype(int)
    df[COLUMNS].to_csv(JOURNAL_PATH, index=False)
    return df[COLUMNS]


def add_trades(plan, est_win_col="Win % (cons.)"):
    """Append plan rows as open trades. Skips ones already logged today. Returns number added."""
    df = load_journal()
    today = date.today().isoformat()
    added = 0
    for _, r in plan.iterrows():
        dup = ((df["ticker"] == r["Ticker"]) & (df["expiration"] == r["Expiration"])
               & (df["strike"].astype(float) == float(r["Strike"])) & (df["opened"] == today))
        if dup.any():
            continue
        premium_per_share = r["Premium $"] / (100 * r["Contracts"])
        df.loc[len(df)] = [np.nan, today, r["Ticker"], r["Expiration"], float(r["Strike"]),
                           int(r["Contracts"]), round(float(premium_per_share), 2),
                           r.get(est_win_col, np.nan), "open", np.nan, "", ""]
        added += 1
    if added:
        save_journal(df)
    return added


def realized_pnl(df):
    """Per-trade realized P&L in dollars for closed/expired/assigned rows; NaN for open."""
    prem = pd.to_numeric(df["premium"], errors="coerce")
    close = pd.to_numeric(df["close_price"], errors="coerce").fillna(0)
    n = pd.to_numeric(df["contracts"], errors="coerce").fillna(1)
    pnl = (prem - close) * 100 * n
    return pnl.where(df["status"].isin(["closed", "expired", "assigned"]))


def mark_to_market(df):
    """Fetch live spot and put price for open trades and add action suggestions."""
    open_df = df[df["status"] == "open"].copy()
    if open_df.empty:
        return open_df
    spots, chains = {}, {}
    out = []
    for _, r in open_df.iterrows():
        t, exp, K = r["ticker"], r["expiration"], float(r["strike"])
        spot = put_now = np.nan
        try:
            if t not in spots:
                spots[t] = float(yf.Ticker(t).history(period="5d")["Close"].iloc[-1])
            spot = spots[t]
            if (t, exp) not in chains:
                chains[(t, exp)] = yf.Ticker(t).option_chain(exp).puts
            puts = chains[(t, exp)]
            row = puts.loc[(puts["strike"] - K).abs() < 1e-6].iloc[0]
            bid, ask, last = row.get("bid", 0) or 0, row.get("ask", 0) or 0, row.get("lastPrice", 0) or 0
            put_now = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        except Exception:
            pass

        prem = float(r["premium"])
        n = int(r["contracts"])
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - date.today()).days
        captured = (prem - put_now) / prem * 100 if not np.isnan(put_now) and prem > 0 else np.nan
        dist = (spot - K) / spot * 100 if not np.isnan(spot) else np.nan

        if np.isnan(put_now):
            action = "No live quote — check your broker"
        elif dte < 0:
            action = "Expired — mark it as expired/assigned"
        elif spot <= K:
            action = "ITM — roll out/down or accept assignment"
        elif captured >= 70:
            action = "Close now (≥70% of premium captured)"
        elif captured >= 50:
            action = "Consider closing (≥50% captured)"
        elif dist < 3:
            action = "At risk — within 3% of strike, consider rolling"
        elif dte <= 1:
            action = "Expires soon — let it expire or close"
        else:
            action = "Hold"

        out.append({
            "id": int(r["id"]), "Ticker": t, "Expiration": exp, "DTE": dte, "Strike": K,
            "Contracts": n, "Sold at": prem, "Spot": round(spot, 2) if not np.isnan(spot) else np.nan,
            "% above strike": round(dist, 1) if not np.isnan(dist) else np.nan,
            "Put now": round(put_now, 2) if not np.isnan(put_now) else np.nan,
            "Unrealized P&L $": round((prem - put_now) * 100 * n) if not np.isnan(put_now) else np.nan,
            "% of premium captured": round(captured, 0) if not np.isnan(captured) else np.nan,
            "Action": action,
        })
    return pd.DataFrame(out)


def journal_stats(df):
    """Realized results and model-vs-reality calibration."""
    pnl = realized_pnl(df)
    done = df[pnl.notna()].copy()
    done["pnl"] = pnl[pnl.notna()]
    if done.empty:
        return None
    resolved = done[done["status"].isin(["closed", "expired"])]
    wins = (resolved["pnl"] > 0).sum()
    est = pd.to_numeric(resolved["est_win_pct"], errors="coerce")
    return {
        "trades": len(done),
        "total_pnl": float(done["pnl"].sum()),
        "win_rate": wins / len(resolved) * 100 if len(resolved) else np.nan,
        "avg_pnl": float(done["pnl"].mean()),
        "avg_est_win": float(est.mean()) if est.notna().any() else np.nan,
        "assigned": int((done["status"] == "assigned").sum()),
        "best": float(done["pnl"].max()),
        "worst": float(done["pnl"].min()),
    }
