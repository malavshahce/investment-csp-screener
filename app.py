import hmac
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import engine as E
import journal as J
import markets as MK
import view_income
import view_lookup
import view_market
import view_portfolio
import view_watchlist
from common import REFRESH_CHOICES, md, money, scatter

st.set_page_config(page_title="Cash-Secured Put Screener", layout="wide", initial_sidebar_state="collapsed")

THEMES = {
    "Midnight":     dict(dark=True,  bg="#0e1117", bg2="#161b26", text="#e6e9ef", muted="#9aa4b5", primary="#00c2a8", on_primary="#04130f", border="rgba(255,255,255,0.14)"),
    "Ocean":        dict(dark=True,  bg="#0b1e2d", bg2="#11304a", text="#e3f2fd", muted="#8fb3cc", primary="#29b6f6", on_primary="#03202f", border="rgba(130,200,255,0.25)"),
    "Forest":       dict(dark=True,  bg="#0f1f17", bg2="#16301f", text="#e6f4ea", muted="#93b59d", primary="#66bb6a", on_primary="#07200d", border="rgba(150,230,170,0.22)"),
    "Sunset":       dict(dark=True,  bg="#1f1219", bg2="#2e1a26", text="#fbe9e7", muted="#c9a39c", primary="#ff7043", on_primary="#2a0d04", border="rgba(255,170,140,0.25)"),
    "Royal Purple": dict(dark=True,  bg="#140f24", bg2="#1f1838", text="#ede7f6", muted="#a99bc9", primary="#b388ff", on_primary="#1a0b33", border="rgba(200,170,255,0.25)"),
    "Terminal":     dict(dark=True,  bg="#000000", bg2="#0a140a", text="#33ff66", muted="#22aa44", primary="#33ff66", on_primary="#000000", border="rgba(51,255,102,0.35)", mono=True),
    "Light":        dict(dark=False, bg="#ffffff", bg2="#f3f5f9", text="#1b2430", muted="#5d6b7e", primary="#0a7cff", on_primary="#ffffff", border="#d5dbe5"),
    "Paper":        dict(dark=False, bg="#faf6ee", bg2="#f0e9da", text="#2b2a26", muted="#7a7261", primary="#b5651d", on_primary="#ffffff", border="#ddd2bb"),
}
DEFAULT_THEME = "Midnight"


def theme_css(t):
    # Streamlit draws tables on a canvas that ignores CSS colors; flip the dark grid to light on light themes.
    chip_fix = ("" if t["dark"] else
                '.stApp [data-testid="stMetricDelta"] { filter: brightness(0.52) saturate(1.5); }')
    table_fix = ("" if t["dark"] else
                 '.stApp [data-testid="stDataFrame"], .stApp [data-testid="stDataEditor"] '
                 '{ filter: invert(1) hue-rotate(180deg); }')
    font = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" if t.get("mono") else "inherit"
    return f"""
<style>
:root, .stApp {{ color-scheme: {"dark" if t["dark"] else "light"}; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {{
    background: {t["bg"]} !important; color: {t["text"]} !important; font-family: {font};
}}
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stWidgetLabel"] {{ color: {t["text"]}; }}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * {{ color: {t["muted"]} !important; }}
/* inputs */
.stApp input, .stApp textarea, .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="textarea"] {{
    background: {t["bg2"]} !important; color: {t["text"]} !important; border-color: {t["border"]} !important; }}
.stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] * {{ color: {t["text"]}; }}
.stApp [data-baseweb="tag"] {{ background: {t["primary"]} !important; }}
.stApp [data-baseweb="tag"] * {{ color: {t["on_primary"]} !important; }}
[data-baseweb="popover"] *, [data-baseweb="menu"], [role="listbox"] {{ background: {t["bg2"]}; color: {t["text"]}; }}
/* containers */
.stApp [data-testid="stForm"], .stApp [data-testid="stExpander"], .stApp [data-testid="stVerticalBlockBorderWrapper"],
.stApp [data-testid="stExpander"] details, .stApp [data-testid="stStatusWidget"] {{
    border-color: {t["border"]} !important; }}
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * {{ background: transparent; color: {t["text"]}; }}
/* tabs */
.stApp [data-baseweb="tab"] {{ color: {t["muted"]}; }}
.stApp [data-baseweb="tab"][aria-selected="true"], .stApp [data-baseweb="tab"][aria-selected="true"] * {{ color: {t["primary"]}; }}
.stApp [data-baseweb="tab-highlight"] {{ background: {t["primary"]} !important; }}
.stApp [data-baseweb="tab-border"] {{ background: {t["border"]} !important; }}
/* buttons */
.stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"], .stApp [data-testid="stBaseButton-primary"],
.stApp [data-testid="stBaseButton-primaryFormSubmit"] {{
    background: {t["primary"]} !important; border-color: {t["primary"]} !important; }}
.stApp button[kind="primary"] *, .stApp button[kind="primaryFormSubmit"] *, .stApp [data-testid="stBaseButton-primary"] *,
.stApp [data-testid="stBaseButton-primaryFormSubmit"] * {{ color: {t["on_primary"]} !important; }}
.stApp button[kind="secondary"], .stApp [data-testid="stBaseButton-secondary"], .stApp [data-testid="stBaseButton-secondaryFormSubmit"] {{
    background: {t["bg2"]} !important; border-color: {t["border"]} !important; }}
.stApp button[kind="secondary"] *, .stApp [data-testid="stBaseButton-secondary"] * {{ color: {t["text"]} !important; }}
/* sliders, radio, checkbox accents */
.stApp [data-baseweb="slider"] [role="slider"] {{ background: {t["primary"]} !important; }}
.stApp [data-testid="stSliderThumbValue"], .stApp [data-testid="stTickBarMin"], .stApp [data-testid="stTickBarMax"] {{ color: {t["primary"]} !important; }}
.stApp input[type="checkbox"], .stApp input[type="radio"] {{ accent-color: {t["primary"]}; }}
/* progress bar track */
.stApp [data-testid="stProgress"] [role="progressbar"] > div {{ background-color: {t["bg2"]} !important; }}
.stApp [data-testid="stProgress"] [role="progressbar"] > div > div {{ background-color: transparent !important; }}
.stApp [data-testid="stProgress"] [role="progressbar"] > div > div > div {{ background-color: {t["primary"]} !important; }}
/* tables (canvas grid reads these variables) */
.stApp [data-testid="stDataFrame"], .stApp [data-testid="stDataFrameResizable"], .stApp [data-testid="stDataEditor"] {{
    --gdg-bg-cell: {t["bg"]}; --gdg-bg-cell-medium: {t["bg2"]}; --gdg-bg-header: {t["bg2"]};
    --gdg-bg-header-has-focus: {t["bg2"]}; --gdg-bg-header-hovered: {t["bg2"]};
    --gdg-text-dark: {t["text"]}; --gdg-text-medium: {t["muted"]}; --gdg-text-light: {t["muted"]};
    --gdg-text-header: {t["muted"]}; --gdg-border-color: {t["border"]}; --gdg-horiz-border-color: {t["border"]};
    --gdg-accent-color: {t["primary"]}; --gdg-accent-light: {t["bg2"]};
    border-color: {t["border"]}; }}
.stApp [data-testid="stDataFrame"] > div, .stApp [data-testid="stDataEditor"] > div {{ background: {t["bg"]} !important; }}
.stApp hr {{ border-color: {t["border"]}; }}
.stApp ::placeholder {{ color: {t["muted"]} !important; opacity: 0.85 !important; }}
.stApp [data-testid="stTooltipIcon"] svg {{ color: {t["muted"]}; fill: {t["muted"]}; }}
/* number input +/- buttons */
.stApp [data-testid="stNumberInputStepDown"], .stApp [data-testid="stNumberInputStepUp"] {{
    background: {t["bg2"]} !important; color: {t["text"]} !important; border-color: {t["border"]} !important; }}
.stApp [data-testid="stNumberInputStepDown"] *, .stApp [data-testid="stNumberInputStepUp"] * {{ color: {t["text"]} !important; fill: {t["text"]} !important; }}
/* radio + checkbox: themed fill, primary when checked */
.stApp [data-baseweb="radio"] > div:first-child, .stApp [data-baseweb="checkbox"] > span:first-child {{
    background-color: {t["bg2"]} !important; border-color: {t["muted"]} !important; }}
.stApp [data-baseweb="radio"]:has(input:checked) > div:first-child, .stApp [data-baseweb="checkbox"]:has(input:checked) > span:first-child {{
    background-color: {t["primary"]} !important; border-color: {t["primary"]} !important; }}
{table_fix}
{chip_fix}
/* layout */
.block-container {{padding-top: 2.2rem; max-width: 1400px;}}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{display: none;}}
.hero h1 {{margin: 0; font-size: 2rem; letter-spacing: -0.02em; color: {t["text"]};}}
.hero p {{margin: 0.2rem 0 0; color: {t["muted"]}; font-size: 0.92rem;}}
[data-testid="stMetricValue"] {{font-size: 1.6rem;}}
</style>
"""


_saved = st.query_params.get("theme", DEFAULT_THEME)
if "theme" not in st.session_state:
    st.session_state["theme"] = _saved if _saved in THEMES else DEFAULT_THEME

head_l, head_m, head_r = st.columns([4, 1.7, 1.3])
with head_r:
    theme_name = st.selectbox("🎨 Theme", list(THEMES), key="theme")
with head_m:
    refresh_choice = st.selectbox("🔄 Auto-refresh", list(REFRESH_CHOICES), key="refresh_choice",
                                  help="How often the live tabs (Market, Lookup, Watchlist, Portfolio) re-download "
                                       "prices on their own. Each panel shows when its data was fetched.")
    st.session_state["refresh_secs"] = REFRESH_CHOICES[refresh_choice]
    if st.button("Refresh now", key="refresh_now", width="stretch"):
        MK.clear_live_caches()
        st.rerun()
st.query_params["theme"] = theme_name  # keeps your choice after a page refresh
st.session_state["palette"] = THEMES[theme_name]   # charts read this to match the theme
st.markdown(theme_css(THEMES[theme_name]), unsafe_allow_html=True)

with head_l:
    st.markdown(
        '<div class="hero"><h1>Cash-Secured Put Screener</h1>'
        "<p>Live yfinance data, fetched fresh on every scan (Yahoo quotes can still be ~15 min delayed). "
        "Not financial advice — verify pricing and liquidity with your broker before trading.</p></div>",
        unsafe_allow_html=True,
    )
st.write("")


# ---------------- Strategy presets ----------------

PRESETS = {
    "Daily (1-3 DTE)": dict(dte_range=(1, 3), win_range=(85, 97), min_oi=500, max_spread=10,
                            note="Very short trades. Small premiums, so only liquid names with tight spreads. "
                                 "Check every morning; gap risk overnight is your main enemy."),
    "Weekly (4-10 DTE)": dict(dte_range=(4, 10), win_range=(82, 95), min_oi=300, max_spread=12,
                              note="The sweet spot for most active sellers: fast time decay, "
                                   "manageable once or twice a week."),
    "Swing (11-25 DTE)": dict(dte_range=(11, 25), win_range=(80, 95), min_oi=300, max_spread=15,
                              note="More cushion and fewer trades. Good when you can't watch positions daily."),
    "Monthly (30-45 DTE)": dict(dte_range=(30, 45), win_range=(80, 95), min_oi=300, max_spread=15,
                                note="Classic 30-45 day income trade. Lowest effort; "
                                     "premium decays slowest, but you can close early at 50% profit."),
    "Custom": dict(note="Set every filter yourself."),
}
DEFAULT_STYLE = "Weekly (4-10 DTE)"


def apply_preset():
    preset = PRESETS.get(st.session_state.get("strategy"), {})
    for k, v in preset.items():
        if k != "note":
            st.session_state[k] = v


for _k, _v in PRESETS[DEFAULT_STYLE].items():
    if _k != "note":
        st.session_state.setdefault(_k, _v)


# ---------------- Optional password gate ----------------
# Set `app_password` in Streamlit secrets (Community Cloud: App settings -> Secrets) to require a login.
try:
    _APP_PASSWORD = st.secrets.get("app_password")
except Exception:
    _APP_PASSWORD = None

if _APP_PASSWORD and not st.session_state.get("authed"):
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Unlock", type="primary"):
            if hmac.compare_digest(pw.encode(), str(_APP_PASSWORD).encode()):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    st.stop()

ON_CLOUD = str(Path(__file__).resolve()).startswith("/mount/src")

(tab_market, tab_lookup, tab_watch, tab_portfolio, tab_income,
 tab_scan, tab_journal, tab_guide) = st.tabs(["Market", "Lookup", "Watchlist", "Portfolio", "Income",
                                              "Put Scanner", "Journal", "Guide"])

with tab_market:
    view_market.render()
with tab_lookup:
    view_lookup.render()
with tab_watch:
    view_watchlist.render()
with tab_portfolio:
    view_portfolio.render(on_cloud=ON_CLOUD, has_password=bool(_APP_PASSWORD))
with tab_income:
    view_income.render()

# =====================================================================
# SCANNER
# =====================================================================
with tab_scan:
    _open, _closed_msg = E.us_market_status()
    _has_scan = bool(st.session_state.get("scan"))
    if not _open and not _has_scan:
        st.warning(_closed_msg)
    with st.expander("⚙️ Scan settings", expanded=not _has_scan):
        st.radio("Trading style", list(PRESETS), index=list(PRESETS).index(DEFAULT_STYLE), horizontal=True,
                 key="strategy", on_change=apply_preset)
        st.caption(PRESETS[st.session_state.get("strategy", DEFAULT_STYLE)]["note"])

        with st.form("settings", border=True):
            t1, t2 = st.columns([3, 2])
            tickers_input = t1.text_input(
                "Extra tickers (optional)", "", placeholder="e.g. NVDA, SOXL, AAPL — leave blank to scan only the universes",
                help="Comma-separated, any number of symbols, added on top of the selected universes. "
                     "Duplicates are removed.",
            )
            chosen_universes = t2.multiselect(
                "Universes to scan", list(E.UNIVERSES) + [E.MARKET_UNIVERSE],
                default=["Most-liquid stocks (~80)", "Index & sector ETFs"],
                help="Adds every ticker in the list to the scan. Nasdaq-100 takes about a minute and the "
                     "full S&P 500 about 5 minutes (Yahoo throttles large scans, so big lists run gently "
                     "and retry automatically). 'Entire US options market' pre-filters every optionable US stock "
                     "by price, volume and your cash (instantly, no Yahoo calls), then scans the most liquid ones "
                     "first, up to the limit in 'Risk & ranking filters'. Results always come from live data.",
            )

            st.markdown("**Your account**")
            a = st.columns(5)
            capital = a[0].number_input("Cash available ($)", min_value=1000, value=25000, step=1000,
                                        help="Cash you can set aside to secure puts (strike × 100 per contract).")
            reserve_pct = a[1].number_input("Keep in reserve %", min_value=0, max_value=90, value=10, step=5,
                                            help="Cash you deliberately leave unused as a safety buffer.")
            max_pos_pct = a[2].number_input("Preferred max per position %", min_value=1, max_value=100, value=15, step=5,
                                            help="The plan sizes positions to stay within this share of your cash. "
                                                 "If the only affordable contract is bigger, one contract is still "
                                                 "allowed and flagged, as long as it fits your deployable cash.")
            max_positions = a[3].number_input("Max positions", min_value=1, max_value=30, value=8, step=1,
                                               help="The plan spreads your cash across up to this many different tickers.")
            max_lev_pct = a[4].number_input("Max leveraged-ETF %", min_value=0, max_value=100, value=15, step=5,
                                            help="Cap on the share of your cash in 2x/3x ETFs like SOXL/TQQQ.")

            st.markdown("**Strategy filters**")
            f = st.columns(4)
            min_dte, max_dte = f[0].slider("Days to expiration", 1, 90, key="dte_range")
            win_prob_min, win_prob_max = f[1].slider("Win probability %", 50, 99, key="win_range")
            min_oi = f[2].number_input(
                "Min open interest", min_value=0, step=5, key="min_oi",
                help="Falls back to volume when Yahoo reports open interest as 0/blank for a contract.",
            )
            max_spread_pct = f[3].number_input("Max spread %", min_value=1, step=1, key="max_spread")

            st.markdown("**Risk & ranking filters**")
            g = st.columns(4)
            min_hist_win = g[0].number_input(
                "Min historical win %", min_value=0, max_value=100, value=70, step=5,
                help="The stock must have finished above this strike in at least this share of past windows "
                     "of the same length (last ~3 years). 0 turns it off.",
            )
            min_premium_usd = g[1].number_input(
                "Min premium per contract ($)", min_value=0, value=10, step=5,
                help="Ignore trades that pay less than this — commissions and slippage eat tiny premiums.",
            )
            min_otm_pct = g[2].number_input(
                "Min % below spot", min_value=0.0, max_value=50.0, value=0.0, step=1.0,
                help="Require the strike to be at least this far below the current price.",
            )
            top_n = g[3].number_input("Top N per ticker", min_value=1, max_value=10, value=3)
            h = st.columns(5)
            skip_downtrend = h[0].checkbox("Skip stocks in a downtrend", value=False,
                                           help="Skips stocks trading below their 50-day average, which is itself "
                                                "below the 200-day average. Off by default: downtrend stocks are shown and labeled "
                                                "in the Trend column.")
            below_sma = h[1].checkbox("Strike below 50-day SMA", value=False)
            exclude_earnings = h[2].checkbox("Exclude earnings in window", value=True,
                                             help="Earnings gaps are the #1 way put sellers get hurt.")
            require_edge = h[3].checkbox("Require positive edge", value=False,
                                         help="Only show puts that pay more than their fair value at the stock's "
                                              "recent real volatility. Stricter; often leaves fewer trades.")
            rank_label = h[4].selectbox("Rank candidates by", list(E.RANK_OPTIONS))
            st.markdown("**Entire-market universe** (only used when it's selected above)")
            u = st.columns(3)
            mk_min_price = u[0].number_input("Min stock price ($)", min_value=0.0, value=5.0, step=1.0)
            mk_min_volume = u[1].number_input("Min daily share volume", min_value=0, value=1_000_000, step=250_000,
                                              help="Yesterday's share volume. Filters out thinly traded stocks whose "
                                                   "options are hard to trade.")
            mk_max_n = u[2].number_input("Max tickers to scan", min_value=50, max_value=2000, value=400, step=50,
                                         help="Most liquid first. Roughly 0.6 seconds per ticker: 400 ≈ 4 min, "
                                              "1,500 ≈ 15 min.")

            h2 = st.columns(4)
            allow_stale = h2[0].checkbox(
                "Use last-trade prices when the market is closed", value=not _open,
                help="After hours Yahoo shows no live bids, so scans find nothing. Turn this on to plan for the next "
                     "session using each option's last traded price. Those rows are labeled and may be hours or "
                     "days old: always re-check live prices before trading.")

            run_button = st.form_submit_button("Run scan", type="primary", width="stretch")

    tickers = [t.strip().upper() for t in tickers_input.replace("\n", ",").split(",") if t.strip()]
    for name in chosen_universes:
        if name in E.UNIVERSES:
            tickers += E.UNIVERSES[name]()
    tickers = list(dict.fromkeys(tickers))

    if run_button:
        rf_rate = E.get_risk_free_rate()
        max_contract_cost = capital * (1 - reserve_pct / 100)
        market_info = None
        if E.MARKET_UNIVERSE in chosen_universes:
            market_tickers, market_info = E.build_market_universe(max_contract_cost, mk_min_price,
                                                                  mk_min_volume, mk_max_n)
            tickers = list(dict.fromkeys(tickers + market_tickers))
            if market_info["error"]:
                st.warning(market_info["error"])
        if not tickers:
            st.warning("Pick at least one universe or type a ticker.")
        else:
            params = E.ScanParams(
                risk_free_rate=rf_rate, min_dte=min_dte, max_dte=max_dte,
                win_prob_min=win_prob_min, win_prob_max=win_prob_max, min_oi=min_oi,
                max_spread_pct=max_spread_pct, top_n=top_n, min_otm_pct=min_otm_pct,
                min_hist_win=min_hist_win, min_premium_usd=min_premium_usd,
                max_contract_cost=max_contract_cost, exclude_earnings=exclude_earnings,
                skip_downtrend=skip_downtrend, below_sma=below_sma, require_edge=require_edge, allow_stale=allow_stale,
                rank_by=rank_label,
            )
            results, notes_map, near_miss, relaxed_used = {}, {}, [], False

            if len(tickers) > E.MAX_TICKERS:
                st.warning(f"Scanning the first {E.MAX_TICKERS} of {len(tickers)} tickers (safety cap).")
                tickers = tickers[:E.MAX_TICKERS]

            eta_min = max(1, round(len(tickers) * (0.25 if len(tickers) <= 120 else 0.6) / 60))
            with st.status(f"Scanning {len(tickers)} ticker(s) — about {eta_min} min…", expanded=False) as status:
                def run_pass(batch, workers, label, prm=params, relaxed=False):
                    """Scan `batch`; return tickers whose price fetch failed (likely rate-limited)."""
                    failed = []
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {pool.submit(E.scan_ticker_for_csp, t, prm): t for t in batch}
                        for i, fut in enumerate(as_completed(futures), start=1):
                            t = futures[fut]
                            try:
                                df, spot, counts = fut.result()
                            except Exception:
                                failed.append(t)
                                continue
                            if not df.empty:
                                df["Relaxed"] = "Relaxed filters" if relaxed else ""
                                results[t] = df
                            elif spot is None:
                                failed.append(t)
                            else:
                                if not relaxed and not counts.get("downtrend_skip") and counts.get("affordable", 0) > 0:
                                    near_miss.append(t)
                                notes_map[t] = E.empty_reason(t, spot, counts, prm)
                            status.update(label=f"{label}: {i}/{len(batch)} — last: {t} "
                                                f"({len(results)} with candidates)")
                    return failed

                # Big scans get throttled by Yahoo, so use fewer parallel workers for them.
                workers = E.MAX_WORKERS if len(tickers) <= 120 else 3
                failed = run_pass(tickers, workers, "Scanned")
                for attempt, cooldown in enumerate((20, 40), start=1):
                    if not failed:
                        break
                    status.update(label=f"Yahoo is throttling — cooling down {cooldown}s "
                                        f"before retry {attempt} ({len(failed)} ticker(s))")
                    time.sleep(cooldown)
                    failed = run_pass(failed, 1 if attempt == 2 else 2, f"Retry {attempt}")

                # Strict filters found (almost) nothing: automatically try looser liquidity/premium filters.
                if sum(len(d) for d in results.values()) < 3 and near_miss:
                    status.update(label=f"Few results — trying relaxed filters on {len(near_miss)} ticker(s)…")
                    run_pass(near_miss, workers, "Relaxed", E.relax_params(params), relaxed=True)
                    relaxed_used = any((d["Relaxed"] != "").any() for d in results.values())

                notes = [v for t, v in notes_map.items() if t not in results]
                notes += [f"{t}: could not fetch price/options data (Yahoo may be rate-limiting; try again shortly)."
                          for t in failed]
                status.update(label="Checking market conditions…")
                regime = E.get_market_regime()
                status.update(
                    label=f"Scan complete — {len(results)} of {len(tickers)} ticker(s) with candidates",
                    state="complete")

            st.session_state["scan"] = {
                "results": results, "notes": notes, "order": tickers, "regime": regime,
                "rf_rate": rf_rate, "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rank_by": rank_label, "max_contract_cost": max_contract_cost, "relaxed_used": relaxed_used,
                "market_info": market_info, "summary_label": f"{len(results)} of {len(tickers)} ticker(s) had candidates",
                "acct": dict(capital=capital, reserve_pct=reserve_pct, max_pos_pct=max_pos_pct,
                             max_positions=max_positions, max_lev_pct=max_lev_pct),
            }
            st.rerun()

    scan_state = st.session_state.get("scan")

    if not scan_state:
        with st.container(border=True):
            st.subheader("Ready to scan")
            st.markdown(
                "1. Pick your **trading style** and enter the **cash** you want to use.\n"
                "2. Choose which universes to scan (and any extra tickers), then press **Run scan**.\n"
                "3. Open the **Trade plan** tab for a ready-to-place basket sized to your cash.\n"
                "4. Log the trades you take in the **Journal** tab. The **Guide** explains every term."
            )
    else:
        acct = scan_state["acct"]
        regime = scan_state["regime"]
        rank_col = E.RANK_OPTIONS[scan_state["rank_by"]]

        st.caption(f"✅ {scan_state.get('summary_label', 'Scan complete')} · fetched {scan_state['fetched_at']} · 13-week T-bill (risk-free rate): "
                   f"{scan_state['rf_rate'] * 100:.2f}% · ranked by: {scan_state['rank_by']} · "
                   f"max cash per contract: ${scan_state['max_contract_cost']:,.0f}".replace("$", "\\$"))

        if regime["vix"] is not None:
            spy_txt = f" · S&P 500: {regime['spy_trend']}" if regime["spy_trend"] else ""
            banner = f"**VIX {regime['vix']:.1f} ({regime['level']}){spy_txt}.** {regime['advice']}"
            {"calm": st.info, "normal": st.success, "elevated": st.warning, "high": st.error}.get(
                regime["level"], st.info)(banner)

        mi = scan_state.get("market_info")
        if mi and not mi["error"]:
            st.info(f"Market universe: {mi['optionable']:,} optionable symbols → {mi['priced']:,} stocks with prices → "
                    f"{mi['passed']:,} pass your price/volume/cash filters → scanned the {mi['scanning']:,} most "
                    f"liquid ones (raise 'Max tickers to scan' to cover more).")

        if any((d.get("Quote", "") != "").any() for d in scan_state["results"].values()):
            st.warning("Some prices are **last-trade prices from when the market was closed** (marked in the Price "
                       "source column). Use these to plan, then re-check live bids and asks when the market opens.")

        if scan_state.get("relaxed_used"):
            st.warning("Few trades passed your strict filters, so I also ran looser ones (" + E.RELAXED_NOTE +
                       "). Those rows are marked **Relaxed filters**. Liquidity is thinner there: use limit orders "
                       "and smaller size.")

        if not scan_state["results"]:
            st.error("No candidates found for any ticker. Try widening a filter or raising the max per position.")
            with st.expander(f"Why ({len(scan_state['notes'])} notes)", expanded=len(scan_state["notes"]) <= 5):
                md("\n".join(f"- {n}" for n in scan_state["notes"]))
            st.stop()

        summary = pd.concat(scan_state["results"].values(), ignore_index=True)
        summary = summary.sort_values(rank_col, ascending=False, na_position="last").reset_index(drop=True)
        CONTRACT_CHOICES = {
            "1 contract per ticker (more different tickers)": 1,
            "Up to 2 contracts per ticker": 2,
            "Up to 3 contracts per ticker": 3,
            "Up to 5 contracts per ticker": 5,
            "As many as my cash allows": None,
        }
        cc1, _ = st.columns([2, 3])
        contracts_label = cc1.selectbox(
            "Contracts per ticker", list(CONTRACT_CHOICES), key="contracts_choice",
            help="One contract per ticker spreads your cash over more different stocks, and you can always add "
                 "contracts at the same strike yourself. Pick a higher number to size up each position; the plan "
                 "updates instantly (no rescan).")
        plan, pstats = E.build_plan(summary, acct["capital"], acct["reserve_pct"], acct["max_pos_pct"],
                                    acct["max_positions"], acct["max_lev_pct"], rank_col,
                                    CONTRACT_CHOICES[contracts_label])

        display_cols = ["Relaxed", "Quote", "Expiration", "DTE", "Strike", "% OTM", "Distance (σ)", "Premium (Bid)", "Mid",
                        "Premium $", "Delta", "Est. Win Prob %", "Hist. Win %", "Win % (cons.)", "Edge $",
                        "Annualized Yield %", "Return/Day %", "Score", "Lev", "IV/HV", "Trend", "RSI",
                        "P&L @ -10% $", "P&L @ -20% $", "Exit Target ($)", "Next Earnings",
                        "Earnings Alert", "Ex-Div Alert", "Breakeven", "Capital Req. $",
                        "Open Interest", "Spread %"]
        column_config = {
            "Est. Win Prob %": st.column_config.NumberColumn("Model win", format="%.1f%%"),
            "Hist. Win %": st.column_config.NumberColumn("Hist. win", format="%.1f%%"),
            "Win % (cons.)": st.column_config.ProgressColumn(
                "Win (cons.)", format="%.1f%%", min_value=50, max_value=100),
            "Score": st.column_config.ProgressColumn(
                "Score", format="%.1f", min_value=0, max_value=float(max(summary["Score"].max(), 1))),
            "Edge $": st.column_config.NumberColumn("Edge ($)", format="%d"),
            "Annualized Yield %": st.column_config.NumberColumn("Ann. yield", format="%.1f%%"),
            "Return/Day %": st.column_config.NumberColumn("Per day", format="%.2f%%"),
            "% OTM": st.column_config.NumberColumn("% OTM", format="%.1f%%"),
            "Distance (σ)": st.column_config.NumberColumn("Dist (σ)", format="%.2f"),
            "Spread %": st.column_config.NumberColumn("Spread", format="%.1f%%"),
            "Strike": st.column_config.NumberColumn("Strike", format="$%g"),
            "Premium (Bid)": st.column_config.NumberColumn("Bid", format="$%.2f"),
            "Mid": st.column_config.NumberColumn("Mid", format="$%.2f"),
            "Breakeven": st.column_config.NumberColumn("Breakeven", format="$%.2f"),
            "Capital Req. $": st.column_config.NumberColumn("Cash needed", format="$%d"),
            "Premium $": st.column_config.NumberColumn("Premium", format="$%d"),
            "P&L @ -10% $": st.column_config.NumberColumn("P&L @ -10% ($)", format="%d"),
            "P&L @ -20% $": st.column_config.NumberColumn("P&L @ -20% ($)", format="%d"),
            "Exit Target ($)": st.column_config.TextColumn("Close at"),
            "Earnings Alert": st.column_config.TextColumn("Earnings"),
            "Ex-Div Alert": st.column_config.TextColumn("Ex-div"),
            "Open Interest": st.column_config.NumberColumn("OI", format="%d"),
            "Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
            "IV/HV": st.column_config.NumberColumn("IV/HV", format="%.2f"),
            "RSI": st.column_config.NumberColumn("RSI", format="%d"),
            "Contracts": st.column_config.NumberColumn("Contracts", format="%d"),
            "Relaxed": st.column_config.TextColumn("Filters"),
            "Quote": st.column_config.TextColumn("Price source"),
            "% of cash": st.column_config.NumberColumn("% of cash", format="%.0f%%"),
            "Sell limit (mid)": st.column_config.NumberColumn("Sell limit (mid)", format="$%.2f"),
            "Cash secured $": st.column_config.NumberColumn("Cash secured", format="$%d"),
        }

        def highlight_earnings(row):
            color = "background-color: rgba(255, 75, 75, 0.10)" if row.get("Earnings Alert") else ""
            return [color] * len(row)

        def show_table(df, cols):
            st.dataframe(
                df[cols].style.apply(highlight_earnings, axis=1),
                column_config=column_config, width="stretch", hide_index=True,
            )

        t_plan, t_all, t_ticker, t_chart = st.tabs(
            ["Trade plan", "All candidates", "By ticker", "Risk vs reward"])

        # ---------- Trade plan ----------
        with t_plan:
            if plan.empty:
                st.warning(
                    f"No position fits your rules. Your limit is ${scan_state['max_contract_cost']:,.0f} of cash "
                    f"per contract. Try more cash, a lower reserve, or cheaper tickers."
                    .replace("$", "\\$")
                )
            else:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Cash deployed", money(pstats['deployed']))
                k1.caption(f"{pstats['deployed_pct']:.0f}% of your cash · {pstats['positions']} positions")
                k2.metric("Premium collected", money(pstats['premium']))
                k2.caption(f"{pstats['return_pct']:.2f}% on deployed cash, if all expire worthless")
                k3.metric("If everything drops 20%", money(pstats['worst_20']))
                k3.caption(f"{pstats['worst_20'] / acct['capital'] * 100:.1f}% of your cash")
                k4.metric("Cash left over", money(pstats['cash_left']))
                k4.caption("reserve + unallocated")

                show_table(plan, ["Ticker", "Expiration", "DTE", "Strike", "Contracts", "Sell limit (mid)",
                                  "Premium $", "Cash secured $", "Win % (cons.)", "Edge $",
                                  "P&L @ -10% $", "P&L @ -20% $", "% of cash", "Breakeven", "Trend", "Earnings Alert",
                                  "Relaxed", "Quote"])

                neg_edge = plan[plan["Edge $"] < 0]
                if not neg_edge.empty:
                    st.info("Negative edge means the premium is below fair value at the stock's recent real "
                            "volatility (options look cheap for the risk): " + ", ".join(neg_edge["Ticker"]) +
                            ". Turn on 'Require positive edge' or rank by Edge $ to prefer richer premiums.")

                flagged_plan = plan[plan["Earnings Alert"] != ""]
                if not flagged_plan.empty:
                    st.warning("Earnings fall inside the window for: " + ", ".join(flagged_plan["Ticker"]) +
                               ". Consider skipping those, or turn on 'Exclude earnings in window'.")
                down = plan[plan["Trend"] == "Downtrend"]
                if not down.empty:
                    st.info("In a downtrend (price below its 50- and 200-day averages): " + ", ".join(down["Ticker"]) +
                            ". They can keep falling, so prefer strikes well below the price and smaller size.")

                big = plan[plan["Size flag"] != ""]
                if not big.empty:
                    st.warning("Above your preferred per-position max: " +
                               ", ".join(f"{r.Ticker} ({r['% of cash']:.0f}% of cash)" for _, r in big.iterrows()) +
                               ". Consider a smaller account share or cheaper names.")
                lev_rows = plan[plan["Lev"] > 1]
                if not lev_rows.empty:
                    st.warning("Leveraged ETFs in the plan (" + ", ".join(lev_rows["Ticker"]) +
                               ") can fall 30%+ in days. Keep them small.")

                st.caption(f"The plan has {pstats['positions']} position(s) because {pstats['stop_reason']}.")
                others = pstats["others"]
                if not others.empty:
                    with st.expander(f"Next-best candidates that didn't fit the plan ({len(others)} tickers)"):
                        show_table(others.head(25), ["Ticker", "Expiration", "DTE", "Strike", "Premium $",
                                                     "Win % (cons.)", "Edge $", "Capital Req. $", "Trend",
                                                     "Earnings Alert", "Relaxed", "Quote"])

                md("- **Sell limit (mid)** is the price to enter as a limit order; the premium shown assumes the "
                   "lower **bid**, so it is conservative.\n"
                   "- One position per ticker, best-ranked first, spread across your deployable cash, up to your max positions, using the contracts-per-ticker setting above.\n"
                   "- Premium, edge and P&L columns are totals for all contracts in the row.")

                if st.button("Add this plan to my journal", type="primary"):
                    n_added = J.add_trades(plan)
                    st.success(f"Added {n_added} trade(s) to your journal." if n_added
                               else "These trades were already logged today.")

        # ---------- All candidates ----------
        with t_all:
            show_table(summary, ["Ticker"] + display_cols)
            st.download_button(
                "Download CSV", summary.to_csv(index=False).encode("utf-8"),
                file_name=f"csp_candidates_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv",
            )

        # ---------- By ticker ----------
        with t_ticker:
            for ticker in scan_state["order"]:
                sub = summary[summary["Ticker"] == ticker]
                if sub.empty:
                    continue
                spot, sma, low = sub["Spot Price"].iloc[0], sub["50D SMA"].iloc[0], sub["52W Low"].iloc[0]
                rsi_val = sub["RSI"].iloc[0]
                with st.container(border=True):
                    h1, h2, h3, h4, h5 = st.columns([1.2, 1, 1, 1, 1])
                    h1.markdown(f"### {ticker}")
                    h2.metric("Spot", f"${spot:,.2f}")
                    h3.metric("vs 50D SMA", f"${sma:,.2f}", f"{(spot / sma - 1) * 100:+.1f}%")
                    h4.metric("vs 52W low", f"${low:,.2f}", f"{(spot / low - 1) * 100:+.1f}%")
                    h5.metric("Trend", sub["Trend"].iloc[0],
                              f"RSI {rsi_val:.0f}" if pd.notna(rsi_val) else None, delta_color="off")
                    show_table(sub, display_cols)

        # ---------- Chart ----------
        with t_chart:
            st.caption("Each dot is a candidate. Up and to the right is better: higher conservative win probability "
                       "and higher yield. Hover for details.")
            scatter(summary, "Win % (cons.)", "Annualized Yield %", "Ticker", size="Capital Req. $", height=420)

        if scan_state["notes"]:
            with st.expander(f"Scan notes ({len(scan_state['notes'])})"):
                md("\n".join(f"- {n}" for n in scan_state["notes"]))

# =====================================================================
# JOURNAL
# =====================================================================
with tab_journal:
    st.subheader("Trade journal")
    st.caption(f"Stored locally in {J.JOURNAL_PATH.name}. Log trades from the Trade plan tab, or add rows below. "
               "Update a row's status and buy-back price when you close it.")

    if ON_CLOUD:
        st.warning("This app is hosted online, where saved files can be wiped whenever the app restarts or is "
                   "redeployed. Download a backup of your journal regularly (button below) and restore it here "
                   "if it ever comes back empty.")

    jdf = J.load_journal()
    stats = J.journal_stats(jdf)
    if stats:
        m = st.columns(5)
        m[0].metric("Realized P&L", money(stats['total_pnl']))
        m[1].metric("Closed trades", stats["trades"])
        m[2].metric("Your win rate", f"{stats['win_rate']:.0f}%" if pd.notna(stats["win_rate"]) else "—")
        m[3].metric("Model's expected", f"{stats['avg_est_win']:.0f}%" if pd.notna(stats["avg_est_win"]) else "—")
        m[4].metric("Assigned", stats["assigned"])
        if pd.notna(stats["win_rate"]) and pd.notna(stats["avg_est_win"]) and stats["trades"] >= 20:
            gap = stats["win_rate"] - stats["avg_est_win"]
            if gap < -5:
                st.warning(f"Your real win rate is {abs(gap):.0f} points below what the tool predicted. "
                           "Tighten the filters (higher win %, lower delta, skip earnings) or trade smaller.")
            else:
                st.success("Your results are in line with the tool's predictions so far.")

    st.markdown("**Open positions — live check**")
    if st.button("Refresh live P&L and suggested actions"):
        with st.spinner("Fetching live prices…"):
            st.session_state["mtm"] = J.mark_to_market(jdf)
    mtm = st.session_state.get("mtm")
    if mtm is not None and not mtm.empty:
        st.dataframe(mtm, hide_index=True, width="stretch", column_config={
            "Action": st.column_config.TextColumn("Action", width="large"),
            "Strike": st.column_config.NumberColumn(format="$%g"),
            "Sold at": st.column_config.NumberColumn(format="$%.2f"),
            "Spot": st.column_config.NumberColumn(format="$%.2f"),
            "Put now": st.column_config.NumberColumn(format="$%.2f"),
            "Unrealized P&L $": st.column_config.NumberColumn("Unrealized P&L ($)", format="%d"),
            "% of premium captured": st.column_config.ProgressColumn(
                "Premium captured", format="%d%%", min_value=0, max_value=100),
            "% above strike": st.column_config.NumberColumn(format="%.1f%%"),
        })
    elif mtm is not None:
        st.info("No open positions to check.")

    st.markdown("**All trades** (edit cells, add or delete rows, then press Save)")
    edited = st.data_editor(
        jdf, num_rows="dynamic", width="stretch", hide_index=True, key="journal_editor",
        column_config={
            "id": st.column_config.NumberColumn("id", disabled=True),
            "opened": st.column_config.TextColumn("Opened", help="YYYY-MM-DD"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "expiration": st.column_config.TextColumn("Expiration", help="YYYY-MM-DD"),
            "strike": st.column_config.NumberColumn("Strike", format="$%g"),
            "contracts": st.column_config.NumberColumn("Contracts", format="%d"),
            "premium": st.column_config.NumberColumn("Sold at (per share)", format="$%.2f"),
            "est_win_pct": st.column_config.NumberColumn("Est. win %", format="%.1f"),
            "status": st.column_config.SelectboxColumn("Status", options=J.STATUSES),
            "close_price": st.column_config.NumberColumn(
                "Bought back at", format="$%.2f",
                help="Price per share you paid to close. Leave 0 or blank if it expired worthless."),
            "closed": st.column_config.TextColumn("Closed", help="YYYY-MM-DD"),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )
    if st.button("Save journal", type="primary"):
        J.save_journal(edited)
        st.session_state.pop("mtm", None)
        st.success("Journal saved.")
        st.rerun()

    with st.expander("Backup & restore"):
        st.download_button("Download journal backup (CSV)", jdf.to_csv(index=False).encode("utf-8"),
                           file_name=f"trade_journal_{datetime.now():%Y%m%d}.csv", mime="text/csv")
        uploaded = st.file_uploader("Restore from a backup CSV (replaces the current journal)", type="csv")
        if uploaded is not None and st.button("Restore this backup"):
            try:
                restored = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                missing = set(J.COLUMNS) - set(restored.columns)
                if missing:
                    st.error(f"That file is missing columns: {', '.join(sorted(missing))}")
                else:
                    J.save_journal(restored[J.COLUMNS])
                    st.session_state.pop("mtm", None)
                    st.success("Journal restored.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read that file: {e}")

    if stats:
        pnl = J.realized_pnl(jdf)
        closed_view = jdf.assign(**{"P&L $": pnl})[pnl.notna()]
        st.markdown("**Realized P&L by trade**")
        st.dataframe(closed_view[["ticker", "expiration", "strike", "contracts", "premium", "close_price",
                                  "status", "P&L $"]], hide_index=True, width="stretch",
                     column_config={"P&L $": st.column_config.NumberColumn("P&L ($)", format="%d")})

# =====================================================================
# GUIDE
# =====================================================================
with tab_guide:
    st.subheader("How to use this tool")
    with st.expander("Tour of the tabs", expanded=True):
        md("""
- **Market:** the day's big picture. Index levels (S&P 500, Nasdaq, Dow, Russell 2000), the VIX fear gauge, Treasury yields, gold, oil, bitcoin and the dollar, plus how each sector is doing and the biggest movers among ~80 heavily traded stocks.
- **Lookup:** search any company or ticker. You get the live price, a chart with 50- and 200-day averages, key stats (P/E, market cap, beta, dividend yield, analyst targets), dividend history, news, a side-by-side compare, and a one-click list of cash-secured put ideas for that stock.
- **Watchlist:** the stocks you follow, with price, daily/5-day/1-month change, where each sits in its 52-week range, analyst upside and yield. Set **alert prices**: when a price crosses one, a banner and pop-up appear.
- **Portfolio (private):** enter your holdings (or import your broker's CSV) to see live value, today's change, gain/loss, allocation by stock and sector, dividends you will receive, a one-year comparison against the S&P 500, upcoming earnings and ex-dividend dates, and plain-language warnings about concentration. A **Hide amounts** switch blurs the numbers if someone is looking at your screen.
- **Income:** the **Dividend calculator** shows what a position pays per year, month, week and day, and what you need to invest for a target monthly income. The **DRIP calculator** compares reinvesting dividends against taking them as cash, with monthly contributions, dividend tax, and a table of "what if the price grows faster or slower". Load any real stock to fill in its actual price, dividend, payout schedule and growth.
- **Put Scanner / Journal:** the cash-secured put tools described below.
- **Refreshing:** every live panel shows a **🕒 Updated** time that is when the data was really downloaded. Use **Refresh now** in the header for fresh data, or switch on **Auto-refresh** (30 seconds to 15 minutes). Prices are cached for about a minute so repeated clicks don't hammer Yahoo.
""")

    with st.expander("Stocks, dividends & portfolio terms"):
        md("""
- **Ticker:** the short code for a stock or fund (AAPL for Apple).
- **Market cap:** the company's total value (share price x number of shares). Above ~$10B is "large cap".
- **P/E ratio:** price divided by yearly earnings per share. High means investors pay a lot per dollar of profit. *Trailing* uses past earnings, *forward* uses expected earnings.
- **Beta:** how much a stock tends to move compared with the market. 1.0 moves like the market; 1.5 swings 50% more; 0.5 swings half as much.
- **52-week range:** the lowest and highest price over the past year. "Near the low" can mean cheap or can mean in trouble.
- **Analyst target:** the average price Wall Street analysts expect in about a year. Treat it as one opinion, not a promise.
- **ETF:** a fund that trades like a stock and holds many investments (SPY holds the S&P 500). **Expense ratio** is its yearly fee.
- **Dividend:** a cash payment a company makes to its shareholders, usually every quarter.
- **Dividend yield:** yearly dividend divided by the share price. A $100 stock paying $3 a year yields 3%.
- **Yield on cost:** yearly dividend divided by what you originally paid. It rises over time if the company keeps raising the dividend.
- **Payout ratio:** the share of profits paid out as dividends. Above ~80% can mean the dividend is hard to sustain.
- **Ex-dividend date:** you must own the stock before this date to receive the next dividend.
- **Dividend growth / CAGR:** the average yearly rate the dividend has grown. **Increase streak** counts consecutive years of raises.
- **DRIP (dividend reinvestment plan):** automatically using each dividend to buy more shares, which then pay their own dividends. This compounding is why the DRIP column grows faster than taking cash.
- **Compounding:** growth on top of earlier growth. Small differences in yield or growth add up a lot over 20-30 years.
- **Cost basis / average cost:** the average price you paid per share. Gain = (current price - average cost) x shares.
- **Weight:** the share of your portfolio each holding makes up. **Concentration** means too much in one stock or sector.
- **Sector:** the industry group a company belongs to (Technology, Energy, Utilities...). Spreading across sectors reduces the damage from a bad year in one.
- **Portfolio beta:** your holdings' betas averaged by weight: how jumpy the whole portfolio is compared with the market.
- **Treasury yield (10-year):** what the US government pays to borrow for 10 years. Rising yields can pressure stock prices and boost bond appeal.
- **Sector ETFs (XLK, XLF...):** funds that track one sector. The Market tab uses them to show which sectors are leading or lagging.
""")

    with st.expander("Your daily routine (about 5 minutes)", expanded=True):
        md("""
1. **Check your open trades first.** Journal tab → *Refresh live P&L*. Close anything marked *Close now* (70%+ of the premium captured) and think about anything *At risk* or *ITM*.
2. **Pick your style.** Weekly suits most people; Daily only if you can watch the market; Monthly if you want low effort.
3. **Enter your cash** and sizing rules. The tool never suggests a trade that needs more cash than you have available.
4. **Choose tickers.** Start with *Most-liquid stocks* and *Index & sector ETFs*. Add the Nasdaq-100 or S&P 500 when you want a wider net.
5. **Run scan and read the banner.** VIX and the S&P trend tell you how aggressive to be today. High fear means smaller size.
6. **Open the Trade plan tab.** It builds a basket sized to your cash: one position per ticker, best-ranked first.
7. **Sanity-check each trade** on your broker: live bid/ask, news, earnings date. Enter a **limit order at the mid price** (the plan shows it) and be patient.
8. **Log what you actually fill** (Add plan to journal, then edit the premium to your real fill).
9. **Manage, don't just wait.** Buy back at 50-70% of max profit, or roll when a stock threatens your strike.
""")

    with st.expander("How the tool picks and sizes trades"):
        md("""
- **Filters first.** Only puts with a live bid and ask, a sane implied volatility, enough open interest (or volume), and a tight spread survive. Expirations with earnings inside the window are skipped by default. Stocks in a downtrend are shown and labeled in the *Trend* column (you can opt to skip them).
- **Sizing.** The cash a put needs is `strike × 100`. Any put that fits your deployable cash (cash minus reserve) is eligible. The plan gives each ticker the number of contracts you choose in **Contracts per ticker** (default 1, so your cash is spread over as many different stocks as possible). Positions that fit an equal share of your cash go first; a single contract larger than that share is added last, cheapest first, and flagged if it exceeds your *Preferred max per position %*.
- **Win probability** is the model chance the stock finishes above your strike. **Historical win %** is how often the stock actually did that in past windows of the same length. The tool uses the **lower** of the two (*Win % (cons.)*), so it never trusts an optimistic number.
- **Ranking.** Default *Score* = conservative win % × annualized yield ÷ leverage. For daily and weekly trades, try ranking by **Edge $** or **Return per day**, because annualized yield exaggerates very short trades.
- **Trade plan.** Takes the best-ranked candidates in order, one per ticker, sized to fit your per-position cap, your reserve, your max positions and your leveraged-ETF cap.
""")

    with st.expander("Rules that protect your account"):
        md("""
- **Only sell puts on stocks you'd be happy to own** at the strike price. Assignment will happen sooner or later.
- **Never put all your cash to work.** Keep a reserve; the default is 10%.
- **Keep any single position small** (10-20% of cash at most) and avoid stacking correlated names (for example NVDA, AMD, SOXL and SMH are really one bet).
- **Skip earnings** unless you deliberately want that gamble.
- **Treat 2x/3x ETFs (SOXL, TQQQ...) as a small side bet.** They can lose a third of their value in days.
- **Look at the crash columns.** *P&L @ -20%* is what a bad week would cost. If you can't stomach it, cut size.
- **A high yield is a warning as much as an opportunity.** The market pays more premium because it sees more risk.
- **Close winners early.** Taking 50-70% of the premium and redeploying beats waiting for the last few cents.
- **Track everything in the Journal.** After 30-50 trades the win-rate comparison tells you whether the model's numbers hold up for *your* trading.
""")

    with st.expander("Glossary — what every term means", expanded=True):
        md("""
**The trade**
- **Cash-secured put (CSP):** you sell a put option and set aside cash to buy 100 shares at the strike if assigned. You collect a payment (premium) now for taking that obligation.
- **Premium:** the money you receive for selling the put. *Premium $* is per contract (100 shares).
- **Strike:** the price at which you'd have to buy the shares.
- **Expiration / DTE:** the date the option ends, and the days left until then.
- **OTM / ITM:** out of the money means the stock is above your strike (good for you); in the money means it's below.
- **Assignment:** being made to buy the shares at the strike. You then own the stock at your breakeven cost.
- **Breakeven:** strike minus premium. Below this, the trade loses money at expiry.
- **Contracts:** one contract covers 100 shares. *Cash secured $* = strike × 100 × contracts.
- **Roll:** closing your put and opening a new one further out in time (and often lower) to avoid assignment or collect more premium.
- **Wheel:** if assigned, sell covered calls on the shares until they are called away, then sell puts again.

**Prices and liquidity**
- **Bid / Ask / Mid:** bid is what buyers pay (what you sell at instantly), ask is what sellers want, mid is halfway. Put your order at the mid and wait for a fill.
- **Spread %:** gap between bid and ask as a share of the ask. Wide spreads mean you lose money just entering and leaving.
- **Open interest (OI):** contracts currently open. **Volume:** contracts traded today. Both show whether you can get out easily. The tool uses volume when Yahoo reports OI as 0.

**Probabilities and value**
- **Delta:** roughly the market's chance the put finishes in the money, and how fast its price moves with the stock. -0.15 means about a 15% chance of assignment.
- **Model win (Est. Win Prob %):** the option-pricing model's chance the put expires worthless, so you keep the whole premium.
- **Hist. win %:** how often the stock really finished above this strike after the same number of days over roughly the last 3 years.
- **Win % (cons.):** the lower of the two. This is the number to trust.
- **IV (implied volatility):** how much movement the options market expects. **HV (historical volatility):** how much the stock actually moved over the last 30 days.
- **IV/HV:** above about 1.2 means the market is paying you more than recent movement justifies, which is favorable to sellers.
- **Edge $:** premium minus the option's fair value at the stock's recent real volatility, per contract. Positive means you're overpaid for the risk.
- **Distance (σ):** how many expected moves the strike is below the price. Above 1 means the market thinks landing there is unlikely.

**Returns**
- **Yield %:** premium divided by the cash tied up (strike minus premium).
- **Return/Day %:** yield divided by days to expiry.
- **Annualized yield:** yield scaled to a year. Useful for comparison, but it exaggerates very short trades and assumes you can repeat them.
- **Score:** conservative win % × annualized yield ÷ leverage. A shortlist tool, not a verdict.
- **Lev:** leverage multiplier of the ETF (3 for SOXL/TQQQ). Score is divided by it so leveraged funds don't look artificially good.
- **Exit target / Close at:** the buy-back price range that locks in 50-70% of your maximum profit.
- **% of premium captured:** how much of the premium you've already earned on an open trade (Journal).

**Risk**
- **P&L @ -10% / -20%:** profit or loss at expiry if the stock drops that much from today, for one contract (totals in the plan).
- **Trend:** *Uptrend* = price above the 50-day average, which is above the 200-day. *Downtrend* = the reverse. *Mixed* = neither.
- **SMA:** simple moving average of the closing price over N days.
- **RSI:** momentum from 0 to 100. Below about 30 is oversold (may bounce, may keep falling); above 70 is overbought.
- **VIX:** the market's fear index. Under 15 calm, 15-20 normal, 20-30 elevated, above 30 high fear.
- **Earnings / Ex-div alert:** a company report or dividend date falls before the option expires. Earnings can gap the stock sharply.
- **Risk-free rate:** the 13-week Treasury bill yield, used in the option-pricing math.
""")

    with st.expander("Limits — read this"):
        md("""
- **Yahoo data is delayed and imperfect** (about 15 minutes; open interest and implied volatility can be stale or missing). Always confirm live prices at your broker before trading. For daily trades, a broker data feed is worth it.
- **After-hours scans use last-trade prices.** When the market is closed Yahoo has no live bids, so the scanner (if you leave 'Use last-trade prices when the market is closed' on) works from each option's last trade and recomputes implied volatility itself, because Yahoo's after-hours volatility numbers are placeholders. Rows are labeled in the *Price source* column. Treat them as a plan for the next session, and re-check live bids at the open.
- **Probabilities are estimates.** Models assume smooth price moves; real markets gap. Gaps and crashes are where put sellers lose the most.
- **Past results don't predict future results.** Historical win % only reflects the last ~3 years.
- **This tool screens; it doesn't guarantee profit.** Selling puts earns small steady income and can take large losses in a sell-off. Size so that a bad week is survivable.
- This is an educational tool, not financial advice.
""")
