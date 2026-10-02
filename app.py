import hmac
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---- Reload guard -------------------------------------------------------------------------------------------
# After a redeploy, Streamlit Cloud can keep OLD copies of this project's modules in memory while it loads the new
# app.py, which shows up as "ImportError: cannot import name ..." until someone reboots the app. This notices when
# any project file changed and drops the cached copies, so every import below is fresh.
def _reload_project_modules():
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.py"))
    build = tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    previous = getattr(sys, "_csp_build", None)
    if previous != build:
        for f in files:
            if f.stem != Path(__file__).stem:
                sys.modules.pop(f.stem, None)
        if previous is not None:
            st.cache_data.clear()
        sys._csp_build = build


_reload_project_modules()
# ---- end reload guard ---------------------------------------------------------------------------------------

import engine as E
import journal as J
import markets as MK
import view_income
import view_lookup
import view_market
import view_portfolio
import view_scanner
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
# PUT SCANNER (see view_scanner.py)
# =====================================================================
with tab_scan:
    view_scanner.render()

# =====================================================================
# JOURNAL
# =====================================================================
with tab_journal:
    st.subheader("Trade journal")
    st.caption(f"Stored locally in {J.JOURNAL_PATH.name}. Log trades from the My plan tab in the Put Scanner, or add rows below. "
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
- **Canadian stocks:** add the Yahoo exchange suffix: **.TO** for the TSX (RY.TO, TD.TO, ENB.TO), **.V** for the TSX Venture, **.CN** for the CSE, **.NE** for NEO. You can also type `TSX:RY`, and US class shares use a dash (BRK-B). If a plain symbol like ATD has no US listing, the app matches it to its Canadian listing for you. The Portfolio tab converts CAD and USD holdings into the currency you choose (CAD or USD) at today's exchange rate.
- **Income:** the **Dividend calculator** shows what a position pays per year, month, week and day, and what you need to invest for a target monthly income. The **DRIP calculator** compares reinvesting dividends against taking them as cash, with monthly contributions, dividend tax, and a table of "what if the price grows faster or slower". Load any real stock to fill in its actual price, dividend, payout schedule and growth.
- **Put Scanner:** pick how often you trade (including **Same day** for day trading), your cash, and how many stocks to search, then press *Find put ideas*. You get every put contract found, a plain-English top picks list, a sized plan for your cash, and full details. **Journal:** log the trades you take and track them.
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

    with st.expander("Day trading with same-day puts (0DTE)"):
        md("""
- **What it is:** a put that **expires today**. You sell it in the morning, collect the payment, and either buy it back later the same day for less (keeping the difference) or let it expire worthless at 4:00 PM New York time.
- **Which stocks:** only those with an option expiring today. On **Fridays** almost every large stock has one; **Monday to Thursday** only some stocks and ETFs do (SPY, QQQ and IWM have them every day). Use **Check which stocks expire today** in the Put Scanner for the exact list.
- **How to scan:** choose **Same day (expires today)**, enter your cash, and press **Find put ideas**. The scan uses the real time left until the close, not whole days.
- **When:** run it while the market is open, ideally after about 9:45 AM New York time. Before that, many strikes have no bids yet, and after 4:00 PM there is nothing to trade.
- **Read *Below price* first:** it is how far under today's price the strike sits. For same-day puts, a bigger gap is safer.
- **Plan the exit before you sell:** decide the price you will buy the put back at (for example when it has lost 50-70% of its value) and the price where you will cut the loss.
- **Why it is risky:** same-day options move very fast. A stock that drops a few percent can turn a small gain into a loss many times larger within minutes. Keep size small, never risk money you can't lose, and expect to be assigned shares if you hold to the close.
""")

    with st.expander("Your daily routine (about 5 minutes)", expanded=True):
        md("""
1. **Check your open trades first.** Journal tab → *Refresh live P&L*. Close anything marked *Close now* (70%+ of the premium captured) and think about anything *At risk* or *ITM*.
2. **Pick your style.** Weekly suits most people; Daily only if you can watch the market; Monthly if you want low effort.
3. **Enter your cash** and sizing rules. The tool never suggests a trade that needs more cash than you have available.
4. **Choose tickers.** Start with *Most-liquid stocks* and *Index & sector ETFs*. Add the Nasdaq-100 or S&P 500 when you want a wider net.
5. **Run scan and read the banner.** VIX and the S&P trend tell you how aggressive to be today. High fear means smaller size.
6. **Open the My plan tab.** It builds a basket sized to your cash: one position per ticker, best-ranked first.
7. **Sanity-check each trade** on your broker: live bid/ask, news, earnings date. Enter a **limit order at the mid price** (the plan shows it) and be patient.
8. **Log what you actually fill** (Add plan to journal, then edit the premium to your real fill).
9. **Manage, don't just wait.** Buy back at 50-70% of max profit, or roll when a stock threatens your strike.
""")

    with st.expander("How the Put Scanner works"):
        md("""
- **Nothing is filtered out.** The scanner shows every put contract that has a price on the stocks it searches: expensive stocks, thinly traded contracts, very high or low volatility, very low or very high chances. You decide.
- **How many stocks:** use *How many stocks to search?* to search the 100, 300, 600 or 1,000 most traded optionable US stocks, or every optionable stock (about 3,600, which takes 40+ minutes). Popular ETFs are added unless you turn that off in *Advanced*.
- **Heads-up:** unusual contracts carry plain warnings instead of being hidden: *nobody has traded this contract*, *wide bid-ask gap*, *unusual volatility*, *low chance of keeping it*, *almost certain but pays very little*, *earnings before expiry*, *stock in downtrend*, *leveraged fund*, and *last trade* (no live bid).
- **Looks sensible:** Yes means a 70-98% chance of keeping the payment, someone has traded the contract, the bid-ask gap is under 30%, and it pays at least $10. Sensible contracts are listed first, but every contract is still in **All details**.
- **Fits my cash:** compares the cash a put needs (strike x 100) with your available cash minus your reserve. It never hides a stock.
- **My plan:** builds a basket from the contracts that look sensible, sized to your cash: one contract per stock by default, spread across as many different stocks as fit. Unlike the lists, the plan does apply safety rules, because it recommends what to actually do.
- **Chance of keeping it:** the lower of the option-pricing model's estimate and how often the stock really finished above that strike in past windows of the same length.
- **Pre-market and after-hours:** options only trade 9:30 AM-4:00 PM New York time, but the stock trades from 4:00 AM to 8:00 PM. When the market is closed, the scanner measures every strike from the stock's latest pre-market or after-hours price (marked in *Stock pre/after-hrs*), so a gap up or down is reflected in *Below price* and the chance of keeping the payment. Big moves (3%+) get a Heads-up. Option prices themselves are still last trades until the open.
- **No live bid:** when the market is closed (or nobody is bidding) the scanner shows the option's last trade and recomputes implied volatility from it, because Yahoo's after-hours volatility numbers are placeholders. Treat those prices as a plan, and re-check at the open.
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
