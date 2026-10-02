"""Put Scanner tab, kept simple: choose your cash and a style, press one button, get a short plain-English
list of cash-secured put ideas. Everything else lives under 'Advanced'."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import streamlit as st

import engine as E
import journal as J
import markets as MK
from common import esc, kpi, md, money, scatter

# label -> (min DTE, max DTE, min open interest, max spread %, one-line description)
SAME_DAY = "Same day (expires today)"
STYLES = {
    SAME_DAY: (0, 0, 50, 15, "Day trading: sell a put that expires today and close it the same day. Needs the market open."),
    "Weekly (4-10 days)": (4, 10, 300, 12, "Most popular: a new trade about once a week."),
    "Monthly (30-45 days)": (30, 45, 300, 15, "Slower and calmer: one trade a month, close early at 50% profit."),
    "Every 2-4 weeks (11-25 days)": (11, 25, 300, 15, "Middle ground between weekly and monthly."),
    "Daily (1-3 days)": (1, 3, 500, 10, "Very short trades with small payments. Needs daily attention."),
}
DEFAULT_STYLE = SAME_DAY
CONTRACT_CHOICES = {
    "1 contract per stock (more different stocks)": 1,
    "Up to 2 contracts per stock": 2,
    "Up to 3 contracts per stock": 3,
    "Up to 5 contracts per stock": 5,
    "As many as my cash allows": None,
}

DETAIL_COLS = ["Relaxed", "Quote", "Expiration", "DTE", "Strike", "% OTM", "Distance (σ)", "Premium (Bid)", "Mid",
               "Premium $", "Delta", "Est. Win Prob %", "Hist. Win %", "Win % (cons.)", "Edge $",
               "Annualized Yield %", "Return/Day %", "Score", "Lev", "IV/HV", "Trend", "RSI", "P&L @ -10% $",
               "P&L @ -20% $", "Exit Target ($)", "Next Earnings", "Earnings Alert", "Ex-Div Alert", "Breakeven",
               "Capital Req. $", "Open Interest", "Spread %"]

_CC = st.column_config
COLUMN_CONFIG = {
    "Relaxed": _CC.TextColumn("Search"), "Quote": _CC.TextColumn("Price source"),
    "Est. Win Prob %": _CC.NumberColumn("Model win", format="%.1f%%"),
    "Hist. Win %": _CC.NumberColumn("Hist. win", format="%.1f%%"),
    "Win % (cons.)": _CC.ProgressColumn("Win (cons.)", format="%.1f%%", min_value=50, max_value=100),
    "Score": _CC.NumberColumn("Score", format="%.1f"),
    "Edge $": _CC.NumberColumn("Edge ($)", format="%d"),
    "Annualized Yield %": _CC.NumberColumn("Ann. yield", format="%.1f%%"),
    "Return/Day %": _CC.NumberColumn("Per day", format="%.2f%%"),
    "% OTM": _CC.NumberColumn("% OTM", format="%.1f%%"),
    "Distance (σ)": _CC.NumberColumn("Dist (σ)", format="%.2f"),
    "Spread %": _CC.NumberColumn("Spread", format="%.1f%%"),
    "Strike": _CC.NumberColumn("Strike", format="$%g"),
    "Premium (Bid)": _CC.NumberColumn("Bid", format="$%.2f"),
    "Mid": _CC.NumberColumn("Mid", format="$%.2f"),
    "Breakeven": _CC.NumberColumn("Breakeven", format="$%.2f"),
    "Capital Req. $": _CC.NumberColumn("Cash needed", format="$%d"),
    "Premium $": _CC.NumberColumn("Premium", format="$%d"),
    "P&L @ -10% $": _CC.NumberColumn("P&L @ -10% ($)", format="%d"),
    "P&L @ -20% $": _CC.NumberColumn("P&L @ -20% ($)", format="%d"),
    "Exit Target ($)": _CC.TextColumn("Close at"), "Earnings Alert": _CC.TextColumn("Earnings"),
    "Ex-Div Alert": _CC.TextColumn("Ex-div"), "Open Interest": _CC.NumberColumn("OI", format="%d"),
    "Delta": _CC.NumberColumn("Delta", format="%.3f"), "IV/HV": _CC.NumberColumn("IV/HV", format="%.2f"),
    "RSI": _CC.NumberColumn("RSI", format="%d"),
    "Contracts": _CC.NumberColumn("Contracts", format="%d"),
    "Sell limit (mid)": _CC.NumberColumn("Sell at (limit)", format="$%.2f"),
    "Cash secured $": _CC.NumberColumn("Cash set aside", format="$%d"),
    "% of cash": _CC.NumberColumn("% of cash", format="%.0f%%"),
}


# ---------------- Running the scan (strict, then relaxed, then widened) ----------------


def _execute(tickers, params, status):
    """Scan in stages so you almost always get something: strict filters, then relaxed, then widened."""
    results, notes_map, near_miss, soft_miss = {}, {}, [], []

    def run_pass(batch, workers, label, prm, tag=""):
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
                    df["Relaxed"] = tag
                    results[t] = df
                elif spot is None:
                    failed.append(t)
                else:
                    if not tag:
                        if counts.get("affordable", 0) > 0 and not counts.get("downtrend_skip"):
                            near_miss.append(t)
                        if not counts.get("too_expensive"):
                            soft_miss.append(t)
                    notes_map[t] = E.empty_reason(t, spot, counts, prm)
                status.update(label=f"{label}: {i}/{len(batch)} checked · {len(results)} with ideas so far")
        return failed

    rows = lambda: sum(len(d) for d in results.values())
    workers = E.MAX_WORKERS if len(tickers) <= 120 else 3
    failed = run_pass(tickers, workers, "Searching", params)
    for attempt, cooldown in enumerate((20, 40), start=1):
        if not failed:
            break
        status.update(label=f"Yahoo is busy: pausing {cooldown}s then retrying {len(failed)} stock(s)…")
        time.sleep(cooldown)
        failed = run_pass(failed, 1 if attempt == 2 else 2, f"Retry {attempt}", params)

    stages = []
    if rows() < 3 and near_miss:
        status.update(label=f"Few ideas: loosening liquidity and probability limits on {len(near_miss)} stock(s)…")
        run_pass(near_miss, workers, "Loosened search", E.relax_params(params), "Relaxed filters")
        stages.append("relaxed")
    if rows() < 3 and params.max_dte > 0:
        pool = [t for t in soft_miss if t not in results]
        if pool:
            status.update(label=f"Still few ideas: widening the search on {len(pool)} stock(s)…")
            run_pass(pool, workers, "Wider search", E.widen_params(params), "Widened search")
            stages.append("widened")

    notes = [v for t, v in notes_map.items() if t not in results]
    notes += [f"{t}: could not fetch price/options data (Yahoo may be busy; try again shortly)." for t in failed]
    return results, notes, stages


# ---------------- Results ----------------


def _heads_up(row):
    flags = []
    if row.get("Earnings Alert"):
        flags.append("earnings before expiry")
    if row.get("Trend") == "Downtrend":
        flags.append("stock in downtrend")
    if row.get("Lev", 1) > 1:
        flags.append(f"{int(row['Lev'])}x leveraged fund")
    if row.get("Quote"):
        flags.append("last-trade price")
    if row.get("Relaxed"):
        flags.append(row["Relaxed"].lower())
    return ", ".join(flags)


def _top_picks(summary, rank_col, limit=12):
    best = summary.sort_values(rank_col, ascending=False, na_position="last").drop_duplicates("Ticker").head(limit)
    view = pd.DataFrame({
        "Stock": best["Ticker"].values,
        "Sell this put": [f"${k:g} put" for k in best["Strike"]],
        "Expires": [("Today" if d == 0 else f"{e}  ({d} days)") for e, d in zip(best["Expiration"], best["DTE"])],
        "Below price": best["% OTM"].values,
        "You collect": best["Premium $"].values,
        "Chance you keep it": best["Win % (cons.)"].values,
        "Return": best["Yield %"].values,
        "Cash to set aside": best["Capital Req. $"].values,
        "Heads-up": [_heads_up(r) for _, r in best.iterrows()],
    })
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Below price": _CC.NumberColumn("Below price", format="%.1f%%",
                                        help="How far under today's price the strike is. A bigger gap is safer."),
        "You collect": _CC.NumberColumn("You collect", format="$%d", help="Paid to you today, per contract (100 shares)."),
        "Chance you keep it": _CC.ProgressColumn("Chance you keep it", format="%.0f%%", min_value=50, max_value=100,
                                                help="Estimated chance the stock stays above the strike, so you keep "
                                                     "the whole payment."),
        "Return": _CC.NumberColumn("Return", format="%.1f%%", help="Payment as a % of the cash set aside, "
                                                                   "for this one trade."),
        "Cash to set aside": _CC.NumberColumn("Cash to set aside", format="$%d"),
    })
    st.caption("**How to read this:** you are paid *You collect* today for agreeing to buy 100 shares at the strike "
               "price if the stock ends below it at expiry. If it stays above, you keep the money and nothing else "
               "happens. *Cash to set aside* is what must sit in your account to cover buying the shares.")
    return best


def _in_words(best, n=3):
    for _, r in best.head(n).iterrows():
        when = "today" if r["DTE"] == 0 else r["Expiration"]
        md(f"**{r['Ticker']}** · Sell 1 **${r['Strike']:g} put** expiring {when} → you collect about "
           f"**${int(r['Premium $']):,}**. If {r['Ticker']} stays above ${r['Strike']:g} you keep it all "
           f"(roughly **{r['Win % (cons.)']:.0f}%** chance). You set aside ${int(r['Capital Req. $']):,}.")


def _plan_tab(summary, acct, rank_col):
    label = st.selectbox("Contracts per stock", list(CONTRACT_CHOICES), key="contracts_choice",
                         help="One contract per stock spreads your cash over more different stocks, and you can always "
                              "add contracts at the same strike yourself. The plan updates instantly.")
    plan, ps = E.build_plan(summary, acct["capital"], acct["reserve_pct"], acct["max_pos_pct"], acct["max_positions"],
                            acct["max_lev_pct"], rank_col, CONTRACT_CHOICES[label])
    if plan.empty:
        st.warning("No position fits your cash. Try more cash, a lower reserve, or cheaper stocks.")
        return
    k = st.columns(4)
    kpi(k[0], "Cash set aside", money(ps["deployed"]), f"{ps['deployed_pct']:.0f}% of your cash · {ps['positions']} stocks")
    kpi(k[1], "You collect", money(ps["premium"]), f"{ps['return_pct']:.2f}% on that cash, if all expire worthless")
    kpi(k[2], "If every stock drops 20%", money(ps["worst_20"]), f"{ps['worst_20'] / acct['capital'] * 100:.1f}% of your cash")
    kpi(k[3], "Cash left over", money(ps["cash_left"]), "reserve + unused")
    cols = ["Ticker", "Expiration", "DTE", "Strike", "Contracts", "Sell limit (mid)", "Premium $", "Cash secured $",
            "Win % (cons.)", "P&L @ -20% $", "Trend", "Earnings Alert", "Relaxed", "Quote"]
    st.dataframe(plan[[c for c in cols if c in plan]], hide_index=True, width="stretch", column_config=COLUMN_CONFIG)

    heads = []
    if not plan[plan["Earnings Alert"] != ""].empty:
        heads.append("Earnings fall before expiry for " + ", ".join(plan[plan["Earnings Alert"] != ""]["Ticker"]) + ".")
    if not plan[plan["Trend"] == "Downtrend"].empty:
        heads.append("Downtrend: " + ", ".join(plan[plan["Trend"] == "Downtrend"]["Ticker"]) + " (can keep falling).")
    if not plan[plan["Lev"] > 1].empty:
        heads.append("Leveraged funds: " + ", ".join(plan[plan["Lev"] > 1]["Ticker"]) + " (can fall 30%+ in days).")
    if not plan[plan["Size flag"] != ""].empty:
        heads.append("Large share of your cash: " + ", ".join(plan[plan["Size flag"] != ""]["Ticker"]) + ".")
    if heads:
        st.info(esc("**Heads-up:** " + " ".join(heads)))
    st.caption(f"The plan has {ps['positions']} stock(s) because {ps['stop_reason']}. "
               "Enter orders as limits at the *Sell at* price; payments shown use the lower bid, so they are cautious.")
    if st.button("Add this plan to my journal", type="primary", key="sc_journal"):
        n = J.add_trades(plan)
        st.success(f"Added {n} trade(s) to your journal." if n else "These trades were already logged today.")


def _details_tab(summary, scan, rank_col):
    st.dataframe(summary[["Ticker"] + [c for c in DETAIL_COLS if c in summary]], hide_index=True, width="stretch",
                 column_config=COLUMN_CONFIG)
    st.download_button("Download CSV", summary.to_csv(index=False).encode("utf-8"),
                       file_name=f"put_ideas_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    with st.expander("Chart: chance of keeping it vs return"):
        scatter(summary, "Win % (cons.)", "Annualized Yield %", "Ticker", size="Capital Req. $", height=380)
    if scan["notes"]:
        with st.expander(f"Why other stocks were left out ({len(scan['notes'])})"):
            md("\n".join(f"- {n}" for n in scan["notes"]))


def _show_results(scan):
    acct = scan["acct"]
    rank_col = E.RANK_OPTIONS[scan["rank_by"]]
    st.markdown(f"**{scan['label']}** · fetched {scan['fetched_at']}")
    regime = scan.get("regime") or {}
    if regime.get("vix") is not None:
        spy = f" · S&P 500 {regime['spy_trend'].lower()}" if regime.get("spy_trend") else ""
        st.caption(f"Market mood: VIX {regime['vix']:.1f} ({regime['level']}){spy}. {regime['advice']}")

    if not scan["results"]:
        st.error("No put ideas found for these settings.")
        if scan.get("style") == SAME_DAY:
            st.markdown("**For same-day trades:**\n"
                        "- The market must be open (9:30 AM to 4:00 PM New York). Early in the session many strikes "
                        "have no bids yet, so try after 9:45 AM.\n"
                        "- Use *Check which stocks expire today* above: only those stocks have a same-day put.\n"
                        "- Lower the *chance of keeping the payment* range in Advanced (for example 75-97%).")
        else:
            st.markdown("**Try one of these:**\n"
                        "- Pick a longer style (Monthly) or raise your cash.\n"
                        "- Open *Advanced* and add the **Entire US options market** to the stocks to scan.\n"
                        "- If the market is closed, make sure *Use last-trade prices when the market is closed* is ticked.")
        with st.expander(f"Why ({len(scan['notes'])} notes)"):
            md("\n".join(f"- {n}" for n in scan["notes"][:60]))
        return

    summary = pd.concat(scan["results"].values(), ignore_index=True)
    summary = summary.sort_values(rank_col, ascending=False, na_position="last").reset_index(drop=True)

    notes = []
    if (summary.get("Quote", "") != "").any():
        notes.append("Some prices are **last trades from when the market was closed**. Use them to plan, and re-check "
                     "live prices at the open.")
    if (summary["Relaxed"] == "Relaxed filters").any():
        notes.append("Few ideas met your strict limits, so some use **looser liquidity/probability limits** (see the "
                     "*Heads-up* column).")
    if (summary["Relaxed"] == "Widened search").any():
        notes.append("Still few ideas, so the search was **widened** (earnings and downtrend stocks allowed, longer time "
                     "window). Check each *Heads-up* before trading.")
    if notes:
        st.info("\n\n".join(notes))

    t_top, t_plan, t_all = st.tabs(["Top picks", "My plan", "All details"])
    with t_top:
        best = _top_picks(summary, rank_col)
        st.markdown("##### The top three, in plain words")
        _in_words(best)
    with t_plan:
        _plan_tab(summary, acct, rank_col)
    with t_all:
        _details_tab(summary, scan, rank_col)


# ---------------- Page ----------------


def _today_panel(extra, universes):
    """Which stocks have options that expire today? (the shortlist for same-day trading)"""
    ny = E.ny_now()
    with st.expander(f"📅 Which stocks have options expiring today? ({ny:%a %b %d})",
                     expanded=not st.session_state.get("scan_v2")):
        st.caption("For same-day trading you can only use stocks that have a put expiring **today**. Mondays to "
                   "Thursdays only some stocks and ETFs (like SPY, QQQ, IWM) have one; Fridays most do.")
        if st.button("Check which stocks expire today", key="sc_today_btn"):
            tickers = [t.strip().upper() for t in extra.replace("\n", ",").split(",") if t.strip()]
            for name in universes:
                if name in E.UNIVERSES:
                    tickers += E.UNIVERSES[name]()
            tickers = tuple(dict.fromkeys(tickers))
            with st.spinner(f"Checking {len(tickers)} stocks…"):
                st.session_state["sc_today"] = MK.get_same_day_expiries(tickers) + (len(tickers),)
        if "sc_today" in st.session_state:
            df, ts, n = st.session_state["sc_today"]
            if df.empty:
                st.info(f"None of the {n} stocks checked have options expiring today. (Weekends and market holidays "
                        "have no same-day expiries.)")
            else:
                df = df.sort_values(["Expiries this week", "Symbol"], ascending=[False, True])
                st.success(f"{len(df)} of {n} stocks have options expiring today.")
                st.dataframe(df, hide_index=True, width="stretch", column_config={
                    "Symbol": _CC.TextColumn("Stock"), "Price": _CC.NumberColumn(format="$%.2f"),
                    "Day %": _CC.NumberColumn("Today", format="%+.2f%%"),
                    "Expiries this week": _CC.NumberColumn("Expiry dates this week", format="%d",
                                                          help="Stocks with several dates this week have daily options.")})
                st.caption(f"Checked {ts:%H:%M:%S}. Choose **Same day** below and press *Find put ideas* to see which "
                           "strikes are worth selling on these stocks.")


def render():
    market_open, closed_msg = E.us_market_status()
    if not market_open:
        st.info("🌙 The US market is closed right now. Same-day trades need live prices, so run **Same day** between about "
                "9:45 AM and 3:45 PM New York time. Other styles use each option's last trade so you can plan ahead.")

    with st.form("scanner_form", border=True):
        c = st.columns([2.2, 1.2, 1.2])
        style = c[0].radio("How often do you want to trade?", list(STYLES), index=list(STYLES).index(DEFAULT_STYLE),
                           key="sc_style", horizontal=True,
                           help="Sets how far away the expiry dates are.\n\n" +
                                "\n\n".join(f"**{k}**: {v[4]}" for k, v in STYLES.items()))
        capital = c[1].number_input("Cash available ($)", min_value=1000, value=25000, step=1000, key="sc_cash",
                                    help="Cash you can set aside. Each put needs strike × 100 per contract.")
        c[2].write("")
        c[2].write("")
        run = c[2].form_submit_button("Find put ideas", type="primary", width="stretch")

        with st.expander("Advanced settings (optional)"):
            a = st.columns([3, 2])
            extra = a[0].text_input("Extra stocks to include (optional)", "", key="sc_extra",
                                    placeholder="e.g. NVDA, AAPL, KO")
            universes = a[1].multiselect(
                "Stocks to search", list(E.UNIVERSES) + [E.MARKET_UNIVERSE], key="sc_universes",
                default=["Most-liquid stocks (~80)", "Index & sector ETFs"],
                help="'Entire US options market' checks every optionable US stock that fits your cash and trades "
                     "enough volume, most-traded first. Bigger lists take longer (about 0.6 seconds per stock).")
            b = st.columns(4)
            reserve = b[0].number_input("Keep in reserve %", 0, 90, 10, 5, key="sc_reserve")
            max_pos = b[1].number_input("Max stocks in plan", 1, 30, 8, 1, key="sc_maxpos")
            pos_pct = b[2].number_input("Preferred max per stock %", 1, 100, 15, 5, key="sc_pospct")
            lev_pct = b[3].number_input("Max in leveraged funds %", 0, 100, 15, 5, key="sc_lev")
            d = st.columns(4)
            win = d[0].slider("Chance of keeping the payment (%)", 50, 99, (80, 95), key="sc_win",
                              help="Only show puts in this range. Higher = safer but pays less.")
            min_prem = d[1].number_input("Min payment per contract ($)", 0, 1000, 20, 5, key="sc_minprem")
            skip_down = d[2].checkbox("Skip stocks in a downtrend", False, key="sc_down")
            excl_earn = d[3].checkbox("Skip trades with earnings before expiry", True, key="sc_earn",
                                      help="Earnings can gap a stock sharply. If this leaves too little, the search "
                                           "widens automatically and labels those rows.")
            e = st.columns(4)
            rank = e[0].selectbox("Rank ideas by", list(E.RANK_OPTIONS), key="sc_rank")
            allow_stale = e[1].checkbox("Use last-trade prices when the market is closed", value=not market_open,
                                        key="sc_stale", help="After hours Yahoo has no live bids. This plans with each "
                                                              "option's last trade instead (labeled in results).")
            m = st.columns(3)
            mk_price = m[0].number_input("Whole market: min stock price ($)", 0.0, 1000.0, 5.0, 1.0, key="sc_mkprice")
            mk_vol = m[1].number_input("Whole market: min daily volume", 0, 100_000_000, 1_000_000, 250_000, key="sc_mkvol")
            mk_n = m[2].number_input("Whole market: max stocks", 50, 2000, 400, 50, key="sc_mkn")

    _today_panel(extra, universes)

    if run:
        tickers = [t.strip().upper() for t in extra.replace("\n", ",").split(",") if t.strip()]
        for name in universes:
            if name in E.UNIVERSES:
                tickers += E.UNIVERSES[name]()
        max_cost = capital * (1 - reserve / 100)
        market_info = None
        if E.MARKET_UNIVERSE in universes:
            mk, market_info = E.build_market_universe(max_cost, mk_price, mk_vol, mk_n)
            tickers += mk
            if market_info["error"]:
                st.warning(market_info["error"])
        tickers = list(dict.fromkeys(tickers))[:E.MAX_TICKERS]
        if not tickers:
            st.warning("Pick at least one group of stocks under Advanced settings.")
        else:
            dmin, dmax, oi, spread, _ = STYLES[style]
            allow_stale = allow_stale and style != SAME_DAY   # a stale last trade is useless for same-day trading
            params = E.ScanParams(
                risk_free_rate=E.get_risk_free_rate(), min_dte=dmin, max_dte=dmax, win_prob_min=win[0],
                win_prob_max=win[1], min_oi=oi, max_spread_pct=spread, top_n=3, min_hist_win=70,
                min_premium_usd=min_prem, max_contract_cost=max_cost, exclude_earnings=excl_earn,
                skip_downtrend=skip_down, allow_stale=allow_stale, rank_by=rank)
            eta = max(1, round(len(tickers) * (0.25 if len(tickers) <= 120 else 0.6) / 60))
            with st.status(f"Searching {len(tickers)} stocks (about {eta} min)…", expanded=False) as status:
                results, notes, stages = _execute(tickers, params, status)
                status.update(label="Checking market mood…")
                regime = E.get_market_regime()
                status.update(label="Done", state="complete")
            st.session_state["scan_v2"] = {
                "results": results, "notes": notes, "regime": regime, "rank_by": rank,
                "label": f"{len(results)} of {len(tickers)} stocks have put ideas", "style": style,
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "market_info": market_info,
                "acct": dict(capital=capital, reserve_pct=reserve, max_pos_pct=pos_pct, max_positions=max_pos,
                             max_lev_pct=lev_pct),
            }
            st.rerun()

    scan = st.session_state.get("scan_v2")
    if not scan:
        st.markdown("**How it works:** set your cash, press **Find put ideas**, and you get a short list of stocks where "
                    "you could be paid to agree to buy shares at a lower price. Nothing is bought until you place an "
                    "order with your broker.")
        return
    mi = scan.get("market_info")
    if mi and not mi["error"]:
        st.caption(f"Whole market: {mi['optionable']:,} optionable symbols → {mi['passed']:,} fit your cash and volume → "
                   f"searched the {mi['scanning']:,} most traded.")
    _show_results(scan)
