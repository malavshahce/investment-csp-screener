"""Put Scanner tab, kept simple: choose a style, your cash and how many stocks to search, press one button.
Nothing is hidden by filters: every put that has a price is shown, with plain warnings (Heads-up) for the odd ones."""
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import streamlit as st

import engine as E
import journal as J
import markets as MK
from common import esc, kpi, md, money, scatter

# label -> (min DTE, max DTE, (unused), (unused), one-line description)
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

HOW_MANY = {
    "Top 100 most traded stocks": 100,
    "Top 300 most traded stocks": 300,
    "Top 600 most traded stocks": 600,
    "Top 1,000 most traded stocks": 1000,
    "Everything optionable (~3,600 stocks, 40+ min)": 4000,
}
DEFAULT_HOW_MANY = "Top 300 most traded stocks"
ROWS_PER_STOCK = {"All strikes": 1_000_000, "Best 50 per stock": 50, "Best 25 per stock": 25, "Best 10 per stock": 10}

DETAIL_COLS = ["Looks sensible", "Heads-up", "Quote", "Expiration", "DTE", "Strike", "% OTM", "Distance (σ)", "Premium (Bid)", "Mid",
               "Premium $", "Delta", "Est. Win Prob %", "Hist. Win %", "Win % (cons.)", "Edge $",
               "Annualized Yield %", "Return/Day %", "Score", "Lev", "IV/HV", "Trend", "RSI", "P&L @ -10% $",
               "P&L @ -20% $", "Exit Target ($)", "Next Earnings", "Earnings Alert", "Ex-Div Alert", "Breakeven",
               "Capital Req. $", "Open Interest", "Volume", "Spread %", "IV %"]

_CC = st.column_config
COLUMN_CONFIG = {
    "Relaxed": _CC.TextColumn("Search"), "Quote": _CC.TextColumn("Price source"),
    "Heads-up": _CC.TextColumn("Heads-up", width="large"), "Looks sensible": _CC.TextColumn("Looks sensible"), "Volume": _CC.NumberColumn("Volume today", format="%d"),
    "IV %": _CC.NumberColumn("IV", format="%.0f%%"),
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


# ---------------- Running the scan ----------------


def _execute(tickers, params, status):
    """One pass over every stock (no filters), pausing and retrying the ones Yahoo was too busy to answer."""
    results, notes_map = {}, {}

    def run_pass(batch, workers, label):
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(E.scan_ticker_for_csp, t, params): t for t in batch}
            for i, fut in enumerate(as_completed(futures), start=1):
                t = futures[fut]
                try:
                    df, spot, counts = fut.result()
                except Exception:
                    failed.append(t)
                    continue
                if not df.empty:
                    results[t] = df
                elif spot is None:
                    failed.append(t)
                else:
                    notes_map[t] = E.empty_reason(t, spot, counts, params)
                status.update(label=f"{label}: {i}/{len(batch)} stocks checked · {len(results)} have puts so far")
        return failed

    workers = E.MAX_WORKERS if len(tickers) <= 120 else 3
    failed = run_pass(tickers, workers, "Searching")
    # A big search can hit Yahoo's rate limit, so pause and retry. For a few typed symbols a failure almost always
    # means a wrong symbol, so retry once, quickly.
    cooldowns = (2,) if len(tickers) <= 10 else (20, 40)
    for attempt, cooldown in enumerate(cooldowns, start=1):
        if not failed:
            break
        status.update(label=f"Yahoo is busy: pausing {cooldown}s then retrying {len(failed)} stock(s)…")
        time.sleep(cooldown)
        failed = run_pass(failed, 1 if attempt == 2 else 2, f"Retry {attempt}")

    notes = [v for t, v in notes_map.items() if t not in results]
    notes += [f"{t}: no price data came back. Check the symbol is spelled right (class shares use a dash, e.g. BRK-B), "
              "or try again in a minute if Yahoo was busy." for t in failed]
    return results, notes, []


# ---------------- Results ----------------


def _heads_up(row):
    """Plain warnings for contracts that deserve a second look (nothing is hidden; this is how you spot them)."""
    flags = []
    if row.get("Quote"):
        flags.append(str(row["Quote"]).lower())
    if row.get("Earnings Alert"):
        flags.append("earnings before expiry")
    if row.get("Trend") == "Downtrend":
        flags.append("stock in downtrend")
    if row.get("Lev", 1) > 1:
        flags.append(f"{int(row['Lev'])}x leveraged fund")
    if not (row.get("Open Interest") or 0) and not (row.get("Volume") or 0):
        flags.append("nobody has traded this contract")
    sp = row.get("Spread %")
    if sp == sp and sp is not None and sp > 30:
        flags.append("wide bid-ask gap")
    iv = row.get("IV %")
    if iv == iv and iv is not None and (iv < 5 or iv > 300):
        flags.append("unusual volatility")
    w = row.get("Win % (cons.)")
    if w != w or w is None:
        flags.append("no probability estimate")
    elif w < 70:
        flags.append("low chance of keeping it")
    elif w > 98:
        flags.append("almost certain but pays very little")
    return ", ".join(flags)


def _top_picks(summary, rank_col, deployable, limit=15, one_per_stock=True):
    # summary is already sensible-first, then ranked
    best = (summary.drop_duplicates("Ticker") if one_per_stock else summary).head(limit)
    view = pd.DataFrame({
        "Stock": best["Ticker"].values,
        "Sell this put": [f"${k:g} put" for k in best["Strike"]],
        "Expires": [("Today" if d == 0 else f"{e}  ({d} days)") for e, d in zip(best["Expiration"], best["DTE"])],
        "Below price": best["% OTM"].values,
        "You collect": best["Premium $"].values,
        "Chance you keep it": best["Win % (cons.)"].values,
        "Return": best["Yield %"].values,
        "Cash to set aside": best["Capital Req. $"].values,
        "Fits my cash": ["Yes" if c <= deployable else "No" for c in best["Capital Req. $"]],
        "Looks sensible": best["Looks sensible"].values,
        "Open interest": best["Open Interest"].values,
        "Heads-up": best["Heads-up"].values,
    })
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Below price": _CC.NumberColumn("Below price", format="%.1f%%",
                                        help="How far under today's price the strike is. A bigger gap is safer."),
        "You collect": _CC.NumberColumn("You collect", format="$%d", help="Paid to you today, per contract (100 shares)."),
        "Chance you keep it": _CC.ProgressColumn("Chance you keep it", format="%.0f%%", min_value=0, max_value=100,
                                                help="Estimated chance the stock stays above the strike, so you keep "
                                                     "the whole payment."),
        "Return": _CC.NumberColumn("Return", format="%.1f%%", help="Payment as a % of the cash set aside, "
                                                                   "for this one trade."),
        "Cash to set aside": _CC.NumberColumn("Cash to set aside", format="$%d"),
        "Open interest": _CC.NumberColumn("Open interest", format="%d",
                                          help="Contracts other people currently hold. 0 means nobody has traded it."),
        "Looks sensible": _CC.TextColumn("Looks sensible", help="Yes = a 70-98% chance of keeping it, someone has traded "
                                                                "the contract, the bid-ask gap is under 30% and it pays "
                                                                "at least $10."),
        "Heads-up": _CC.TextColumn("Heads-up", width="large"),
    })
    st.caption("**How to read this:** " + ("one row per stock, its best put" if one_per_stock else "the best puts on this stock") +
               " (contracts that *look sensible* are listed first). You are paid *You collect* today for agreeing to "
               "buy 100 shares at the strike price if the stock ends below it at expiry. If it stays above, you keep the "
               "money. *Cash to set aside* is what must sit in your account to cover buying the shares. Nothing is hidden: "
               "unusual contracts are shown with a *Heads-up*; **All details** has every contract.")
    return best


def _in_words(best, n=3):
    for _, r in best.head(n).iterrows():
        when = "today" if r["DTE"] == 0 else r["Expiration"]
        md(f"**{r['Ticker']}** · Sell 1 **${r['Strike']:g} put** expiring {when} → you collect about "
           f"**${int(r['Premium $']):,}**. If {r['Ticker']} stays above ${r['Strike']:g} you keep it all "
           f"(roughly **{r['Win % (cons.)']:.0f}%** chance). You set aside ${int(r['Capital Req. $']):,}.")


def _plan_tab(summary, acct, rank_col):
    # The plan recommends what to actually do with your cash, so it only draws from contracts that are sensible to
    # trade. Every other contract is still listed in All details.
    pool = summary[summary["Looks sensible"] == "Yes"]
    st.caption("The plan only picks contracts with a 70-98% chance of being kept, some trading activity, a bid-ask gap "
               "under 30% and a payment of at least $10. These are safety rules for the plan only; "
               "**All details** shows every contract.".replace("$", "\\$"))
    if pool.empty:
        st.warning("No contract met the plan's safety rules. See All details for every contract.")
        return
    label = st.selectbox("Contracts per stock", list(CONTRACT_CHOICES), key="contracts_choice",
                         help="One contract per stock spreads your cash over more different stocks, and you can always "
                              "add contracts at the same strike yourself. The plan updates instantly.")
    plan, ps = E.build_plan(pool, acct["capital"], acct["reserve_pct"], acct["max_pos_pct"], acct["max_positions"],
                            acct["max_lev_pct"], rank_col, CONTRACT_CHOICES[label])
    if plan.empty:
        st.warning("No position fits your cash. Try more cash or a lower reserve.")
        return
    k = st.columns(4)
    kpi(k[0], "Cash set aside", money(ps["deployed"]), f"{ps['deployed_pct']:.0f}% of your cash · {ps['positions']} stocks")
    kpi(k[1], "You collect", money(ps["premium"]), f"{ps['return_pct']:.2f}% on that cash, if all expire worthless")
    kpi(k[2], "If every stock drops 20%", money(ps["worst_20"]), f"{ps['worst_20'] / acct['capital'] * 100:.1f}% of your cash")
    kpi(k[3], "Cash left over", money(ps["cash_left"]), "reserve + unused")
    cols = ["Ticker", "Expiration", "DTE", "Strike", "Contracts", "Sell limit (mid)", "Premium $", "Cash secured $",
            "Win % (cons.)", "P&L @ -20% $", "Trend", "Earnings Alert", "Quote"]
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
    if not plan[plan["Quote"] != ""].empty:
        heads.append("Some prices are last trades, not live bids: re-check before trading.")
    if heads:
        st.info(esc("**Heads-up:** " + " ".join(heads)))
    st.caption(f"The plan has {ps['positions']} stock(s) because {ps['stop_reason']}. "
               "Enter orders as limits at the *Sell at* price; payments shown use the lower bid, so they are cautious.")
    if st.button("Add this plan to my journal", type="primary", key="sc_journal"):
        n = J.add_trades(plan)
        st.success(f"Added {n} trade(s) to your journal." if n else "These trades were already logged today.")


def _details_tab(summary, scan, rank_col):
    st.caption(f"Every put contract found: {len(summary):,} rows. Click a column heading to sort, or use the search icon "
               "at the top right of the table.")
    st.dataframe(summary[["Ticker"] + [c for c in DETAIL_COLS if c in summary]], hide_index=True, width="stretch",
                 column_config=COLUMN_CONFIG, height=520)
    st.download_button("Download CSV", summary.to_csv(index=False).encode("utf-8"),
                       file_name=f"put_ideas_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
    with st.expander("Chart: chance of keeping it vs return"):
        sample = summary.dropna(subset=["Win % (cons.)"])
        sample = sample.sample(min(len(sample), 1500), random_state=0) if len(sample) > 1500 else sample
        scatter(sample, "Win % (cons.)", "Annualized Yield %", "Ticker", size="Capital Req. $", height=380)
    if scan["notes"]:
        with st.expander(f"Stocks with no put prices ({len(scan['notes'])})"):
            md("\n".join(f"- {n}" for n in scan["notes"][:200]))


def _stock_cards(scan, summary):
    """When you looked up specific stocks: one summary row each, plus a plain reason for any with no puts."""
    for t in scan["specific"]:
        d = summary[summary["Ticker"] == t]
        if d.empty:
            why = [n for n in scan["notes"] if n.startswith(t + ":")]
            st.warning(f"**{t}**: " + (why[0].split(":", 1)[1].strip() if why else "no put prices found in this expiry "
                                          "window. Check the symbol, or try another style."))
            continue
        r = d.iloc[0]
        c = st.columns(6)
        kpi(c[0], t, f"${r['Spot Price']:,.2f}", "stock price")
        kpi(c[1], "Trend", str(r["Trend"]), f"RSI {r['RSI']:.0f}" if r["RSI"] == r["RSI"] else None)
        kpi(c[2], "Put contracts", f"{len(d):,}", f"{d['Expiration'].nunique()} expiry date(s)")
        kpi(c[3], "Next earnings", str(r["Next Earnings"]), "earnings before expiry" if (d["Earnings Alert"] != "").any() else None)
        kpi(c[4], "50-day average", f"${r['50D SMA']:,.2f}")
        kpi(c[5], "52-week low", f"${r['52W Low']:,.2f}")


def _show_results(scan):
    acct = scan["acct"]
    rank_col = E.RANK_OPTIONS[scan["rank_by"]]
    st.markdown(f"**{scan['label']}** · fetched {scan['fetched_at']}")
    regime = scan.get("regime") or {}
    if regime.get("vix") is not None:
        spy = f" · S&P 500 {regime['spy_trend'].lower()}" if regime.get("spy_trend") else ""
        st.caption(f"Market mood: VIX {regime['vix']:.1f} ({regime['level']}){spy}. {regime['advice']}")

    if not scan["results"] and scan.get("specific"):
        _stock_cards(scan, pd.DataFrame({"Ticker": []}))   # a plain reason for each stock typed
        st.markdown("**Try:** another style (the stock may have no expiry date in this window), or check the symbol "
                    "(class shares use a dash, e.g. BRK-B).")
        return

    if not scan["results"]:
        st.error("No put prices were found for these stocks and expiry dates.")
        if scan.get("style") == SAME_DAY:
            st.markdown("**For same-day trades:**\n"
                        "- Only stocks with a put expiring **today** qualify. Use *Check which stocks expire today* above.\n"
                        "- Early in the session many strikes have no prices yet, so try after 9:45 AM New York time.")
        else:
            st.markdown("**Try one of these:**\n"
                        "- Pick a different style: some stocks have no expiry date in the window you chose.\n"
                        "- Search more stocks with the *How many stocks to search* menu.")
        with st.expander(f"Why ({len(scan['notes'])} notes)"):
            md("\n".join(f"- {n}" for n in scan["notes"][:60]))
        return

    summary = pd.concat(scan["results"].values(), ignore_index=True)
    summary["Heads-up"] = summary.apply(_heads_up, axis=1)
    sensible = (summary["Win % (cons.)"].between(70, 98) & ((summary["Open Interest"] > 0) | (summary["Volume"] > 0))
                & (summary["Spread %"] <= 30) & (summary["Premium $"] >= 10))
    summary["Looks sensible"] = sensible.map({True: "Yes", False: "No"})
    summary = (summary.assign(_ok=sensible).sort_values(["_ok", rank_col], ascending=[False, False], na_position="last")
               .drop(columns="_ok").reset_index(drop=True))

    if not scan.get("specific"):
        flt = st.text_input("Show only these stocks (optional)", key="sc_filter",
                            placeholder="type symbols to narrow the results below, e.g. NVDA, TSLA")
        wanted = {_norm(t) for t in re.split(r"[,\s;]+", flt or "") if t.strip()}
        if wanted:
            summary = summary[summary["Ticker"].isin(wanted)].reset_index(drop=True)
            if summary.empty:
                st.info("None of those stocks have contracts in this search. They may be outside the stocks searched, "
                        "or have no puts in this expiry window. Use *look up specific stocks* above to search them directly.")
                return
    else:
        _stock_cards(scan, summary)

    hide = st.checkbox("Hide contracts nobody has traded (0 open interest and 0 volume)", value=False, key="sc_hide0",
                       help="Off by default so you see everything. Turn on to remove contracts with no activity.")
    if hide:
        summary = summary[(summary["Open Interest"] > 0) | (summary["Volume"] > 0)].reset_index(drop=True)
        if summary.empty:
            st.info("Every contract was untraded, so nothing is left. Turn the checkbox off to see them.")
            return

    if (summary["Quote"] != "").any():
        st.info("Some prices are **last trades, not live bids** (the market is closed or the contract has no buyer). "
                "They are marked in *Heads-up*. Use them to plan, and re-check live prices before trading.")

    t_top, t_plan, t_all = st.tabs(["Top picks", "My plan", "All details"])
    with t_top:
        deployable = acct["capital"] * (1 - acct["reserve_pct"] / 100)
        best = _top_picks(summary, rank_col, deployable, limit=15, one_per_stock=not scan.get("specific"))
        st.markdown("##### The top three, in plain words")
        _in_words(best)
    with t_plan:
        _plan_tab(summary, acct, rank_col)
    with t_all:
        _details_tab(summary, scan, rank_col)


# ---------------- Page ----------------


def _norm(sym):
    """AAPL / aapl / BRK.B -> AAPL / AAPL / BRK-B (the form Yahoo uses)."""
    return sym.strip().upper().replace(".", "-").replace("/", "-")


def _ticker_list(specific, groups, how_many, include_etfs, mk_price, mk_vol):
    """If you typed specific stocks, search exactly those. Otherwise search the N most traded optionable US stocks
    plus popular ETFs and any extra groups. Nothing is dropped for being expensive."""
    typed = [_norm(t) for t in re.split(r"[,\s;]+", specific or "") if t.strip()]
    if typed:
        typed = list(dict.fromkeys(typed))
        return typed, {"optionable": 0, "priced": 0, "passed": 0, "scanning": len(typed), "error": None, "specific": True}
    mk, info = E.build_market_universe(float("inf"), mk_price, mk_vol, HOW_MANY[how_many])
    tickers = list(mk) if not info["error"] else list(E.LIQUID_STOCKS)
    if include_etfs:
        tickers += E.INDEX_ETFS + E.LEVERAGED_ETFS
    for g in groups:
        tickers += E.UNIVERSES[g]()
    return list(dict.fromkeys(tickers))[:E.MAX_TICKERS], info


def _today_panel(get_tickers):
    """Which stocks have options that expire today? (the shortlist for same-day trading)"""
    ny = E.ny_now()
    with st.expander(f"📅 Which stocks have options expiring today? ({ny:%a %b %d})",
                     expanded=not st.session_state.get("scan_v2")):
        st.caption("For same-day trading you can only use stocks that have a put expiring **today**. Mondays to "
                   "Thursdays only some stocks and ETFs (like SPY, QQQ, IWM) have one; Fridays most do.")
        if st.button("Check which stocks expire today", key="sc_today_btn"):
            tickers, _ = get_tickers()
            with st.spinner(f"Checking {len(tickers)} stocks…"):
                st.session_state["sc_today"] = MK.get_same_day_expiries(tuple(tickers)) + (len(tickers),)
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
                st.caption(f"Checked {ts:%H:%M:%S}. Choose **Same day** below and press *Find put ideas* to see the "
                           "puts on these stocks.")


def render():
    market_open, closed_msg = E.us_market_status()
    if not market_open:
        st.info("🌙 The US market is closed right now, so many contracts have no live bid. Those show their last trade "
                "(marked in *Heads-up*) so you can plan ahead. Same-day trades need the market open.")

    with st.form("scanner_form", border=True):
        c = st.columns([2.8, 1.0, 1.0])
        style = c[0].radio("How often do you want to trade?", list(STYLES), index=list(STYLES).index(DEFAULT_STYLE),
                           key="sc_style", horizontal=True,
                           help="Sets how far away the expiry dates are.\n\n" +
                                "\n\n".join(f"**{k}**: {v[4]}" for k, v in STYLES.items()))
        capital = c[1].number_input("Cash available ($)", min_value=1000, value=25000, step=1000, key="sc_cash",
                                    help="Used for the *Fits my cash* column and the plan. It never hides a stock.")
        c[2].write("")
        c[2].write("")
        run = c[2].form_submit_button("Find put ideas", type="primary", width="stretch")
        d = st.columns([1.3, 2.0])
        how_many = d[0].selectbox("How many stocks to search?", list(HOW_MANY), index=list(HOW_MANY).index(DEFAULT_HOW_MANY),
                                  key="sc_howmany",
                                  help="The most traded optionable US stocks first (about 0.6 seconds per stock: 300 is "
                                       "about 3 minutes). 'Everything optionable' covers every US stock that has options.")
        specific = d[1].text_input("…or look up specific stocks", "", key="sc_specific",
                                   placeholder="type symbols, e.g. NVDA, AAPL, BRK.B (searches only these, every contract)",
                                   help="Leave empty to search the stocks chosen on the left. If you type symbols, only "
                                        "those are searched, whether or not they are among the most traded.")

        with st.expander("Advanced settings (optional)"):
            groups = st.multiselect("Add more groups (optional)", list(E.UNIVERSES), key="sc_groups", default=[],
                                    help="Add a ready-made list on top of the search above.")
            m = st.columns(4)
            include_etfs = m[0].checkbox("Include popular ETFs", True, key="sc_etfs",
                                         help="SPY, QQQ, IWM, sector funds and leveraged funds.")
            mk_price = m[1].number_input("Min stock price ($)", 0.0, 1000.0, 1.0, 1.0, key="sc_mkprice",
                                         help="Only to skip sub-$1 penny stocks when picking the most traded.")
            mk_vol = m[2].number_input("Min daily share volume", 0, 100_000_000, 100_000, 50_000, key="sc_mkvol",
                                       help="Only used to rank and pick the most traded stocks.")
            rows_label = m[3].selectbox("Contracts to keep per stock", list(ROWS_PER_STOCK), key="sc_rows",
                                        help="'All strikes' keeps every put. With more than 600 stocks the best 50 per "
                                             "stock are kept so the page stays fast.")
            b = st.columns(5)
            reserve = b[0].number_input("Keep in reserve %", 0, 90, 10, 5, key="sc_reserve")
            max_pos = b[1].number_input("Max stocks in plan", 1, 30, 8, 1, key="sc_maxpos")
            pos_pct = b[2].number_input("Preferred max per stock %", 1, 100, 15, 5, key="sc_pospct")
            lev_pct = b[3].number_input("Max in leveraged funds %", 0, 100, 15, 5, key="sc_lev")
            rank = b[4].selectbox("Rank ideas by", list(E.RANK_OPTIONS), key="sc_rank")

    get_tickers = lambda: _ticker_list(specific, groups, how_many, include_etfs, mk_price, mk_vol)
    _today_panel(get_tickers)

    if run:
        tickers, market_info = get_tickers()
        if market_info["error"] and not market_info.get("specific"):
            st.warning(market_info["error"] + " Using the built-in list of ~80 most-traded stocks instead.")
        is_specific = bool(market_info.get("specific"))
        rows_cap = ROWS_PER_STOCK["All strikes"] if is_specific else ROWS_PER_STOCK[rows_label]
        cap_note = len(tickers) > 600 and rows_cap > 50
        if cap_note:
            rows_cap = 50
        dmin, dmax, _, _, _ = STYLES[style]
        params = E.ScanParams(
            risk_free_rate=E.get_risk_free_rate(), min_dte=dmin, max_dte=dmax, win_prob_min=0, win_prob_max=100,
            min_oi=0, max_spread_pct=float("inf"), top_n=rows_cap, min_hist_win=0, min_premium_usd=0,
            max_contract_cost=float("inf"), allow_stale=True, show_all=True, rank_by=rank)
        eta = max(1, round(len(tickers) * (0.25 if len(tickers) <= 120 else 0.6) / 60))
        with st.status(f"Searching {len(tickers):,} stocks (about {eta} min)…", expanded=False) as status:
            results, notes, _ = _execute(tickers, params, status)
            status.update(label="Checking market mood…")
            regime = E.get_market_regime()
            status.update(label="Done", state="complete")
        n_rows = sum(len(d) for d in results.values())
        st.session_state["scan_v2"] = {
            "results": results, "notes": notes, "regime": regime, "rank_by": rank, "style": style,
            "label": f"{n_rows:,} put contracts across {len(results):,} of {len(tickers):,} "
                     f"stock{'s' if len(tickers) != 1 else ''}",
            "specific": tickers if is_specific else None,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "market_info": market_info, "cap_note": cap_note,
            "acct": dict(capital=capital, reserve_pct=reserve, max_pos_pct=pos_pct, max_positions=max_pos,
                         max_lev_pct=lev_pct),
        }
        st.rerun()

    scan = st.session_state.get("scan_v2")
    if not scan:
        st.markdown("**How it works:** choose how often you trade, press **Find put ideas**, and you see the put contracts "
                    "on the most traded stocks, where you could be paid to agree to buy shares at a lower price. "
                    "Type a symbol in *look up specific stocks* to see every contract for just that stock. Nothing is hidden: unusual contracts carry a *Heads-up*. Nothing is bought until you place an order "
                    "with your broker.")
        return
    mi = scan.get("market_info")
    if mi and not mi["error"] and not mi.get("specific"):
        st.caption(f"{mi['optionable']:,} optionable symbols → {mi['priced']:,} US stocks with prices → searched the "
                   f"{mi['scanning']:,} most traded.")
    if scan.get("cap_note"):
        st.caption("With this many stocks, the best 50 contracts per stock are kept so the page stays fast.")
    _show_results(scan)
