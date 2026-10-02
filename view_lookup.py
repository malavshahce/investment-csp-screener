"""Stock lookup tab: search any ticker/company, live quote, chart, key facts, dividends, news, put ideas, compare."""
import altair as alt
import pandas as pd
import streamlit as st

import calc
import engine as E
import markets as MK
from common import _show, esc, big_number, category_bar, kpi, line_chart, live_panel, md, pct, updated_caption


def _pick_symbol():
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Search any stock, ETF, index or crypto", key="lk_query",
                      placeholder="Apple, TSLA, S&P 500, bitcoin, XLK…")
    c2.selectbox("Chart range", list(MK.PERIODS), index=4, key="lk_period")
    if not q.strip():
        return None
    results = MK.search_symbols(q)
    options = [(r["symbol"], f"{r['symbol']} — {r['name']} · {r['type'].title()} · {r['exchange']}") for r in results]
    raw = q.strip().upper().replace(" ", "")
    if raw and len(raw) <= 10 and not any(s == raw for s, _ in options):
        options.append((raw, f"{raw} (use exactly as a ticker)"))
    if not options:
        st.warning("No matches. Try a ticker symbol like AAPL.")
        return None
    labels = [lbl for _, lbl in options]
    choice = st.selectbox("Matches", labels, key="lk_choice", label_visibility="collapsed")
    return options[labels.index(choice)][0] if choice in labels else options[0][0]


def _range_bar(low, high, price):
    if not (low and high and high > low):
        return
    pos = min(max((price - low) / (high - low), 0), 1)
    st.progress(pos, text=f"52-week range: \\${low:,.2f} — \\${high:,.2f}  ·  now at {pos * 100:.0f}% of the range")


def _header(symbol, quote, prof):
    name = prof.get("longName") or prof.get("shortName") or symbol
    st.subheader(f"{name} ({symbol})")
    bits = [prof.get("sector"), prof.get("industry"), prof.get("quoteType", "").title() or None]
    st.caption(" · ".join(b for b in bits if b))
    price = float(quote["Price"])
    c = st.columns(5)
    c[0].metric("Price", f"${price:,.2f}", f"{quote['Day %']:+.2f}% today")
    c[1].metric("Market cap", big_number(prof.get("marketCap")))
    pe, fpe = prof.get("trailingPE"), prof.get("forwardPE")
    kpi(c[2], "P/E (trailing)", f"{pe:.1f}" if pe else "—", f"forward {fpe:.1f}" if fpe else None)
    rate = MK.forward_dividend(symbol, prof)
    dy = rate / price * 100 if rate and price else 0.0
    kpi(c[3], "Dividend yield", f"{dy:.2f}%" if dy else "None", f"${rate:.2f}/yr" if dy else None)
    beta = prof.get("beta")
    kpi(c[4], "Beta", f"{beta:.2f}" if beta else "—", "vs market (1.0)")
    ext, _ = MK.get_extended_prices((symbol,))
    if not ext.empty:
        e = ext.iloc[0]
        st.markdown(esc(f"**{e['Session']}: ${e['Ext Price']:,.2f}** ({e['Ext %']:+.2f}% vs the ${e['Close']:,.2f} close) "
                        f"· last trade {e['Ext Time']} New York time. Options don't trade in this session."))
    _range_bar(prof.get("fiftyTwoWeekLow"), prof.get("fiftyTwoWeekHigh"), price)
    tgt, rec = prof.get("targetMeanPrice"), prof.get("recommendationKey")
    if tgt:
        up = (tgt / price - 1) * 100
        n = prof.get("numberOfAnalystOpinions")
        st.caption(f"Analyst average target ${tgt:,.2f} ({up:+.1f}% vs price)"
                   f"{f' · consensus: {rec}' if rec else ''}{f' · {n} analysts' if n else ''}")


def _chart(symbol, period_label):
    df, fetched = MK.get_history(symbol, period_label)
    if df.empty:
        st.info("No price history for that range.")
        return
    close = df["Close"].dropna()
    chg = (close.iloc[-1] / close.iloc[0] - 1) * 100
    daily = period_label in ("1M", "6M", "1Y")
    plot = pd.DataFrame({"Price": close})
    if daily:
        full, _ = MK.get_history(symbol, "1Y")
        fc = full["Close"].dropna()
        for n in (50, 200):
            if len(fc) >= n:
                plot[f"SMA {n}"] = fc.rolling(n).mean().reindex(close.index)
    if plot.index.tz is not None:
        plot.index = plot.index.tz_localize(None)
    st.caption(f"{period_label} change: **{chg:+.2f}%**" + ("  ·  lines: price, 50-day and 200-day average" if daily else ""))
    line_chart(plot, height=340)
    if "Volume" in df and df["Volume"].sum() > 0 and period_label != "Max":
        vol = df["Volume"].copy()
        if vol.index.tz is not None:
            vol.index = vol.index.tz_localize(None)
        vdf = vol.rename("Volume").rename_axis("Date").reset_index()
        _show(alt.Chart(vdf).mark_bar(color="#7a8aa0", opacity=0.7).encode(
            x=alt.X("Date:T", axis=None), y=alt.Y("Volume:Q", title=None, axis=alt.Axis(format="~s")),
            tooltip=["Date:T", alt.Tooltip("Volume:Q", format=",.0f")]).properties(height=80))


def _technicals(symbol):
    rsi, trend, close = MK.rsi_and_trend(symbol)
    c = st.columns(4)
    c[0].metric("Trend", trend)
    kpi(c[1], "RSI (14)", f"{rsi:.0f}" if rsi == rsi else "—",
        "oversold (<30)" if rsi == rsi and rsi < 30 else "overbought (>70)" if rsi == rsi and rsi > 70 else "neutral (30-70)")
    if len(close) >= 60:
        c[2].metric("50-day average", f"${close.tail(50).mean():,.2f}")
        if len(close) >= 200:
            c[3].metric("200-day average", f"${close.tail(200).mean():,.2f}")


def _overview(prof):
    if prof.get("longBusinessSummary"):
        with st.expander("About this company", expanded=False):
            st.write(prof["longBusinessSummary"])
    facts = {
        "Sector": prof.get("sector"), "Industry": prof.get("industry"), "Country": prof.get("country"),
        "Employees": f"{prof['fullTimeEmployees']:,}" if prof.get("fullTimeEmployees") else None,
        "Price / Book": f"{prof['priceToBook']:.2f}" if prof.get("priceToBook") else None,
        "Profit margin": pct(prof["profitMargins"] * 100, 1, False) if prof.get("profitMargins") else None,
        "Return on equity": pct(prof["returnOnEquity"] * 100, 1, False) if prof.get("returnOnEquity") else None,
        "Revenue growth": pct(prof["revenueGrowth"] * 100, 1) if prof.get("revenueGrowth") else None,
        "Earnings growth": pct(prof["earningsGrowth"] * 100, 1) if prof.get("earningsGrowth") else None,
        "Debt / Equity": f"{prof['debtToEquity']:.0f}" if prof.get("debtToEquity") else None,
        "Avg volume": f"{prof['averageVolume']:,.0f}" if prof.get("averageVolume") else None,
        "Fund category": prof.get("category"),
        "Fund assets": big_number(prof.get("totalAssets")) if prof.get("totalAssets") else None,
        "Expense ratio": pct(prof["annualReportExpenseRatio"] * 100, 2, False) if prof.get("annualReportExpenseRatio") else None,
        "Website": prof.get("website"),
    }
    facts = {k: v for k, v in facts.items() if v}
    if facts:
        st.dataframe(pd.DataFrame({"Fact": list(facts), "Value": [str(v) for v in facts.values()]}),
                     hide_index=True, width="stretch")
    lo, mean, hi = prof.get("targetLowPrice"), prof.get("targetMeanPrice"), prof.get("targetHighPrice")
    if mean:
        md(f"**Analyst price targets:** low ${lo:,.2f} · average ${mean:,.2f} · high ${hi:,.2f}" if lo and hi else
           f"**Analyst average target:** ${mean:,.2f}")


def _dividends(symbol, price, prof):
    div, fetched = MK.get_dividend_history(symbol)
    stats = MK.dividend_stats(div, price)
    if not stats["has_dividends"]:
        st.info(f"{symbol} has no dividend history on Yahoo (it may not pay one).")
        return
    fwd = MK.forward_dividend(symbol, prof)
    c = st.columns(5)
    kpi(c[0], "Paid, last 12 months", f"${stats['ttm']:.2f}", f"{stats['count']} payments · {stats['yield_pct']:.2f}% of price")
    kpi(c[1], "Current annual rate", f"${fwd:.2f}" if fwd else "—",
        f"{fwd / price * 100:.2f}% yield going forward" if fwd and price else None)
    kpi(c[2], "Paid", stats["frequency"], f"last: ${stats['last']:.2f}")
    g = stats["growth_cagr"]
    kpi(c[3], "Dividend growth", f"{g:.1f}%/yr" if g == g else "—", "5-year compound rate")
    kpi(c[4], "Increase streak", f"{stats['streak']} yrs", "consecutive yearly raises")
    if fwd and stats["ttm"] and abs(fwd / stats["ttm"] - 1) > 0.15:
        st.caption("The current annual rate differs from what was actually paid over the last 12 months because the "
                   "company recently changed its dividend. The current rate is the better guide to future income.")
    yearly = stats["by_year"].tail(15)
    st.markdown("**Dividends paid per share, by year** (the current year is partial)")
    category_bar(pd.DataFrame({"Year": [str(y) for y in yearly.index], "Dividend per share": yearly.values}),
                 "Year", "Dividend per share", order=[str(y) for y in yearly.index], fmt="$,.2f", height=240)
    last = div.tail(12).iloc[::-1].copy()
    last["Date"] = last["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(last, hide_index=True, width="stretch",
                 column_config={"Dividend": st.column_config.NumberColumn(format="$%.4f")})
    if st.button("Load into the Income calculators", key="lk_to_income"):
        _preload_income(symbol)
        st.success("Loaded. Open the **Income** tab: the calculators are pre-filled with this stock.")
    updated_caption(fetched, "dividend history updates hourly")


def _preload_income(symbol):
    """Fill the Income tab's shared inputs from a ticker (also used by the Income tab itself)."""
    quote, _ = MK.get_quotes((symbol,))
    if quote.empty:
        return False
    price = float(quote.iloc[0]["Price"])
    prof, _ = MK.get_profile(symbol)
    div, _ = MK.get_dividend_history(symbol)
    stats = MK.dividend_stats(div, price)
    dps = stats["ttm"] if stats.get("has_dividends") else MK.annual_dividend_rate(prof)
    hist, _ = MK.get_history(symbol, "5Y")
    close = hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float)
    pg = float("nan")
    if len(close) > 52:
        years = (close.index[-1] - close.index[0]).days / 365.25
        pg = calc.cagr(float(close.iloc[0]), float(close.iloc[-1]), years)
    dg = stats.get("growth_cagr", float("nan")) if stats.get("has_dividends") else float("nan")
    st.session_state.update({
        "inc_symbol": symbol, "inc_price": round(price, 2), "inc_dps": round(float(dps), 4),
        "inc_freq": stats.get("frequency", "Quarterly") if stats.get("has_dividends") else "Quarterly",
        "inc_div_growth": round(min(max(dg, 0.0), 12.0), 1) if dg == dg else 3.0,
        "inc_price_growth": round(min(max(pg, 0.0), 8.0), 1) if pg == pg else 6.0,
        "inc_loaded_at": MK._now(),
    })
    return True


def _news(symbol):
    items, fetched = MK.get_news(symbol)
    if not items:
        st.info("No recent headlines from Yahoo for this symbol.")
        return
    for it in items:
        when = it["published"].strftime("%b %d, %H:%M") if it["published"] else ""
        title = f"[{it['title']}]({it['link']})" if it["link"] else it["title"]
        md(f"**{title}**  \n{it['publisher']} · {when}")
        if it["summary"]:
            st.caption(it["summary"][:240] + ("…" if len(it["summary"]) > 240 else ""))
    updated_caption(fetched, "headlines refresh every 15 min")


def _put_ideas(symbol):
    st.caption("Runs the Put Scanner on just this ticker, ignoring your cash limit, so you can see what is on offer.")
    c = st.columns(3)
    dte = c[0].slider("Days to expiration", 1, 90, (4, 45), key="lk_dte")
    win = c[1].slider("Win probability %", 50, 99, (75, 97), key="lk_win")
    skip_earn = c[2].checkbox("Exclude earnings in window", value=False, key="lk_earn")
    key = f"lk_puts_{symbol}"
    if st.button("Find cash-secured put ideas", key="lk_find"):
        params = E.ScanParams(risk_free_rate=E.get_risk_free_rate(), min_dte=dte[0], max_dte=dte[1],
                              win_prob_min=win[0], win_prob_max=win[1], min_oi=50, max_spread_pct=25, top_n=12,
                              min_hist_win=0, min_premium_usd=5, exclude_earnings=skip_earn)
        with st.spinner("Scanning option chains…"):
            df, spot, counts = E.scan_ticker_for_csp(symbol, params)
        st.session_state[key] = (df, E.empty_reason(symbol, spot, counts, params) if df.empty else "", MK._now())
    if key in st.session_state:
        df, why, ts = st.session_state[key]
        if df.empty:
            st.info(why or "No candidates found.")
        else:
            cols = ["Expiration", "DTE", "Strike", "% OTM", "Mid", "Premium $", "Delta", "Win % (cons.)",
                    "Annualized Yield %", "Edge $", "Capital Req. $", "Earnings Alert"]
            st.dataframe(df[cols], hide_index=True, width="stretch", column_config={
                "Strike": st.column_config.NumberColumn(format="$%g"),
                "% OTM": st.column_config.NumberColumn(format="%.1f%%"),
                "Mid": st.column_config.NumberColumn(format="$%.2f"),
                "Premium $": st.column_config.NumberColumn("Premium", format="$%d"),
                "Win % (cons.)": st.column_config.NumberColumn("Win (cons.)", format="%.1f%%"),
                "Annualized Yield %": st.column_config.NumberColumn("Ann. yield", format="%.1f%%"),
                "Edge $": st.column_config.NumberColumn("Edge ($)", format="%d"),
                "Capital Req. $": st.column_config.NumberColumn("Cash needed", format="$%d"),
                "Earnings Alert": st.column_config.TextColumn("Earnings"),
            })
        updated_caption(ts, "option prices from Yahoo")


def _compare(symbol):
    extra = st.text_input("Compare with (comma-separated tickers)", value="SPY, QQQ", key="lk_cmp")
    per = st.radio("Period", ["6mo", "1y", "5y"], index=1, horizontal=True, key="lk_cmp_period")
    syms = tuple(dict.fromkeys([symbol] + [s.strip().upper() for s in extra.split(",") if s.strip()]))[:8]
    mat, fetched = MK.get_price_matrix(syms, per)
    if mat.empty:
        st.info("No data to compare.")
        return
    norm = mat / mat.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else 1) * 100
    st.caption("Every line starts at 100, so you can compare growth directly.")
    line_chart(norm, height=340)
    end = (norm.iloc[-1] - 100).sort_values(ascending=False)
    st.dataframe(pd.DataFrame({"Symbol": end.index, f"Return over {per}": end.values}), hide_index=True,
                 width="stretch", column_config={f"Return over {per}": st.column_config.NumberColumn(format="%+.1f%%")})
    updated_caption(fetched)


def _panel():
    st.subheader("Stock lookup")
    symbol = _pick_symbol()
    if not symbol:
        st.info("Type a company name or ticker above to see its price, chart, dividends, news and more.")
        return
    quotes, qts = MK.get_quotes((symbol,))
    if quotes.empty:
        st.warning(f"Could not get a price for {symbol}. Check the symbol or try again shortly.")
        return
    quote = quotes.iloc[0]
    prof, _ = MK.get_profile(symbol)
    _header(symbol, quote, prof)
    t_chart, t_over, t_div, t_news, t_put, t_cmp = st.tabs(
        ["Chart", "Overview", "Dividends", "News", "Put ideas", "Compare"])
    with t_chart:
        _chart(symbol, st.session_state.get("lk_period", "1Y"))
        _technicals(symbol)
        updated_caption(qts)
    with t_over:
        _overview(prof)
    with t_div:
        _dividends(symbol, float(quote["Price"]), prof)
    with t_news:
        _news(symbol)
    with t_put:
        _put_ideas(symbol)
    with t_cmp:
        _compare(symbol)


def render():
    live_panel(_panel)
