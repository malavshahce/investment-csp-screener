"""Market overview tab: indices, VIX, rates & commodities, sector performance and top movers."""
import pandas as pd
import streamlit as st

import engine as E
import markets as MK
from common import live_panel, md, signed_bar, updated_caption

_ALL = tuple(dict.fromkeys(list(MK.INDICES.values()) + list(MK.MACRO.values()) +
                           list(MK.SECTORS.values()) + E.LIQUID_STOCKS))


def _tile(col, label, quotes, symbol, decimals=2, inverse=False, prefix=""):
    row = quotes[quotes["Symbol"] == symbol]
    if row.empty:
        col.metric(label, "—")
        return
    r = row.iloc[0]
    col.metric(label, f"{prefix}{r['Price']:,.{decimals}f}", f"{r['Day %']:+.2f}%",
               delta_color="inverse" if inverse else "normal")


def _mover_table(df, title):
    st.markdown(f"**{title}**")
    st.dataframe(
        df[["Symbol", "Price", "Day %", "5D %"]], hide_index=True, width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Day %": st.column_config.NumberColumn("Day", format="%+.2f%%"),
            "5D %": st.column_config.NumberColumn("5 days", format="%+.2f%%"),
        })


def _panel():
    st.subheader("Market overview")
    with st.spinner("Loading market data…"):
        quotes, fetched = MK.get_quotes(_ALL)
    updated_caption(fetched)
    if quotes.empty:
        st.warning("Could not download market data right now (Yahoo may be rate-limiting). "
                   "Press Refresh in the header in a minute.")
        return

    regime, _ = MK.get_regime()
    if regime["vix"] is not None:
        spy = f" · S&P 500: {regime['spy_trend']}" if regime["spy_trend"] else ""
        banner = f"**VIX {regime['vix']:.1f} ({regime['level']}){spy}.** {regime['advice']}"
        {"calm": st.info, "normal": st.success, "elevated": st.warning, "high": st.error}.get(
            regime["level"], st.info)(banner)

    cols = st.columns(len(MK.INDICES))
    for c, (label, sym) in zip(cols, MK.INDICES.items()):
        _tile(c, label, quotes, sym, inverse=(sym == "^VIX"))

    cols = st.columns(len(MK.MACRO))
    for c, (label, sym) in zip(cols, MK.MACRO.items()):
        dec = 3 if sym in ("^TNX", "^IRX") else 2
        _tile(c, label, quotes, sym, decimals=dec, prefix="$" if sym in ("GC=F", "CL=F", "BTC-USD") else "")

    st.divider()
    left, right = st.columns([3, 2])
    q = quotes.set_index("Symbol")
    sec = pd.DataFrame([{"Sector": name, "ETF": sym, "Day %": q.loc[sym, "Day %"], "5D %": q.loc[sym, "5D %"],
                         "1M %": q.loc[sym, "1M %"]} for name, sym in MK.SECTORS.items() if sym in q.index])
    with left:
        st.markdown("**Sector performance today**")
        if not sec.empty:
            signed_bar(sec, "Sector", "Day %", height=360)
    with right:
        st.markdown("**Sectors: day, 5 days, 1 month**")
        if not sec.empty:
            st.dataframe(sec.sort_values("1M %", ascending=False), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%+.1f%%")
                                        for c in ("Day %", "5D %", "1M %")})
            best, worst = sec.sort_values("Day %").iloc[-1], sec.sort_values("Day %").iloc[0]
            st.caption(f"Leading today: {best['Sector']} ({best['Day %']:+.2f}%). "
                       f"Lagging: {worst['Sector']} ({worst['Day %']:+.2f}%).")

    st.divider()
    st.markdown("**Top movers among ~80 of the most-traded stocks**")
    stocks = quotes[quotes["Symbol"].isin(E.LIQUID_STOCKS)].copy()
    if stocks.empty:
        return
    stocks["Dollar Vol"] = stocks["Price"] * stocks["Volume"].fillna(0)
    a, b, c = st.columns(3)
    with a:
        _mover_table(stocks.sort_values("Day %", ascending=False).head(8), "Top gainers today")
    with b:
        _mover_table(stocks.sort_values("Day %").head(8), "Top decliners today")
    with c:
        _mover_table(stocks.sort_values("Dollar Vol", ascending=False).head(8), "Most active (by dollar volume)")
    st.caption("Click a column header to sort. Search any ticker in the Lookup tab.")


def render():
    live_panel(_panel)
