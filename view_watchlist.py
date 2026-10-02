"""Watchlist tab: live prices for the stocks you follow, with price alerts and upcoming events."""
import numpy as np
import pandas as pd
import streamlit as st

import markets as MK
import portfolio as PF
from common import esc, live_panel, rerun_fragment, updated_caption


def _alert_status(price, above, below):
    if above and above > 0 and price >= above:
        return f"🔔 Above ${above:,.2f}"
    if below and below > 0 and price <= below:
        return f"🔔 Below ${below:,.2f}"
    return ""


def _add_form(wl):
    with st.form("wl_add", clear_on_submit=True):
        c = st.columns([2, 1.2, 1.2, 2, 1])
        t = c[0].text_input("Add a ticker", placeholder="e.g. KO")
        a = c[1].number_input("Alert above ($)", min_value=0.0, value=0.0, step=1.0, help="0 = no alert")
        b = c[2].number_input("Alert below ($)", min_value=0.0, value=0.0, step=1.0, help="0 = no alert")
        n = c[3].text_input("Note", placeholder="optional")
        c[4].write("")
        if c[4].form_submit_button("Add", type="primary", width="stretch") and t.strip():
            sym = PF._norm_ticker(t)
            matched, _ = MK.resolve_symbols((sym,))
            sym = matched.get(sym) or ""
            if not sym:
                st.session_state["wl_msg"] = (f"Could not find a price for '{PF._norm_ticker(t)}'. Canadian stocks need a "
                                              "suffix (RY.TO for the TSX, .V for TSX Venture); US class shares use a "
                                              "dash (BRK-B).")
            else:
                new = pd.DataFrame([{"ticker": sym, "alert_above": a or np.nan,
                                     "alert_below": b or np.nan, "note": n}])
                PF.save_watchlist(pd.concat([wl, new], ignore_index=True))
                st.session_state.pop("wl_msg", None)
            rerun_fragment()


def _panel():
    st.subheader("Watchlist")
    wl = PF.load_watchlist()
    _add_form(wl)
    if st.session_state.get("wl_msg"):
        st.warning(st.session_state["wl_msg"])
    if wl.empty:
        st.info("Your watchlist is empty. Add a ticker above. Set an alert price to be flagged when it crosses.")
        return

    symbols = tuple(wl["ticker"])
    quotes, fetched = MK.get_quotes(symbols)
    profiles, _ = MK.get_profiles(symbols)
    q = quotes.set_index("Symbol") if not quotes.empty else pd.DataFrame()

    rows = []
    for _, r in wl.iterrows():
        s = r["ticker"]
        if s not in q.index:
            rows.append({"Ticker": s, "Name": "(no quote)", "Note": r["note"]})
            continue
        price = float(q.loc[s, "Price"])
        p = profiles.get(s, {})
        lo, hi, tgt = p.get("fiftyTwoWeekLow"), p.get("fiftyTwoWeekHigh"), p.get("targetMeanPrice")
        rows.append({
            "Ticker": s, "Name": p.get("shortName") or s, "Cur.": p.get("currency") or "", "Price": price, "Day %": float(q.loc[s, "Day %"]),
            "5D %": float(q.loc[s, "5D %"]), "1M %": float(q.loc[s, "1M %"]),
            "1M trend": q.loc[s, "Trend (1M)"],
            "52W position %": (price - lo) / (hi - lo) * 100 if lo and hi and hi > lo else np.nan,
            "Analyst upside %": (tgt / price - 1) * 100 if tgt else np.nan,
            "Div Yield %": MK.forward_dividend(s, p) / price * 100,
            "Alert": _alert_status(price, r["alert_above"], r["alert_below"]), "Note": r["note"],
        })
    df = pd.DataFrame(rows)

    triggered = df[df.get("Alert", pd.Series(dtype=str)).fillna("") != ""] if "Alert" in df else pd.DataFrame()
    if not triggered.empty:
        st.warning(esc("**Price alerts:** " + " · ".join(f"{r.Ticker} {r.Alert}" for r in triggered.itertuples())))
        seen = st.session_state.setdefault("wl_notified", set())
        for r in triggered.itertuples():
            key = (r.Ticker, r.Alert)
            if key not in seen:
                st.toast(esc(f"{r.Ticker}: {r.Alert} (now ${r.Price:,.2f})"), icon="🔔")
                seen.add(key)

    def tint(v):
        if v is None or v != v:
            return ""
        return "color: #1fb76a" if v > 0 else "color: #ef5350" if v < 0 else ""

    st.dataframe(
        df.style.map(tint, subset=[c for c in ("Day %", "5D %", "1M %") if c in df]),
        hide_index=True, width="stretch", column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Day %": st.column_config.NumberColumn("Day", format="%+.2f%%"),
            "5D %": st.column_config.NumberColumn("5 days", format="%+.2f%%"),
            "1M %": st.column_config.NumberColumn("1 month", format="%+.2f%%"),
            "1M trend": st.column_config.LineChartColumn("1M trend"),
            "52W position %": st.column_config.ProgressColumn("52-wk range", format="%.0f%%", min_value=0, max_value=100,
                                                              help="0% = at the 52-week low, 100% = at the high"),
            "Analyst upside %": st.column_config.NumberColumn("Analyst upside", format="%+.1f%%"),
            "Div Yield %": st.column_config.NumberColumn("Yield", format="%.2f%%"),
        })
    updated_caption(fetched)

    c1, c2 = st.columns(2)
    with c1.expander("Edit list & alerts"):
        edited = st.data_editor(wl, num_rows="dynamic", hide_index=True, width="stretch", key="wl_editor",
                                column_config={
                                    "ticker": st.column_config.TextColumn("Ticker"),
                                    "alert_above": st.column_config.NumberColumn("Alert above ($)", format="$%.2f"),
                                    "alert_below": st.column_config.NumberColumn("Alert below ($)", format="$%.2f"),
                                    "note": st.column_config.TextColumn("Note")})
        if st.button("Save watchlist", type="primary", key="wl_save"):
            PF.save_watchlist(edited)
            st.session_state["wl_notified"] = set()
            rerun_fragment()
        st.download_button("Download watchlist backup (CSV)", wl.to_csv(index=False).encode("utf-8"),
                           file_name="watchlist.csv", mime="text/csv", key="wl_dl")
    with c2.expander("Upcoming earnings & ex-dividend dates"):
        if st.button("Load dates for my watchlist", key="wl_events"):
            with st.spinner("Checking calendars…"):
                ev, ts = MK.get_events(symbols)
            st.session_state["wl_events_data"] = (ev, ts)
        if "wl_events_data" in st.session_state:
            ev, ts = st.session_state["wl_events_data"]
            ev = ev.copy()
            for c in ("Next Earnings", "Ex-Dividend"):
                ev[c] = ev[c].astype(str).replace({"None": "—"})
            st.dataframe(ev.sort_values("Next Earnings"), hide_index=True, width="stretch")
            updated_caption(ts, "calendars refresh every 15 min")
    st.caption("Alerts are checked each time this tab refreshes (turn on auto-refresh in the header). "
               "They show on screen only; the app does not send emails or notifications while closed.")


def render():
    live_panel(_panel)
