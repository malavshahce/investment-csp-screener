"""Private portfolio tab: holdings, live value, gain/loss, allocation, dividend income and DRIP projection."""
import io

import numpy as np
import pandas as pd
import streamlit as st

import calc
import markets as MK
import portfolio as PF
from common import big_number, category_bar, kpi, line_chart, live_panel, md, money, rerun_fragment, signed_bar, updated_caption

TEMPLATE = "ticker,shares,avg_cost,account,notes\nAAPL,10,150.00,Brokerage,example row\nKO,25,58.40,Roth IRA,\n"


def _manage(holdings, on_cloud, has_password):
    with st.expander("Manage holdings (add, edit, import, export)", expanded=holdings.empty):
        if on_cloud and not has_password:
            st.error("This app is online without a password. Set `app_password` in the app's Secrets before entering "
                     "real holdings, or anyone with the link can see them.")
        st.caption("Your holdings are saved only in `portfolio.csv` next to the app (never uploaded to GitHub). "
                   "Edit the table, or import a CSV exported from your broker.")
        edited = st.data_editor(
            holdings, num_rows="dynamic", hide_index=True, width="stretch", key="pf_editor",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "shares": st.column_config.NumberColumn("Shares", format="%.4f"),
                "avg_cost": st.column_config.NumberColumn("Avg cost / share ($)", format="$%.2f"),
                "account": st.column_config.TextColumn("Account"),
                "notes": st.column_config.TextColumn("Notes"),
            })
        c = st.columns(3)
        if c[0].button("Save holdings", type="primary", key="pf_save"):
            PF.save_holdings(edited)
            st.success("Saved.")
            rerun_fragment()
        c[1].download_button("Download backup (CSV)", holdings.to_csv(index=False).encode("utf-8"),
                             file_name="portfolio_backup.csv", mime="text/csv", key="pf_dl")
        c[2].download_button("CSV template", TEMPLATE.encode("utf-8"), file_name="portfolio_template.csv",
                             mime="text/csv", key="pf_tpl")

        up = st.file_uploader("Import a CSV (columns like Symbol, Quantity, Average Cost are recognised)",
                              type="csv", key="pf_upload")
        mode = st.radio("When importing", ["Replace my holdings", "Add to my holdings"], horizontal=True, key="pf_mode")
        if up is not None and st.button("Import this file", key="pf_import"):
            try:
                new, missing = PF.normalize_import(pd.read_csv(io.BytesIO(up.getvalue())))
                if missing:
                    st.error("Could not find these columns: " + ", ".join(missing) +
                             ". Use the template for the expected names.")
                else:
                    base = holdings if mode.startswith("Add") else pd.DataFrame(columns=PF.HOLDING_COLUMNS)
                    PF.save_holdings(pd.concat([base, new], ignore_index=True))
                    st.success(f"Imported {len(new)} rows.")
                    rerun_fragment()
            except Exception as e:
                st.error(f"Could not read that file: {e}")


def _tint(v):
    if v is None or v != v:
        return ""
    return "color: #1fb76a" if v > 0 else "color: #ef5350" if v < 0 else ""


def _summary_row(sm):
    c = st.columns(5)
    kpi(c[0], "Portfolio value", money(sm["value"]), f"cost {money(sm['cost'])}")
    c[1].metric("Today", money(sm["day"]), f"{sm['day_pct']:+.2f}%")
    c[2].metric("Total gain / loss", money(sm["gain"]), f"{sm['gain_pct']:+.1f}%" if sm["gain_pct"] == sm["gain_pct"] else None)
    kpi(c[3], "Dividends per year", money(sm["annual_div"]), f"{sm['yield_pct']:.2f}% yield")
    kpi(c[4], "Portfolio beta", f"{sm['beta']:.2f}" if sm["beta"] == sm["beta"] else "—",
        f"{sm['payers']} of {sm['positions']} pay dividends")


def _holdings_table(df):
    compact = ["Ticker", "Name", "Shares", "Price", "Value", "Weight %", "Day %", "Gain $", "Gain %", "Yield %",
               "Annual Div $", "Sector"]
    full = ["Ticker", "Name", "Account", "Shares", "Avg Cost", "Price", "Value", "Weight %", "Day $", "Day %",
            "Gain $", "Gain %", "Yield %", "Annual Div $", "Yield on Cost %", "Sector", "Beta"]
    more = st.toggle("Show all columns", key="pf_all_cols", help="Adds account, average cost, today in dollars, "
                     "yield on cost and beta.")
    cols = [c for c in (full if more else compact) if c in df]
    tint_cols = [c for c in ("Day $", "Day %", "Gain $", "Gain %") if c in cols]
    st.dataframe(
        df[cols].style.map(_tint, subset=tint_cols), hide_index=True, width="stretch",
        column_config={
            "Shares": st.column_config.NumberColumn(format="%.3f"),
            "Avg Cost": st.column_config.NumberColumn(format="$%.2f"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Value": st.column_config.NumberColumn(format="$%.0f"),
            "Weight %": st.column_config.ProgressColumn("Weight", format="%.1f%%", min_value=0, max_value=100),
            "Day $": st.column_config.NumberColumn("Today ($)", format="%.0f"),
            "Day %": st.column_config.NumberColumn("Today", format="%+.2f%%"),
            "Gain $": st.column_config.NumberColumn("Gain ($)", format="%.0f"),
            "Gain %": st.column_config.NumberColumn("Gain", format="%+.1f%%"),
            "Yield %": st.column_config.NumberColumn("Yield", format="%.2f%%"),
            "Annual Div $": st.column_config.NumberColumn("Dividends / yr", format="$%.0f"),
            "Yield on Cost %": st.column_config.NumberColumn("Yield on cost", format="%.2f%%"),
            "Beta": st.column_config.NumberColumn(format="%.2f"),
        })


def _charts(df, holdings):
    total = df["Value"].sum()
    by_t = (df.groupby("Ticker")["Value"].sum() / total * 100).sort_values(ascending=False)
    sec = (df.groupby("Sector")["Value"].sum() / total * 100).sort_values(ascending=False)
    a, b = st.columns(2)
    with a:
        st.markdown("**By holding (% of portfolio)**")
        category_bar(by_t.rename("Weight").rename_axis("Holding").reset_index(), "Holding", "Weight",
                     order=list(by_t.index), fmt=".1f")
    with b:
        st.markdown("**By sector (% of portfolio)**")
        category_bar(sec.rename("Weight").rename_axis("Sector").reset_index(), "Sector", "Weight",
                     order=list(sec.index), fmt=".1f", horizontal=True, height=max(120, 42 * len(sec)))

    st.markdown("**Value over the past year vs the S&P 500**")
    st.caption("Uses today's share counts for the whole year (it shows how this mix performed, "
               "not your actual account history).")
    syms = tuple(dict.fromkeys(list(df["Ticker"]) + ["SPY"]))
    mat, ts = MK.get_price_matrix(syms, "1y")
    if mat.empty or "SPY" not in mat:
        st.info("Not enough price history to draw the chart.")
        return
    shares = df.groupby("Ticker")["Shares"].sum()
    held = [t for t in shares.index if t in mat.columns]
    value = (mat[held] * shares[held]).sum(axis=1)
    out = pd.DataFrame({"Your portfolio": value / value.iloc[0] * 100, "S&P 500 (SPY)": mat["SPY"] / mat["SPY"].iloc[0] * 100})
    line_chart(out, height=300)
    r1 = out.iloc[-1] - 100
    st.caption(f"1-year: your mix {r1.iloc[0]:+.1f}% vs S&P 500 {r1.iloc[1]:+.1f}%.")
    updated_caption(ts, "prices refresh every 10 min")


def _income(df, sm):
    payers = df[df["Annual Div $"] > 0].copy()
    payers = payers.assign(**{"Cost": payers["Cost"].fillna(payers["Value"])})   # unknown cost: treat value as cost
    if payers.empty:
        st.info("None of your holdings pay a dividend, so there is no income to project.")
        return
    c = st.columns(4)
    c[0].metric("Per year", money(sm["annual_div"]))
    c[1].metric("Per month", money(sm["annual_div"] / 12))
    c[2].metric("Portfolio yield", f"{sm['yield_pct']:.2f}%")
    c[3].metric("Yield on cost", f"{sm['yoc_pct']:.2f}%" if sm["yoc_pct"] == sm["yoc_pct"] else "—")
    st.markdown("**Dividend income by holding (per year)**")
    by_pay = payers.groupby("Ticker")["Annual Div $"].sum().sort_values(ascending=False)
    category_bar(by_pay.rename("Dividends").rename_axis("Holding").reset_index(), "Holding", "Dividends",
                 order=list(by_pay.index), fmt="$,.0f")

    st.markdown("**Project this portfolio's dividends (with and without reinvesting)**")
    c = st.columns(5)
    years = c[0].slider("Years", 1, 40, 15, key="pf_years")
    dg = c[1].number_input("Dividend growth %/yr", min_value=-10.0, max_value=30.0, value=5.0, step=0.5, key="pf_dg")
    pg = c[2].number_input("Price growth %/yr", min_value=-10.0, max_value=30.0, value=6.0, step=0.5, key="pf_pg")
    contrib = c[3].number_input("Add per month ($)", min_value=0.0, value=0.0, step=100.0, key="pf_contrib")
    tax = c[4].number_input("Tax on dividends %", min_value=0.0, max_value=60.0, value=0.0, step=1.0, key="pf_tax")
    base_value = float(payers["Value"].sum())
    y = float(payers["Annual Div $"].sum()) / base_value * 100          # blended yield of the dividend payers
    kw = dict(initial_shares=base_value / 100, price=100.0, annual_dps=y, freq_per_year=4, div_growth_pct=dg,
              price_growth_pct=pg, years=years, monthly_contribution=contrib, tax_rate_pct=tax,
              start_cost_basis=float(payers["Cost"].sum()))
    drip, plain = calc.compare_drip(**kw)
    d, p = drip.iloc[-1], plain.iloc[-1]
    m = st.columns(4)
    m[0].metric(f"Value in {years} yrs (DRIP)", money(d["Total"]), f"{money(d['Total'] - p['Total'])} vs not reinvesting")
    kpi(m[1], "Yearly income then (DRIP)", money(d["AnnualIncome"]), f"{money(d['AnnualIncome'] / 12)}/month")
    m[2].metric("Dividends received", money(d["CumDividends"]))
    m[3].metric("Yield on cost then", f"{d['YieldOnCost']:.1f}%")
    line_chart(pd.DataFrame({"Reinvest (DRIP)": drip["Total"], "Take as cash": plain["Total"]}, index=drip["Year"]), zero=True,
                  height=260)
    st.caption("Treats your dividend-paying holdings as one blended position (quarterly payments). "
               "An illustration, not a forecast.")


def _events(df):
    if st.button("Load upcoming earnings & ex-dividend dates for my holdings", key="pf_events"):
        with st.spinner("Checking calendars…"):
            st.session_state["pf_events_data"] = MK.get_events(tuple(df["Ticker"]))
    if "pf_events_data" in st.session_state:
        ev, ts = st.session_state["pf_events_data"]
        ev = ev.copy()
        for c in ("Next Earnings", "Ex-Dividend"):
            ev[c] = ev[c].astype(str).replace({"None": "—"})
        st.dataframe(ev.sort_values("Next Earnings"), hide_index=True, width="stretch")
        updated_caption(ts, "calendars refresh every 15 min")


def _make_panel(on_cloud, has_password):
    def _panel():
        st.subheader("My portfolio (private)")
        top = st.columns([1, 3])
        hide = top[0].toggle("Hide amounts", key="pf_hide", help="Blurs the numbers: handy when someone is watching your screen.")
        top[1].caption("Only you can see this tab: it sits behind your app password and your data stays in a local file.")
        if hide:
            st.markdown("<style>.st-key-pf_values{filter:blur(9px);user-select:none;}</style>", unsafe_allow_html=True)

        holdings = PF.load_holdings()
        _manage(holdings, on_cloud, has_password)
        if holdings.empty:
            st.info("Add your holdings above (ticker, shares, average cost) to see live value, gains, dividends and more.")
            return

        df, fetched = PF.value_holdings(holdings)
        if df.empty or "Value" not in df or df["Value"].notna().sum() == 0:
            st.warning("Could not get prices for your holdings right now. Try Refresh in a minute.")
            return
        bad = df[df["Value"].isna()]["Ticker"].tolist()
        if bad:
            st.warning("No quote for: " + ", ".join(bad) + ". Check the ticker symbols (class shares use a dash, e.g. BRK-B).")
        df = df.dropna(subset=["Value"])
        sm = PF.summarize(df)
        ignored = int((holdings["shares"].fillna(0) <= 0).sum())
        if ignored:
            st.caption(f"{ignored} row(s) with zero or negative shares are ignored (short positions aren't supported).")
        if sm.get("missing_cost"):
            st.caption(f"{sm['missing_cost']} holding(s) have no average cost, so they are left out of gain/loss "
                       "and yield-on-cost. Add the cost in Manage holdings.")

        with st.container(key="pf_values"):
            _summary_row(sm)
            updated_caption(fetched)
            _holdings_table(df)
            for note in PF.insights(df):
                st.info(note)
            t1, t2, t3 = st.tabs(["Allocation & performance", "Dividend income", "Upcoming events"])
            with t1:
                _charts(df, holdings)
            with t2:
                _income(df, sm)
            with t3:
                _events(df)
    return _panel


def render(on_cloud=False, has_password=False):
    live_panel(_make_panel(on_cloud, has_password))
