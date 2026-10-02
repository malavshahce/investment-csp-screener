"""Income tab: dividend calculator and DRIP (dividend reinvestment) calculator."""
import pandas as pd
import streamlit as st

import calc
import markets as MK
from common import category_bar, kpi, line_chart, md, money, updated_caption
from view_lookup import _preload_income

_DEFAULTS = {"inc_symbol": "", "inc_price": 100.0, "inc_dps": 3.0, "inc_freq": "Quarterly",
             "inc_div_growth": 5.0, "inc_price_growth": 6.0}
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _load_bar():
    c = st.columns([2, 1, 3])
    sym = c[0].text_input("Load a real stock (optional)", key="inc_symbol_in", placeholder="KO, SCHD, O, JEPI, AAPL…")
    if c[1].button("Load data", key="inc_load", width="stretch") and sym.strip():
        with st.spinner("Loading price, dividends and growth…"):
            ok = _preload_income(sym.strip().upper())
        if not ok:
            st.warning(f"Could not load {sym.upper()}.")
    ts = st.session_state.get("inc_loaded_at")
    if st.session_state.get("inc_symbol"):
        c[2].caption(f"Using **{st.session_state['inc_symbol']}**: price, annual dividend, payout frequency and "
                     f"growth rates were filled in from Yahoo"
                     f"{f' at {ts:%H:%M:%S}' if ts else ''}. Everything below is editable.")
    else:
        c[2].caption("Or just type your own numbers below.")


def _shared_inputs():
    c = st.columns(4)
    price = c[0].number_input("Share price ($)", min_value=0.01, step=1.0, key="inc_price", format="%.2f")
    dps = c[1].number_input("Dividend per share per year ($)", min_value=0.0, step=0.25, key="inc_dps", format="%.4f")
    freq = c[2].selectbox("Paid", list(calc.FREQUENCIES), key="inc_freq")
    c[3].metric("Dividend yield", f"{dps / price * 100:.2f}%")
    return price, dps, calc.FREQUENCIES[freq]


def _dividend_calculator(price, dps, freq):
    mode = st.radio("What do you want to know?", ["How much income will my shares pay?",
                                                  "How much do I need to invest for a target income?"],
                    horizontal=True, key="inc_mode")

    if mode.startswith("How much income"):
        c = st.columns(3)
        know = c[0].radio("I know my…", ["Number of shares", "Dollar amount"], horizontal=True, key="inc_know")
        if know == "Number of shares":
            shares = c[1].number_input("Shares", min_value=0.0, value=100.0, step=10.0, key="inc_shares")
        else:
            amount = c[1].number_input("Amount to invest ($)", min_value=0.0, value=10000.0, step=1000.0, key="inc_amount")
            shares = amount / price
            c[1].caption(f"≈ {shares:,.2f} shares")
        tax = c[2].number_input("Tax on dividends (%)", min_value=0.0, max_value=60.0, value=0.0, step=1.0, key="inc_tax")

        s = calc.income_summary(shares, price, dps)
        k = 1 - tax / 100
        st.markdown(f"**Position value {money(s['value'])}**" + (f" · income shown after {tax:g}% tax" if tax else ""))
        m = st.columns(6)
        for col, label, key in zip(m, ["Per year", "Per month", "Per quarter", "Per week", "Per day", "Yield"],
                                   ["annual", "monthly", "quarterly", "weekly", "daily", "yield_pct"]):
            col.metric(label, f"{s[key]:.2f}%" if key == "yield_pct" else money(s[key] * k, 2))

        st.markdown("**When you would get paid**")
        schedule = _pay_schedule(shares, dps, freq, k)
        category_bar(schedule.rename_axis('Month').reset_index(), 'Month', 'Income', order=_MONTHS, height=200, fmt='$,.2f')

        years = st.slider("Project dividend income over (years)", 1, 40, 15, key="inc_years_div")
        g = st.session_state["inc_div_growth"]
        proj = calc.simulate(shares, price, dps, freq, g, 0.0, years, drip=False, tax_rate_pct=tax)
        st.caption(f"If the company keeps raising its dividend {g:g}% a year (change the growth rate in the DRIP tab or "
                   f"edit it there), your yearly income grows like this, without reinvesting.")
        out = pd.DataFrame({"Year": proj["Year"], "Annual income": proj["AnnualIncome"] * k,
                            "Total dividends so far": proj["CumDividends"]}).set_index("Year")
        line_chart(out, height=260, zero=True)
    else:
        c = st.columns(2)
        target = c[0].number_input("Target income per month ($)", min_value=0.0, value=1000.0, step=100.0, key="inc_target")
        y = c[1].number_input("Dividend yield to assume (%)", min_value=0.1, value=round(dps / price * 100, 2) or 4.0,
                              step=0.25, key="inc_target_yield")
        need = calc.investment_for_income(target, y)
        shares = need / price
        m = st.columns(3)
        m[0].metric("You need to invest", money(need))
        m[1].metric("Which is about", f"{shares:,.1f} shares")
        m[2].metric("Yearly income", money(target * 12))
        st.markdown(f"**What ${target:,.0f}/month takes at different yields**".replace("$", "\\$"))
        tbl = pd.DataFrame({"Yield %": [2, 3, 4, 5, 6, 7, 8]})
        tbl["Investment needed"] = [calc.investment_for_income(target, y_) for y_ in tbl["Yield %"]]
        st.dataframe(tbl, hide_index=True, width="stretch",
                     column_config={"Yield %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Investment needed": st.column_config.NumberColumn(format="$%.0f")})
        st.caption("Very high yields can signal a risky or unsustainable dividend. Check the payout ratio and "
                   "dividend history in the Lookup tab.")


def _pay_schedule(shares, dps, freq, keep):
    """Income by month: actual pay months from history when the loaded stock matches, else a typical pattern."""
    sym = st.session_state.get("inc_symbol")
    months = []
    if sym:
        div, _ = MK.get_dividend_history(sym)
        recent = div[div["Date"] > pd.Timestamp.now() - pd.Timedelta(days=366)] if not div.empty else div
        months = sorted(set(recent["Date"].dt.month)) if not recent.empty else []
    if not months or len(months) != freq:
        step = 12 // freq
        months = list(range(step, 13, step)) if freq < 12 else list(range(1, 13))
    per = shares * dps / freq * keep
    return pd.DataFrame({"Income": [per if (i + 1) in months else 0.0 for i in range(12)]}, index=_MONTHS)


def _drip_calculator(price, dps, freq):
    c = st.columns(4)
    know = c[0].radio("Starting with…", ["Dollar amount", "Number of shares"], horizontal=True, key="drip_know")
    if know == "Dollar amount":
        amount = c[1].number_input("Starting amount ($)", min_value=0.0, value=10000.0, step=1000.0, key="drip_amount")
        shares0 = amount / price
        c[1].caption(f"≈ {shares0:,.2f} shares")
    else:
        shares0 = c[1].number_input("Starting shares", min_value=0.0, value=100.0, step=10.0, key="drip_shares")
        amount = shares0 * price
    contrib = c[2].number_input("Add every month ($)", min_value=0.0, value=0.0, step=100.0, key="drip_contrib")
    years = c[3].slider("Years", 1, 40, 20, key="drip_years")
    c = st.columns(4)
    dg = c[0].number_input("Dividend growth (%/yr)", min_value=-10.0, max_value=30.0, step=0.5, key="inc_div_growth")
    pg = c[1].number_input("Share price growth (%/yr)", min_value=-10.0, max_value=30.0, step=0.5, key="inc_price_growth")
    tax = c[2].number_input("Tax on dividends (%)", min_value=0.0, max_value=60.0, value=0.0, step=1.0, key="drip_tax")
    target = c[3].number_input("Target income per month ($)", min_value=0.0, value=1000.0, step=100.0, key="drip_target")

    if shares0 <= 0 and contrib <= 0:
        st.info("Enter a starting amount or a monthly contribution.")
        return
    kw = dict(initial_shares=shares0, price=price, annual_dps=dps, freq_per_year=freq, div_growth_pct=dg,
              price_growth_pct=pg, years=years, monthly_contribution=contrib, tax_rate_pct=tax)
    drip, plain = calc.compare_drip(**kw)
    d, p = drip.iloc[-1], plain.iloc[-1]

    st.markdown("#### Results")
    m = st.columns(4)
    m[0].metric("Final value WITH reinvesting", money(d["Total"]), f"{money(d['Total'] - p['Total'])} vs not reinvesting")
    kpi(m[1], "Final value WITHOUT reinvesting", money(p["Total"]), f"incl. {money(p['Cash'])} cash dividends")
    kpi(m[2], "Yearly dividend income (DRIP)", money(d["AnnualIncome"]), f"vs {money(p['AnnualIncome'])} without")
    kpi(m[3], "Shares owned (DRIP)", f"{d['Shares']:,.1f}", f"from {shares0:,.1f}")
    m = st.columns(4)
    m[0].metric("Total you put in", money(d["Invested"]))
    m[1].metric("Dividends received (after tax)", money(d["CumDividends"]))
    kpi(m[2], "Yield on your cost", f"{d['YieldOnCost']:.1f}%", "yearly income ÷ money put in")
    hit = drip[drip["AnnualIncome"] / 12 >= target]
    kpi(m[3], f"Reach {money(target)}/month", f"Year {int(hit['Year'].iloc[0])}" if target and not hit.empty else "Not within range",
        "dividends alone, with DRIP")

    chart = pd.DataFrame({"Reinvest dividends (DRIP)": drip["Total"], "Take dividends as cash": plain["Total"],
                          "Money you put in": drip["Invested"]}, index=drip["Year"])
    st.markdown("**Portfolio value over time**")
    line_chart(chart, height=300, zero=True)
    inc = pd.DataFrame({"With DRIP": drip["AnnualIncome"], "Without DRIP": plain["AnnualIncome"]}, index=drip["Year"])
    a, b = st.columns(2)
    with a:
        st.markdown("**Yearly dividend income**")
        line_chart(inc, height=240, zero=True)
    with b:
        st.markdown("**Shares owned (DRIP)**")
        line_chart(pd.DataFrame({"Shares": drip["Shares"]}, index=drip["Year"]), height=240, zero=True)

    st.markdown("**What if the share price grows faster or slower?**")
    rows = []
    for g in (-2, 0, 3, 6, 9, 12):
        r, _ = calc.compare_drip(**{**kw, "price_growth_pct": g})
        rows.append({"Price growth %/yr": g, "Final value (DRIP)": r["Total"].iloc[-1],
                     "Yearly income (DRIP)": r["AnnualIncome"].iloc[-1]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Price growth %/yr": st.column_config.NumberColumn(format="%+.0f%%"),
        "Final value (DRIP)": st.column_config.NumberColumn(format="$%.0f"),
        "Yearly income (DRIP)": st.column_config.NumberColumn(format="$%.0f")})

    with st.expander("Year-by-year table"):
        show = drip.copy()
        show["No-DRIP total"] = plain["Total"]
        st.dataframe(show[["Year", "Shares", "Price", "Value", "AnnualIncome", "Dividends", "CumDividends",
                           "Invested", "YieldOnCost", "No-DRIP total"]], hide_index=True, width="stretch",
                     column_config={
                         "Shares": st.column_config.NumberColumn(format="%.2f"),
                         "Price": st.column_config.NumberColumn(format="$%.2f"),
                         "Value": st.column_config.NumberColumn("Value (DRIP)", format="$%.0f"),
                         "AnnualIncome": st.column_config.NumberColumn("Yearly income", format="$%.0f"),
                         "Dividends": st.column_config.NumberColumn("Paid this year", format="$%.0f"),
                         "CumDividends": st.column_config.NumberColumn("Paid so far", format="$%.0f"),
                         "Invested": st.column_config.NumberColumn(format="$%.0f"),
                         "YieldOnCost": st.column_config.NumberColumn("Yield on cost", format="%.1f%%"),
                         "No-DRIP total": st.column_config.NumberColumn(format="$%.0f")})
    st.caption("An illustration, not a forecast. It assumes smooth growth in price and dividends. Real stocks move "
               "up and down, and dividends can be cut. Dividends are assumed to be reinvested at the share price on "
               "the payment month with no fees; tax is applied before reinvesting.")


def _panel():
    st.subheader("Dividend & DRIP calculators")
    for k, v in _DEFAULTS.items():
        st.session_state.setdefault(k, v)
    _load_bar()
    price, dps, freq = _shared_inputs()
    t1, t2 = st.tabs(["Dividend calculator", "DRIP calculator"])
    with t1:
        _dividend_calculator(price, dps, freq)
    with t2:
        _drip_calculator(price, dps, freq)


def render():
    st.fragment(_panel)()
