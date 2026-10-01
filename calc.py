"""Dividend and DRIP (dividend reinvestment) math. Pure functions: no Streamlit, no network."""
import numpy as np
import pandas as pd

FREQUENCIES = {"Monthly": 12, "Quarterly": 4, "Semi-annual": 2, "Annual": 1}


def freq_from_count(payments_per_year):
    """Map a count of payments in a trailing year to the nearest standard frequency label."""
    if payments_per_year <= 0:
        return "Quarterly"
    nearest = min(FREQUENCIES.values(), key=lambda f: abs(f - payments_per_year))
    return next(k for k, v in FREQUENCIES.items() if v == nearest)


def income_summary(shares, price, annual_dps):
    """Income and yield figures for a position. annual_dps = dividends per share per year."""
    annual = shares * annual_dps
    value = shares * price
    return {
        "value": value,
        "annual": annual,
        "monthly": annual / 12,
        "quarterly": annual / 4,
        "weekly": annual / 52,
        "daily": annual / 365,
        "yield_pct": annual / value * 100 if value else 0.0,
    }


def investment_for_income(target_monthly_income, yield_pct):
    """Cash needed to generate a monthly income at a given dividend yield."""
    if yield_pct <= 0:
        return float("nan")
    return target_monthly_income * 12 / (yield_pct / 100)


def simulate(initial_shares, price, annual_dps, freq_per_year, div_growth_pct, price_growth_pct, years,
             monthly_contribution=0.0, drip=True, tax_rate_pct=0.0, start_cost_basis=None):
    """Month-by-month projection. Returns a DataFrame with one row per year (row 0 = today).

    - Price compounds smoothly at price_growth_pct per year.
    - The dividend per share steps up by div_growth_pct at the start of each year.
    - Dividends are paid every 12/freq_per_year months, taxed at tax_rate_pct, then reinvested at that
      month's price if drip=True, otherwise kept as cash.
    - monthly_contribution buys more shares at the start of each month.
    """
    freq = max(int(freq_per_year), 1)
    step = 12 // freq
    g_price = price_growth_pct / 100.0
    g_div = div_growth_pct / 100.0
    tax = tax_rate_pct / 100.0

    shares = float(initial_shares)
    invested = float(start_cost_basis) if start_cost_basis is not None else float(initial_shares) * price
    cash = 0.0
    cum_div = 0.0
    year_div = 0.0

    rows = [dict(Year=0, Shares=shares, Price=price, Value=shares * price, Cash=0.0,
                 Total=shares * price, AnnualIncome=shares * annual_dps, Dividends=0.0,
                 CumDividends=0.0, Invested=invested)]
    for m in range(1, int(years) * 12 + 1):
        p_m = price * (1 + g_price) ** (m / 12.0)
        dps = annual_dps * (1 + g_div) ** ((m - 1) // 12)
        if monthly_contribution:
            shares += monthly_contribution / p_m
            invested += monthly_contribution
        if m % step == 0:
            net = shares * dps / freq * (1 - tax)
            cum_div += net
            year_div += net
            if drip:
                shares += net / p_m
            else:
                cash += net
        if m % 12 == 0:
            next_dps = annual_dps * (1 + g_div) ** (m // 12)
            rows.append(dict(Year=m // 12, Shares=shares, Price=p_m, Value=shares * p_m, Cash=cash,
                             Total=shares * p_m + cash, AnnualIncome=shares * next_dps,
                             Dividends=year_div, CumDividends=cum_div, Invested=invested))
            year_div = 0.0
    df = pd.DataFrame(rows)
    df["YieldOnCost"] = np.where(df["Invested"] > 0, df["AnnualIncome"] / df["Invested"] * 100, 0.0)
    return df


def compare_drip(**kwargs):
    """Run the same inputs with and without dividend reinvestment."""
    kwargs = dict(kwargs)
    kwargs.pop("drip", None)
    return simulate(drip=True, **kwargs), simulate(drip=False, **kwargs)


def cagr(first, last, years):
    if first is None or last is None or first <= 0 or last <= 0 or years <= 0:
        return float("nan")
    return ((last / first) ** (1 / years) - 1) * 100
