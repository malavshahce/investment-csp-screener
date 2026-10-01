"""Small UI helpers shared by every tab."""
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

# Auto-refresh choices shown in the header (label -> seconds, None = off)
REFRESH_CHOICES = {"Auto-refresh off": None, "Every 30 sec": 30, "Every 1 min": 60,
                   "Every 5 min": 300, "Every 15 min": 900}


def md(text):
    """st.markdown that treats $ literally (no LaTeX)."""
    st.markdown(text.replace("$", "\\$"))


def money(x, decimals=0):
    """-$1,234 / $1,234 formatting (negative sign before the dollar sign)."""
    if x is None or x != x:
        return "—"
    s = f"{abs(x):,.{decimals}f}"
    return f"-${s}" if x < 0 else f"${s}"


def pct(x, decimals=2, signed=True):
    if x is None or x != x:
        return "—"
    return f"{x:+.{decimals}f}%" if signed else f"{x:.{decimals}f}%"


def big_number(x):
    """12,345,678,900 -> $12.35B"""
    if x is None or x != x:
        return "—"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"${x / div:,.2f}{unit}"
    return f"${x:,.0f}"


def updated_caption(fetched_at, extra="Yahoo data can be ~15 min delayed"):
    """Timestamp line for live panels. fetched_at is when the data was actually downloaded."""
    if fetched_at is None:
        return
    age = (datetime.now() - fetched_at).total_seconds()
    ago = f"{int(age)}s ago" if age < 90 else f"{int(age // 60)} min ago"
    st.caption(f"🕒 Updated {fetched_at:%b %d, %Y %H:%M:%S} ({ago}) · {extra}")


def rerun_fragment():
    """Re-run just the current fragment; fall back to a full rerun if we are not inside a fragment rerun."""
    try:
        st.rerun(scope="fragment")
    except st.errors.StreamlitAPIException:
        st.rerun()


def live_panel(render_fn):
    """Run render_fn as a fragment: it refreshes itself on the header's auto-refresh interval and
    re-runs alone (not the whole app) when you interact with widgets inside it."""
    interval = st.session_state.get("refresh_secs")
    st.fragment(run_every=interval)(render_fn)()


def _show(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)


def line_chart(df, height=320, zero=False, y_format=None):
    """Multi-series line chart whose y-axis fits the data (st.line_chart always includes 0, which flattens prices)."""
    d = df.copy()
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    temporal = pd.api.types.is_datetime64_any_dtype(d.index)
    long = d.rename_axis("x").reset_index().melt("x", var_name="Series", value_name="Value").dropna()
    y = alt.Y("Value:Q", scale=alt.Scale(zero=zero), title=None, axis=alt.Axis(format=y_format) if y_format else alt.Axis())
    chart = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("x:T" if temporal else "x:Q", title=None), y=y,
        color=alt.Color("Series:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("x", title=""), "Series", alt.Tooltip("Value:Q", format=",.2f")],
    ).properties(height=height)
    _show(chart)


def signed_bar(df, label, value, height=360, horizontal=True, fmt="+.2f"):
    """Bars sorted by value, green when positive and red when negative."""
    order = alt.EncodingSortField(field=value, order="descending")
    color = alt.condition(alt.datum[value] >= 0, alt.value("#1fb76a"), alt.value("#ef5350"))
    tip = [label, alt.Tooltip(f"{value}:Q", format=fmt)]
    if horizontal:
        chart = alt.Chart(df).mark_bar().encode(y=alt.Y(f"{label}:N", sort=order, title=None),
                                                x=alt.X(f"{value}:Q", title=None), color=color, tooltip=tip)
    else:
        chart = alt.Chart(df).mark_bar().encode(x=alt.X(f"{label}:N", sort=order, title=None),
                                                y=alt.Y(f"{value}:Q", title=None), color=color, tooltip=tip)
    _show(chart.properties(height=height))


def category_bar(df, label, value, order=None, height=240, fmt=",.2f"):
    """Bar chart that keeps categories in the given order (st.bar_chart sorts them alphabetically)."""
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{label}:N", sort=order if order else None, title=None),
        y=alt.Y(f"{value}:Q", title=None),
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=fmt)],
    ).properties(height=height)
    _show(chart)
