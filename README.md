# Investment toolkit (Streamlit)

Live-data stock tools built on Yahoo Finance (`yfinance`). Run locally with `streamlit run app.py`.

| Tab | What it does |
|---|---|
| **Market** | Indices, VIX, yields, gold/oil/bitcoin, sector performance, top movers |
| **Lookup** | Search any ticker/company: quote, chart, stats, dividends, news, compare, put ideas |
| **Watchlist** | Followed stocks with live prices and on-screen price alerts |
| **Portfolio** | Private holdings: value, gain/loss, allocation, dividend income, DRIP projection |
| **Income** | Dividend calculator and DRIP (dividend reinvestment) calculator |
| **Put Scanner** | Cash-secured put screener with a capital-aware trade plan |
| **Journal** | Log trades, live P&L, suggested actions, win-rate calibration |
| **Guide** | How to use everything and a glossary |

## Data and privacy
- Prices come from Yahoo (about 15 minutes delayed). Every live panel shows when its data was downloaded.
  Use **Refresh now** or **Auto-refresh** in the header.
- Your holdings, watchlist and trade journal are saved in `portfolio.csv`, `watchlist.csv` and
  `trade_journal.csv` next to the app. They are git-ignored and never uploaded.
- Set `app_password` in Streamlit secrets (see `.streamlit/secrets.toml.example`) to require a login.
  Do this before hosting the app online.

## Files
`app.py` (layout, scanner, journal, guide) · `engine.py` (scanner and option math) · `markets.py` (data layer) ·
`calc.py` (dividend/DRIP math) · `portfolio.py` (holdings + valuation) · `journal.py` · `common.py` ·
`view_*.py` (one per tab).
