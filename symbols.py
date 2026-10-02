"""Ticker symbols the way Yahoo Finance expects them, including Canadian listings.

Yahoo adds an exchange suffix to non-US stocks: RY.TO is Royal Bank on the Toronto Stock Exchange (TSX), ABC.V is a
TSX Venture stock, .CN is the Canadian Securities Exchange, .NE is the NEO exchange. US class shares use a dash
instead (BRK.B -> BRK-B). Pure functions, no network."""

CANADA_SUFFIXES = (".TO", ".V", ".CN", ".NE")
OTHER_SUFFIXES = (".L", ".PA", ".DE", ".AX", ".HK", ".SI", ".T", ".NS", ".BO", ".SW", ".MI", ".AS", ".MC", ".SA")
KNOWN_SUFFIXES = CANADA_SUFFIXES + OTHER_SUFFIXES

# other spellings people use for the same suffix
_ALT_SUFFIX = {".TSX": ".TO", ".TSE": ".TO", ".TSXV": ".V", ".CVE": ".V", ".CSE": ".CN", ".CNSX": ".CN", ".NEO": ".NE"}

# exchange names (as in broker exports or "TSX:RY") -> Yahoo suffix
EXCHANGE_SUFFIX = {
    "TSX": ".TO", "TSE": ".TO", "TORONTO": ".TO", "TORONTO STOCK EXCHANGE": ".TO",
    "TSXV": ".V", "TSX-V": ".V", "TSX VENTURE": ".V", "CVE": ".V", "VENTURE": ".V",
    "CSE": ".CN", "CNSX": ".CN", "CANADIAN SECURITIES EXCHANGE": ".CN",
    "NEO": ".NE", "NEOE": ".NE", "CBOE CANADA": ".NE",
}

_SUFFIX_CURRENCY = {".TO": "CAD", ".V": "CAD", ".CN": "CAD", ".NE": "CAD", ".L": "GBP", ".PA": "EUR", ".DE": "EUR",
                    ".AX": "AUD", ".HK": "HKD", ".SI": "SGD", ".T": "JPY", ".NS": "INR", ".BO": "INR",
                    ".SW": "CHF", ".MI": "EUR", ".AS": "EUR", ".MC": "EUR", ".SA": "BRL"}


def suffix_of(symbol):
    """The exchange suffix of a symbol (e.g. '.TO'), or '' for a plain US-style symbol."""
    s = str(symbol).upper()
    for suf in KNOWN_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return suf
    return ""


def normalize_symbol(raw):
    """Turn what a person typed into a Yahoo symbol.

    'ry.to' -> 'RY.TO'     'TSX:RY' -> 'RY.TO'     'brk.b' -> 'BRK-B'     'bbd.b.to' -> 'BBD-B.TO'     ' aapl ' -> 'AAPL'
    """
    s = str(raw).strip().upper().replace(" ", "")
    if not s:
        return s
    if ":" in s:
        prefix, _, rest = s.partition(":")
        s = rest + EXCHANGE_SUFFIX[prefix] if prefix in EXCHANGE_SUFFIX else rest   # NYSE:KO -> KO
    for alt, real in _ALT_SUFFIX.items():
        if s.endswith(alt) and len(s) > len(alt):
            s = s[:-len(alt)] + real
            break
    suf = suffix_of(s)
    if suf:
        return s[:-len(suf)].replace(".", "-").replace("/", "-") + suf
    return s.replace(".", "-").replace("/", "-")


def with_exchange(symbol, exchange):
    """Add the Canadian suffix implied by an exchange name from a broker export (TSX -> .TO) if the symbol has none."""
    sym = normalize_symbol(symbol)
    if suffix_of(sym):
        return sym
    return sym + EXCHANGE_SUFFIX.get(str(exchange).strip().upper(), "")


def listing_currency(symbol, yahoo_currency=None):
    """(currency code, price scale). Yahoo reports London prices in pence ('GBp'), so scale them to pounds."""
    cur = (yahoo_currency or "").strip()
    if cur == "GBp":
        return "GBP", 0.01
    if cur:
        return cur.upper(), 1.0
    return _SUFFIX_CURRENCY.get(suffix_of(symbol), "USD"), 1.0
