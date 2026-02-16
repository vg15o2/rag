import os
import pandas as pd
from langchain.tools import tool

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "archive")

# Pre-load all CSVs into a single DataFrame on import
_frames = []
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith(".csv"):
        _frames.append(pd.read_csv(os.path.join(DATA_DIR, f)))
ALL_DATA = pd.concat(_frames, ignore_index=True)
ALL_DATA["Date"] = pd.to_datetime(ALL_DATA["Date"])

# Build a lookup for quick name/symbol matching
COIN_NAMES = ALL_DATA["Name"].unique().tolist()
COIN_SYMBOLS = ALL_DATA["Symbol"].unique().tolist()


def _match_coin(query: str) -> str | None:
    """Find a coin name from a query string (case-insensitive)."""
    q = query.lower()
    for name in COIN_NAMES:
        if name.lower() in q:
            return name
    for sym in COIN_SYMBOLS:
        if sym.lower() in q:
            # Return the full name for the matched symbol
            return ALL_DATA[ALL_DATA["Symbol"] == sym]["Name"].iloc[0]
    return None


@tool
def csv_search(query: str) -> str:
    """Search cryptocurrency CSV data by coin name or symbol.

    Use this tool to quickly look up data for a specific cryptocurrency.
    You can search by coin name (e.g. 'Bitcoin') or symbol (e.g. 'BTC').
    Returns the first and last few rows plus summary statistics.

    Args:
        query: A coin name or symbol to search for (e.g. 'Bitcoin', 'ETH', 'Dogecoin')
    """
    coin = _match_coin(query)
    if coin is None:
        return (
            f"No coin matched for '{query}'. "
            f"Available coins: {', '.join(COIN_NAMES)}"
        )

    df = ALL_DATA[ALL_DATA["Name"] == coin].sort_values("Date")
    symbol = df["Symbol"].iloc[0]
    total_rows = len(df)
    date_min = df["Date"].min().strftime("%Y-%m-%d")
    date_max = df["Date"].max().strftime("%Y-%m-%d")

    # Show summary
    lines = [
        f"=== {coin} ({symbol}) ===",
        f"Rows: {total_rows} | Date range: {date_min} to {date_max}",
        "",
        "Summary statistics:",
        df[["High", "Low", "Open", "Close", "Volume", "Marketcap"]]
        .describe()
        .to_string(),
        "",
        f"First 3 rows:",
        df.head(3).to_string(index=False),
        "",
        f"Last 3 rows:",
        df.tail(3).to_string(index=False),
    ]
    return "\n".join(lines)
