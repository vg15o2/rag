"""Generate file_info.txt using hybrid approach: pandas stats + LLM descriptions."""

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "archive")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "file_info.txt")

client = OpenAI()


def extract_stats(filepath: str) -> dict:
    """Extract hard facts from a CSV using pandas."""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    return {
        "name": df["Name"].iloc[0],
        "symbol": df["Symbol"].iloc[0],
        "rows": len(df),
        "date_min": df["Date"].min().strftime("%Y-%m-%d"),
        "date_max": df["Date"].max().strftime("%Y-%m-%d"),
        "price_min": f"${df['Low'].min():,.2f}",
        "price_max": f"${df['High'].max():,.2f}",
        "avg_volume": f"${df['Volume'].mean():,.0f}",
        "avg_marketcap": f"${df['Marketcap'].mean():,.0f}",
    }


def generate_description(filename: str, stats: dict) -> str:
    """Use GPT-4o-mini to write a natural description from the stats."""
    stats_text = (
        f"File: {filename}\n"
        f"Cryptocurrency: {stats['name']} ({stats['symbol']})\n"
        f"Rows: {stats['rows']}\n"
        f"Date range: {stats['date_min']} to {stats['date_max']}\n"
        f"Price range (Low to High): {stats['price_min']} to {stats['price_max']}\n"
        f"Average daily volume: {stats['avg_volume']}\n"
        f"Average market cap: {stats['avg_marketcap']}\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write concise dataset descriptions for a file catalog. "
                    "Write exactly 2-3 sentences describing what this CSV file contains. "
                    "Include the cryptocurrency name, symbol, date range, and key stats. "
                    "Start with the filename. Do not add any extra commentary."
                ),
            },
            {"role": "user", "content": stats_text},
        ],
    )
    return response.choices[0].message.content.strip()


def main():
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    entries = []

    for filename in csv_files:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Processing {filename}...", end=" ")

        stats = extract_stats(filepath)
        description = generate_description(filename, stats)
        entries.append(description)

        print("done")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n\n".join(entries))

    print(f"\nGenerated {OUTPUT_FILE} with {len(entries)} entries.")


if __name__ == "__main__":
    main()
