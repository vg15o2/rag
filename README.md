# Crypto Data Analyst

A conversational CLI data analyst powered by GPT-4o-mini that answers questions about cryptocurrency price data using LangChain and pandas.

## Data

23 cryptocurrencies with daily OHLCV data (2013-2021):

Aave, Binance Coin, Bitcoin, Cardano, Chainlink, Cosmos, Crypto.com Coin, Dogecoin, EOS, Ethereum, IOTA, Litecoin, Monero, NEM, Polkadot, Solana, Stellar, Tether, TRON, Uniswap, USD Coin, Wrapped Bitcoin, XRP

Columns: `SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap`

## Setup

```bash
cd ~/RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
source venv/bin/activate
python app.py
```

```
=== Crypto Data Analyst  ===
Ask me anything about cryptocurrency price data (2013-2021).

You: What was Bitcoin's highest closing price?
You: Compare ETH and BTC market cap in 2020
You: Which coin had the best return in 2021?
You: quit
```

## How It Works

The LLM agent has two tools:

- **csv_search** - Quick keyword lookup by coin name/symbol. Returns summary stats and sample rows.
- **Python REPL** - Executes pandas code on the fly. All data is pre-loaded as `df` (combined DataFrame) and `coins` (dict of per-coin DataFrames).

The system prompt enforces strict rules to prevent looping: pick one tool per question, answer immediately after getting output.

## Project Structure

```
├── app.py                 # CLI entry point (chat loop)
├── agent.py               # LangChain agent (GPT-4o-mini + tools)
├── tools/
│   ├── csv_search.py      # Keyword search across CSVs
│   └── python_repl.py     # Python REPL with pre-loaded DataFrames
├── archive/               # 23 cryptocurrency CSV files
├── requirements.txt       # Dependencies
└── .env                   # OpenAI API key (not tracked)
```

## Tech Stack

- LangChain + LangGraph
- OpenAI GPT-4o-mini
- pandas
