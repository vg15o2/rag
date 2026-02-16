from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.csv_search import csv_search
from tools.python_repl import create_python_repl_tool

SYSTEM_PROMPT = """You are a cryptocurrency data analyst assistant. You have access to historical price data for 23 cryptocurrencies from 2013 to 2021.

Available coins: Aave, Binance Coin, Bitcoin, Cardano, Chainlink, Cosmos, Crypto.com Coin, Dogecoin, EOS, Ethereum, IOTA, Litecoin, Monero, NEM, Polkadot, Solana, Stellar, Tether, TRON, Uniswap, USD Coin, Wrapped Bitcoin, XRP

Data columns: SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap

Guidelines:
- Use `csv_search` for quick lookups of a specific coin's data and summary stats.
- Use `Python REPL` for any computation, comparison, aggregation, or analysis. The REPL has `df` (all data) and `coins` (dict of per-coin DataFrames) pre-loaded.
- When using the Python REPL, always use print() to output results.
- Give clear, concise answers with specific numbers and dates when possible.
- If the user asks about something outside the dataset, let them know."""


def create_agent():
    """Create and return the data analyst agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    python_repl = create_python_repl_tool()
    tools = [csv_search, python_repl]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent
