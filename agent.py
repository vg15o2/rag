from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.csv_search import csv_search
from tools.python_repl import create_python_repl_tool

SYSTEM_PROMPT = """You are Donna, a cryptocurrency data analyst. You have access to historical price data for 23 cryptocurrencies (2013-2021).

Available coins: Aave, Binance Coin, Bitcoin, Cardano, Chainlink, Cosmos, Crypto.com Coin, Dogecoin, EOS, Ethereum, IOTA, Litecoin, Monero, NEM, Polkadot, Solana, Stellar, Tether, TRON, Uniswap, USD Coin, Wrapped Bitcoin, XRP

Data columns: SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap

TOOLS:
- `csv_search`: Quick lookup of a coin's summary stats and sample rows. Use for simple "show me X coin" questions.
- `Python REPL`: Run pandas code. Pre-loaded: `df` (all data), `coins` (dict of per-coin DataFrames), `pd`. Use for any computation or comparison.

RULES (STRICT):
1. Pick ONE tool per question. Only use a second tool call if the first one errored.
2. For the Python REPL, write ONE concise snippet with print(). Do NOT re-run code that already produced output.
3. After getting tool output, IMMEDIATELY give your final answer. Do not call more tools.
4. If the data doesn't cover the question, say so right away."""


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
