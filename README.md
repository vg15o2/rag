# Crypto Data Analyst

A conversational CLI data analyst powered by GPT-4o-mini that answers questions about cryptocurrency price data using RAG (Retrieval-Augmented Generation) with ChromaDB and LangChain.

## Data

23 cryptocurrencies with daily OHLCV data (2013-2021), renamed to `a.csv`–`w.csv`.

Columns: `SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap`

## Setup

```bash
cd ~/RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

Build the vector index:

```bash
python generate_file_info.py
python build_index.py
```

## Usage

```bash
source venv/bin/activate
python app.py
```

## How It Works

1. **Semantic Search** — user query is embedded and matched against file descriptions in ChromaDB
2. **Data Retrieval** — matched CSV files are identified
3. **Python REPL** — pandas code runs on pre-loaded DataFrames for computation
4. **LLM Generation** — GPT-4o-mini produces the final answer

## Project Structure

```
├── app.py                 # CLI entry point (chat loop)
├── rag_agent.py           # RAG agent (ChromaDB + REPL)
├── agent.py               # Phase 1 agent (keyword search + REPL)
├── build_index.py         # Embeds file_info.txt into ChromaDB
├── generate_file_info.py  # Generates file descriptions (pandas + LLM)
├── file_info.txt          # Generated metadata for each CSV
├── tools/
│   ├── csv_search.py      # Keyword search across CSVs
│   └── python_repl.py     # Python REPL with pre-loaded DataFrames
├── archive/               # 23 cryptocurrency CSV files (a.csv–w.csv)
├── requirements.txt       # Dependencies
└── .env                   # OpenAI API key (not tracked)
```

## Tech Stack

- LangChain + LangGraph
- ChromaDB (vector store)
- OpenAI GPT-4o-mini + text-embedding-3-small
- pandas
