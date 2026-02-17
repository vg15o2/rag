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

## Overview

### One-Time Setup Scripts

#### `generate_file_info.py` → creates `file_info.txt`

```
archive/a.csv ──┐
archive/b.csv ──┤
...             ├──→ [pandas: extract_stats()] ──→ raw stats per file
archive/w.csv ──┘                                        │
                                                         ▼
                                              [OpenAI GPT-4o-mini]
                                              "Write a 2-3 sentence
                                               description of this data"
                                                         │
                                                         ▼
                                                  file_info.txt
                                              (23 paragraphs, one per CSV)
```

- Loops through all 23 CSVs in `archive/`
- `extract_stats()` uses **pandas** to pull hard facts: coin name, symbol, row count, date range, min/max price, avg volume, avg market cap
- Sends those stats to **GPT-4o-mini** via `generate_description()` which writes a natural 2-3 sentence description
- Writes all 23 descriptions to `file_info.txt`

#### `build_index.py` → creates `chroma_db/`

```
file_info.txt ──→ split into 23 entries
                        │
                        ▼
              [OpenAI text-embedding-3-small]
              converts each description into
              a 1536-dimension vector
                        │
                        ▼
                   chroma_db/
              (persistent vector database)
              stores: vector + text + metadata
              metadata = {"filename": "c.csv"}
```

- Reads `file_info.txt`, splits on blank lines into 23 entries
- Extracts the filename (e.g. `c.csv`) as metadata for each entry
- Uses OpenAI `text-embedding-3-small` to convert each description into a vector
- Stores everything in a persistent ChromaDB collection at `chroma_db/`

### Runtime: Startup Sequence

```
python app.py
    │
    ▼
app.py: load_dotenv()
    │  Reads .env → sets OPENAI_API_KEY in environment
    │
    ▼
app.py: from rag_agent import create_rag_agent
    │
    │  Triggers rag_agent.py to load:
    │  ├─ Imports tools/python_repl.py
    │  ├─ Opens chroma_db/ (loads vector index into memory)
    │  └─ Creates OpenAI embeddings client
    │
    ▼
app.py: agent, reset_repl = create_rag_agent()
    │
    │  Inside create_rag_agent():
    │  1. Creates ChatOpenAI(model="gpt-4o-mini")
    │  2. Calls _create_guarded_repl():
    │     ├─ create_python_repl_tool() → starts a live Python interpreter
    │     │   └─ Runs setup code: loads ALL 23 CSVs into `df` and `coins` dict
    │     └─ Wraps it with call counter (max 2 calls per question)
    │  3. Registers tools: [rag_search, python_repl]
    │  4. create_react_agent() → LangGraph ReAct agent
    │  5. Returns (agent, reset_function)
    │
    ▼
Chat loop ready. Waits for user input.
```

### Runtime: Per-Question Flow

```
User question
    │
    ▼
reset_repl() → counter = 0
    │
    ▼
agent.stream(question)
    │
    ┌──────────────────────────────────────┐
    │         LLM DECISION STEP            │
    │                                      │
    │  GPT-4o-mini receives:               │
    │  - System prompt (rules + tools)     │
    │  - User question                     │
    │  - Any previous tool results         │
    │                                      │
    │  Decides: call a tool? or answer?    │
    └───────────┬──────────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
  TOOL CALL          FINAL ANSWER
       │                 │
       ▼                 ▼
  rag_search        app.py prints
     OR             the response
  python_repl
       │
       │ tool output
       ▼
  Back to LLM DECISION STEP
```

### Example Traces

**"What was Bitcoin's highest price?"** (rag_search only)

```
Step 1: LLM → calls rag_search("Bitcoin's highest price")
Step 2: rag_search embeds query → ChromaDB returns c.csv description
        "[c.csv] Bitcoin (BTC)... price range $65.53 to $64,863.10"
Step 3: LLM sees price in description → answers directly
        → "Bitcoin's highest price was $64,863.10"
```

**"Compare Bitcoin and Litecoin closing prices"** (rag_search + REPL)

```
Step 1: LLM → calls rag_search("Compare Bitcoin and Litecoin")
Step 2: Returns descriptions for c.csv (BTC) and l.csv (LTC)
Step 3: LLM → needs computation → calls python_repl with pandas merge code
Step 4: REPL executes using pre-loaded `coins` dict → returns table
Step 5: LLM formats and presents the comparison
```

**"Top 3 coins by market cap"** (REPL only, skips rag_search)

```
Step 1: LLM sees "top 3" = cross-coin question → skips rag_search
Step 2: Calls python_repl: df.groupby('Name')['Marketcap'].mean().nlargest(3)
Step 3: REPL returns result → LLM presents answer
```

### Guard Rails

```
Per question:
    reset_repl() sets counter = 0

    REPL call 1: counter → 1  ✓  executes code
    REPL call 2: counter → 2  ✓  executes code
    REPL call 3: counter → 3  ✗  returns "STOP: give your answer NOW"

    recursion_limit = 25: hard cap on total LangGraph steps
```

### File Dependency Map

```
.env ──────────────────→ app.py (loads API key)
                            │
                            └──→ rag_agent.py
                                    │
                                    ├──→ chroma_db/           (reads vector index)
                                    │       ▲
                                    │       │ created by
                                    │    build_index.py ←── file_info.txt
                                    │                           ▲
                                    │                           │ created by
                                    │                     generate_file_info.py ←── archive/*.csv
                                    │
                                    └──→ tools/python_repl.py  ←── archive/*.csv
```
