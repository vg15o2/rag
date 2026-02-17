"""RAG-based cryptocurrency data analyst agent."""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from tools.python_repl import create_python_repl_tool

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "archive")
COLLECTION_NAME = "file_info"

# Initialize vector store
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=CHROMA_DIR,
)


@tool
def rag_search(query: str) -> str:
    """Search for relevant cryptocurrency data files using semantic search.

    Returns file descriptions and filenames for matching coins.
    Use this to identify WHICH coin/file is relevant.
    For actual computation, use the Python REPL after this.

    Args:
        query: Natural language question about cryptocurrency data
    """
    results = _vectorstore.similarity_search(query, k=3)

    if not results:
        return "No relevant files found for your query."

    output_parts = []
    for doc in results:
        filename = doc.metadata["filename"]
        output_parts.append(f"[{filename}] {doc.page_content}")

    return "\n\n".join(output_parts)


# Track REPL calls to prevent infinite retry loops
_repl_call_count = 0


def _create_guarded_repl():
    """Wrap the Python REPL with a call counter to prevent retry loops."""
    inner_repl = create_python_repl_tool()

    @tool
    def python_repl(code: str) -> str:
        """Execute Python code in a REPL with pre-loaded crypto data.

        Pre-loaded variables:
        - `df`: pandas DataFrame with ALL crypto data (columns: SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap)
        - `coins`: dict mapping coin name -> individual DataFrame (e.g. coins['Bitcoin'], coins['Ethereum'])
        - `pd`: pandas

        Always use print() to show results.

        Args:
            code: Python code to execute
        """
        global _repl_call_count
        _repl_call_count += 1
        if _repl_call_count > 2:
            return "STOP: You have already called the REPL twice. Give your final answer NOW using the data you already have."
        result = inner_repl.invoke(code)
        if not result or not result.strip():
            return "Code ran but produced no output. Did you forget print()? Give your answer from available data."
        return result

    return python_repl


def _reset_repl_counter():
    global _repl_call_count
    _repl_call_count = 0


SYSTEM_PROMPT = """You are a cryptocurrency data analyst. You answer questions using a RAG pipeline.

TOOLS:
- `rag_search`: Finds which CSV files match a query. Returns descriptions for top 3 matches.
- `python_repl`: Run pandas code. Pre-loaded: `df` (all data), `coins` (dict per coin, e.g. coins['Bitcoin']), `pd`.

RULES:
1. Call rag_search AT MOST once. Call python_repl AT MOST once. NEVER retry a tool.
2. If rag_search already has the answer (e.g. row count, date range, price range), answer directly WITHOUT calling python_repl.
3. Only use python_repl when you need to COMPUTE something (averages, comparisons, filtering by date).
4. In python_repl: write ONE short snippet, always use print(). If it errors, answer from what you have.
5. After ANY tool output, give your final answer IMMEDIATELY."""


def create_rag_agent():
    """Create and return the RAG-based data analyst agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    python_repl = _create_guarded_repl()
    tools = [rag_search, python_repl]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    # Return agent and reset function
    return agent, _reset_repl_counter
