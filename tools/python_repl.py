import os
import pandas as pd
from langchain_experimental.tools import PythonREPLTool

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "archive")


def create_python_repl_tool() -> PythonREPLTool:
    """Create a PythonREPLTool with all CSV data pre-loaded as DataFrames."""
    repl_tool = PythonREPLTool()

    # Pre-load data into the REPL's global namespace
    setup_code = f"""
import pandas as pd
import os

DATA_DIR = {DATA_DIR!r}

# Load all CSVs into a single combined DataFrame
_frames = []
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith('.csv'):
        _frames.append(pd.read_csv(os.path.join(DATA_DIR, f)))
df = pd.concat(_frames, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'])

# Also create individual DataFrames per coin for convenience
coins = {{}}
for name in df['Name'].unique():
    var_name = name.replace(' ', '').replace('.', '')
    coins[name] = df[df['Name'] == name].sort_values('Date').reset_index(drop=True)

print(f"Loaded {{len(df)}} rows across {{df['Name'].nunique()}} coins.")
print(f"Coins: {{', '.join(sorted(df['Name'].unique()))}}")
print(f"Columns: {{list(df.columns)}}")
print(f"Date range: {{df['Date'].min().strftime('%Y-%m-%d')}} to {{df['Date'].max().strftime('%Y-%m-%d')}}")
"""
    # Execute setup code to populate the REPL environment
    repl_tool.run(setup_code)

    # Override the tool description to be more helpful for the agent
    repl_tool.description = (
        "Execute Python code in an interactive REPL environment. "
        "The following variables are pre-loaded:\n"
        "- `df`: A pandas DataFrame with ALL cryptocurrency data combined "
        "(columns: SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap)\n"
        "- `coins`: A dict mapping coin name -> individual DataFrame\n"
        "- `pd`: pandas is imported\n\n"
        "Use this for computations, aggregations, filtering, and analysis. "
        "Always use print() to output results."
    )

    return repl_tool
