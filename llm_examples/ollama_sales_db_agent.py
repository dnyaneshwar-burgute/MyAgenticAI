
from agents import (
    Agent,
    Runner,
    function_tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
from dotenv import load_dotenv
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(override=True)

# Configure local Ollama
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Local Ollama ignores this value
)

local_model = OpenAIChatCompletionsModel(
    model="qwen3:8b",
    openai_client=ollama_client,
)

# Disable OpenAI tracing when using a local model
set_tracing_disabled(True)


@function_tool
def run_sql(query: str) -> str:
    """Execute a read-only SQL query on the sales database."""

    # Enforce read-only database access
    db_path = os.path.join(BASE_DIR, "sales.db")
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
    )

    try:
        # SQLite-level read-only query protection
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return str(rows)
    finally:
        conn.close()


agent = Agent(
    name="Sales Assistant",
    model=local_model,
    instructions="""
    You help users answer questions about the sales database.

    Use the run_sql tool whenever database information is needed.
    Generate only read-only SQL queries.
    Never modify or delete database records.
    Explain the results clearly.
    """,
    tools=[run_sql],
)

result = Runner.run_sync(
    agent,
    "Which product names generated more than ₹200,000 in revenue?",
)

print(result.final_output)
