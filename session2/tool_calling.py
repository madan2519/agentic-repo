# tool_using_agent.py
import os

# (Optional) load env vars from .env if present
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool


# -------------------------
# Tools (simple, deterministic stubs)
# -------------------------

@tool
def get_weather(city: str) -> str:
    """Get current demo weather for a given city.
    Use when the user asks about weather, forecast, or temperature for a location.
    """
    data = {
        "chennai": "Chennai: 32°C, humid, chance of thunderstorms",
        "mumbai": "Mumbai: 29°C, light rain, breezy",
        "delhi": "Delhi: 26°C, clear skies",
    }
    return data.get(city.strip().lower(), f"{city}: 28°C, partly cloudy (demo)")

@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression (digits, + - * / . ( )).
    Use this for math so users receive exact numeric results.
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Unsupported expression. Use digits and + - * / ( )."
    try:
        # SUPER-restricted eval: no builtins, no names
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


# -------------------------
# Agent setup (model + behavior)
# -------------------------

MODEL_ID = os.getenv("LLM_MODEL", "gpt-4o-mini")  # choose any available chat model
llm = ChatOpenAI(model=MODEL_ID, temperature=0)   # low temperature for consistent tool use

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Decide when to call tools. "
    "For weather questions, extract the city and call get_weather. "
    "For math, pass the exact arithmetic expression to calculator. "
    "If neither tool is needed, answer directly and concisely."
)

# create_agent builds a LangGraph-backed agent loop that can call tools automatically
agent = create_agent(
    model=llm,
    tools=[get_weather, calculator],
    system_prompt=SYSTEM_PROMPT,
)


def ask_agent(user_input: str) -> str:
    """Invoke the agent with chat-style messages and return the final text response."""
    state = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

    # Robustly extract the final assistant reply across versions:
    content = getattr(state, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(state, dict) and "messages" in state and state["messages"]:
        last = state["messages"][-1]
        return getattr(last, "content", str(last))

    return str(state)


if __name__ == "__main__":
    print("✅ Tool-Using Agent (LangChain) ready.")
    print("Model:", MODEL_ID)
    print("Type a question (or 'exit'):\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q or q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            ans = ask_agent(q)
            print(f"Assistant: {ans}\n")
        except Exception as e:
            print("Error:", e)
            print("Tip: Ensure OPENAI_API_KEY is set and the model name is valid.\n")