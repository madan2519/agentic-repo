# chain.py - Two agents with tools and a chain calling both

import os
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()


# =========================
# Tools
# =========================
@tool
def get_weather(city: str) -> str:
    """Return demo weather for a city."""
    data = {
        "chennai": "Chennai: 32°C, humid, chance of thunderstorms",
        "mumbai": "Mumbai: 29°C, light rain, breezy",
        "delhi": "Delhi: 26°C, clear skies",
    }
    return data.get((city or "").strip().lower(), f"{city}: 28°C, partly cloudy (demo)")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (digits/operators/parentheses)."""
    allowed = set("0123456789+-*/(). ")
    expr = str(expression or "")
    if not set(expr) <= allowed:
        return "Unsupported expression. Use digits and + - * / ( )."
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return f"{expr} = {result}"
    except Exception as e:
        return f"Error: {e}"


# =========================
# Model
# =========================
MODEL_ID = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL_ID, temperature=0)


# =========================
# Agents
# =========================
calculator_agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="You are a calculator assistant. Use the calculator tool to compute expressions.",
)

weather_agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a weather assistant. Use the get_weather tool to provide weather information.",
)


# =========================
# Chain that calls both agents using pipe
# =========================

# Since agents are runnables, we can chain them with |
# This creates a sequential chain: calculator_agent -> weather_agent
chained_agents = calculator_agent | weather_agent


# =========================
# Main method to test the chain
# =========================
def main():
    print("=== Testing the chained agents ===")
    result = chained_agents.invoke({
        "messages": [("user", "Calculate 12*5 and get weather for Chennai")]
    })
    print("All messages:")
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")
    print("\nFinal result (last message):")
    print(result["messages"][-1].content if result["messages"] else "No messages")


if __name__ == "__main__":
    main()
