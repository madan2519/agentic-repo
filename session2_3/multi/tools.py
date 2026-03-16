from langchain_core.tools import tool

# -------------------------
# Demo tools (purely local, no network)
# -------------------------

@tool
def researcher_kb(query: str) -> str:
    """Search a tiny in-memory knowledge base. Return a concise bullet list."""
    q = (query or "").lower()
    if "sorting" in q:
        return (
            "- For small lists or nearly-sorted data: Insertion sort.\n"
            "- For general purpose: Timsort (Python), Merge sort (stable), Quicksort (fast avg-case).\n"
            "- Consider constraints: input size, stability, memory."
        )
    if "weather" in q:
        return (
            "- Weather varies by city; ensure you know the location.\n"
            "- Always surface units (°C/°F) and time window.\n"
            "- Demo note: use get_weather(city) tool for specifics."
        )
    return (
        "- Clarify requirements and constraints first.\n"
        "- Prefer small, testable steps and verifiable outputs.\n"
        "- Summarize assumptions explicitly."
    )

@tool
def get_weather(city: str) -> str:
    """Demo weather for a city (offline stub)."""
    data = {
        "chennai": "Chennai: 32°C, humid, chance of thunderstorms",
        "mumbai": "Mumbai: 29°C, light rain, breezy",
        "delhi": "Delhi: 26°C, clear skies",
    }
    return data.get((city or "").strip().lower(), f"{city}: 28°C, partly cloudy (demo)")

@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Unsupported expression. Use digits and + - * / ( )."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def unit_test(expected: str, expression: str) -> str:
    """Evaluate an expression and compare against an expected value string."""
    calc = calculator.run(expression)  # use the calculator tool we already defined
    if "Error" in calc or "Unsupported" in calc:
        return f"TEST: ERROR running expression -> {calc}"
    # calc looks like "12*(5+2) = 84"
    try:
        actual = calc.split("=")[-1].strip()
        status = "PASS" if str(expected).strip() == actual else "FAIL"
        return f"TEST: {status} | expected={expected} actual={actual}"
    except Exception as e:
        return f"TEST: ERROR parsing calculator output: {e}"

