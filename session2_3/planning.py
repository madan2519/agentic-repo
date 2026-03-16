# planning_agent.py
import json
import os
from typing import Dict, List, Any

# (Optional) load .env if present
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

from langsmith import trace

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import tool


# -------------------------
# Tools (simple demo stubs)
# -------------------------

@tool
def get_weather(city: str) -> str:
    """Return demo weather for the given city."""
    city_lc = (city or "").strip().lower()
    demo = {
        "chennai": "Chennai: 32°C, humid, chance of thunderstorms",
        "mumbai": "Mumbai: 29°C, light rain, breezy",
        "delhi": "Delhi: 26°C, clear skies",
    }
    return demo.get(city_lc, f"{city}: 28°C, partly cloudy (demo)")

@tool
def calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression like '12 * (5 + 2)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Unsupported expression. Use digits and + - * / ( )."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"



# Map action name -> callable (what the executor can run)
ACTION_REGISTRY = {
    "get_weather": lambda args: get_weather.invoke(args) if isinstance(args, dict) else get_weather.invoke({"city": str(args)}),
    "calculator": lambda args: calculator.invoke(args),
}


# -------------------------
# Model configuration
# -------------------------

MODEL_ID = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Lower temperature for more deterministic plans
llm = ChatOpenAI(model=MODEL_ID, temperature=0)


# -------------------------
# 1) PLANNER: produce a JSON plan
# -------------------------

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a planning module. "
            "Given a user task, break it into 2–5 minimal steps using ONLY these actions:\n"
            " - get_weather(city: str)\n"
            " - calculator(expression: str)\n"
            "Return STRICT JSON with this schema:\n"
            "{{\n"
            '  "steps": [\n'
            '    {{"id": 1, "action": "get_weather" | "calculator" | "respond", "args": {{ ... }}}},\n'
            '    {{"id": 2, "action": "...", "args": {{ ... }}}}\n'
            "  ]\n"
            "}}\n"
            "Do not include explanations or extra keys—JSON only."
        ),
        ("human", "Task: {task}")
    ]
)
# LCEL
planner_chain = planner_prompt | llm | JsonOutputParser()  # strict JSON parsing



# -------------------------
# 2) EXECUTOR: run the plan step-by-step
# -------------------------

def execute_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Execute each step from the plan and capture step-wise results.
    Returns a list of logs: [{id, action, args, result}, ...]
    """
    logs: List[Dict[str, Any]] = []

    steps = plan.get("steps") or []
    if not isinstance(steps, list) or not steps:
        # Fallback: if plan is empty, create a trivial respond step
        steps = [{"id": 1, "action": "respond", "args": {"text": "No actionable steps. Provide a concise answer."}}]

    for step in steps:
        sid = step.get("id")
        action = step.get("action")
        args = step.get("args", {})

        if action in ACTION_REGISTRY:
            try:
                result = ACTION_REGISTRY[action](args)
            except Exception as e:
                result = f"Execution error: {e}"
        elif action == "respond":
            # This action is a direct text response (no tool call)
            result = args.get("text", "")
        else:
            result = f"Unknown action '{action}'. Skipped."

        logs.append(
            {
                "id": sid,
                "action": action,
                "args": args,
                "result": result,
            }
        )
    return logs


# -------------------------
# 3) SUMMARIZER: synthesize final answer from logs
# -------------------------

summarizer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant. Using the user task and the execution logs, "
            "write a concise final answer. If the logs contain direct answers, integrate them."
        ),
        ("human", "User Task:\n{task}\n\nExecution Logs (JSON):\n{logs_json}\n")
    ]
)
summarizer_chain = summarizer_prompt | llm | StrOutputParser()


def plan_execute_answer(user_task: str) -> str:
    # 1) Plan
    with trace("planning"):
        try:
            plan = planner_chain.invoke({"task": user_task})
            # print("Generated Plan:", json.dumps(plan, ensure_ascii=False, indent=2))  # Debug: see the raw plan
            # planner_chain already returns a Python dict via JsonOutputParser
            if isinstance(plan, str):
                plan = json.loads(plan)
        except Exception as e:
            # If planning fails, create a trivial respond plan
            plan = {"steps": [{"id": 1, "action": "respond", "args": {"text": f"Planning failed: {e}"}}]}

    # 2) Execute
    with trace("execution"):
        logs = execute_plan(plan)

    # 3) Summarize final answer
    with trace("summarization"):
        logs_json = json.dumps(logs, ensure_ascii=False, indent=2)
        final_answer = summarizer_chain.invoke({"task": user_task, "logs_json": logs_json})
    return final_answer


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    print("✅ Planning Agent (LangChain) ready.")
    print("Model:", MODEL_ID)
    print("Type your task (or 'exit'):\n")
    print("What's the weather in Chennai and calculate 12 + 5?\n")

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
            ans = plan_execute_answer(q)
            print(f"Assistant: {ans}\n")
        except Exception as e:
            print("Error:", e)
            print("Tip: Check OPENAI_API_KEY and your model name.\n")
