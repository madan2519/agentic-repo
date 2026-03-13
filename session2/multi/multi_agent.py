# multi_agent_system.py
import os
from typing import Optional

# (Optional) Load local .env
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import (
    get_weather,  # Weather tool
    calculator,   # Calculator for math
    researcher_kb,  # Simple knowledge base for research
    unit_test,   # Unit testing tool for numeric results
)


# -------------------------
# Models (shared)
# -------------------------
MODEL_ID = os.getenv("LLM_MODEL", "gpt-4o-mini")

llm_connection = ChatOpenAI(model=MODEL_ID, temperature=0)  # shared connection for all agents
# Lower temperature for more deterministic orchestration
researcher_llm = llm_connection
coder_llm       = llm_connection
reviewer_llm    = llm_connection

# -------------------------
# Role Agents
# -------------------------

# 1) Researcher: clarifies requirements & drafts a spec. Has a small KB + weather tool (if needed).
RESEARCHER_SYSTEM = (
    "You are the Researcher. Your job:\n"
    "1) Clarify the user's task and constraints.\n"
    "2) Produce a concise SPEC with assumptions, steps, and data points needed.\n"
    "3) If the task involves domain info, call researcher_kb(query).\n"
    "4) If the task asks about weather, call get_weather(city) with a city.\n"
    "Output a short bullet list SPEC; avoid implementation details."
)
researcher_agent = create_agent(
    model=researcher_llm,
    tools=[researcher_kb, get_weather],
    system_prompt=RESEARCHER_SYSTEM,
)

# 2) Coder: produces a minimal implementation/answer. Has a calculator tool for numeric work.
CODER_SYSTEM = (
    "You are the Coder. Given the Researcher SPEC, produce the solution.\n"
    "- If a numeric result is needed, call calculator(expression) to compute.\n"
    "- Keep answers short and concrete. Include the final result explicitly if numeric."
)
coder_agent = create_agent(
    model=coder_llm,
    tools=[calculator],
    system_prompt=CODER_SYSTEM,
)

# 3) Reviewer: checks the solution vs. SPEC. Can run unit tests when numeric outputs are present.
REVIEWER_SYSTEM = (
    "You are the Reviewer. Compare the Coder's solution with the Researcher SPEC.\n"
    "- If the solution includes a numeric result, validate with unit_test(expected, expression) when obvious.\n"
    "- Respond with PASS or FAIL and 1–3 bullet feedback points. If FAIL, say what to fix."
)
reviewer_agent = create_agent(
    model=reviewer_llm,
    tools=[unit_test, calculator],
    system_prompt=REVIEWER_SYSTEM,
)



