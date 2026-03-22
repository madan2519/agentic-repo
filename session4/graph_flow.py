# session4_graph_with_create_agent_conditional.py
from __future__ import annotations

import os
from typing import TypedDict, Annotated, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage
from langchain_core.tools import tool

from langchain.agents import create_agent  # inner agent runtime (no bind_tools)

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from dotenv import load_dotenv  # pip install python-dotenv
load_dotenv()


# =========================
# 1) Outer graph state
# =========================
class AgentState(TypedDict):
    # We keep a running conversation. Each node appends only its new messages.
    messages: Annotated[List[AnyMessage], add_messages]
    # name: str  # Just to show we can have other state variables if needed


# =========================
# 2) Demo tools (no network)
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


TOOLS = [get_weather, calculator]


# =========================
# 3) Inner agent (create_agent) — no bind_tools
# =========================
SYSTEM_PROMPT = (
    "You are a concise assistant. Use tools when needed. "
    "If math is requested, use the calculator tool. "
    "If weather is requested, extract the city and use get_weather. "
    "Answer clearly and helpfully."
)

MODEL_ID = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL_ID, temperature=0)

inner_agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


# =========================
# 4) Outer nodes
# =========================
async def agent_node(state: AgentState) -> AgentState:
    """
    Call the inner agent with the current messages.
    Append only the final AI message to the outer state.
    """
    print("state before agent_node:", state)
    result = await inner_agent.ainvoke({"messages": state["messages"]})

    final_ai = None
    if isinstance(result, dict) and "messages" in result and result["messages"]:
        last = result["messages"][-1]
        final_ai = last if isinstance(last, AIMessage) else last
    elif isinstance(result, AIMessage):
        final_ai = result

    final_result = {"messages": [final_ai]} if final_ai else {"messages": []}
    print("state after agent_node:", final_result)
    return final_result


async def concise_node(state: AgentState) -> AgentState:
    """
    Rewrite the last AI message to one sentence (post‑processing).
    Demonstrates a second node and conditional routing after 'agent'.
    """
    print("state before concise_node:", state)
    if not state["messages"]:
        return {"messages": []}

    last_ai = state["messages"][-1]
    if not isinstance(last_ai, AIMessage) or not isinstance(last_ai.content, str):
        return {"messages": []}

    rewrite_llm = ChatOpenAI(model=MODEL_ID, temperature=0)
    prompt_msgs = [
        ("system", "Rewrite the assistant's answer in ONE short sentence. Keep key numbers."),
        ("human", f"Answer:\n{last_ai.content}")
    ]
    rewritten = await rewrite_llm.ainvoke(prompt_msgs)
    # Append the new concise AI message (delta)
    concise_result = {"messages": [AIMessage(content=rewritten.content)]}
    print("state after concise_node:", concise_result)
    return concise_result


# =========================
# 5) Conditional routing
# =========================
def route_after_agent(state: AgentState) -> str:
    """
    Decide where to go after the 'agent' node:
      - If user asked for brief/tldr/concise OR the answer is long => 'concise'
      - Otherwise => 'end' (finish)
    """
    # Grab the last user message (for intent)
    user_text = ""
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human" and isinstance(m.content, str):
            user_text = m.content.lower()
            break

    # Grab the last AI message (for length)
    ai_text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and isinstance(m.content, str):
            ai_text = m.content
            break

    # Simple triggers for teaching purposes
    wants_brief = any(k in user_text for k in ["brief", "concise", "tldr", "one sentence"])
    long_answer = len(ai_text) > 250  # arbitrary threshold for demo

    if wants_brief or long_answer:
        return "concise"
    return "end"


# =========================
# 6) Build & compile the outer graph
# =========================
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("concise", concise_node)

    # START -> agent
    graph.add_edge(START, "agent")

    # Conditional: agent -> (concise | END)
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "concise": "concise",
            "end": END,
        },
    )

    # After concise rewrite, finish
    graph.add_edge("concise", END)

    # Compiling the graph optimizes it for execution and generates visualizations
    compiled_graph = graph.compile()
    # png_bytes = compiled_graph.get_graph().draw_mermaid_png()
    # with open("graph_flow_mermaid.png", "wb") as f:
    #     f.write(png_bytes)
    return compiled_graph
