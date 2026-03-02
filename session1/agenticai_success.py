from dotenv import load_dotenv
load_dotenv()

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Define the Tool (The "Actuator")
# This shows the class exactly how we give the AI a capability
@tool
def simple_calculator(expression: str) -> str:
    """Use this tool to calculate mathematical expressions. Input must be a valid math string."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {e}"

# 2. Initialize the Brain (The LLM)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Equip the brain with our tool
tools = [simple_calculator]

# 3. Create the modern LangGraph Agent (This completely replaces AgentExecutor!)
agent_executor = create_react_agent(llm, tools)

# 4. Run the Workflow
print("--- Agentic AI ---")
question = "What is 34534 multiplied by 2342, and then divided by 4?"

# We run the agent and stream the steps so the class can see the AI reasoning live
for step in agent_executor.stream({"messages": [("human", question)]}, stream_mode="values"):
    step["messages"][-1].pretty_print()