from dotenv import load_dotenv
import os

# Load environment variables from a .env file (if present)
load_dotenv()
# ---------------------------------------------------------------------------
# Ensure the key is set before proceeding
if "OPENAI_API_KEY" not in os.environ:
    print("Please set the OPENAI_API_KEY environment variable.")

# if "COHERE_API_KEY" not in os.environ:
#     print("Please set the COHERE_API_KEY environment variable.")

from langchain_openai import ChatOpenAI
# from langchain_cohere import ChatCohere

# Initialize the LLM (using a powerful model is recommended for ReAct)
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# llm = ChatCohere(model="command-a-03-2025", temperature=0)

# ---------------------------------------------------------------------------
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage
from tool import tools

from middle_ware import enhance_final_output  # Your custom middleware function


SYSTEM_CONTENT = """
You are a financial assistant.

You have access to tools that return structured JSON data.

When you call a tool:
- Carefully read the Observation.
- If the Observation contains JSON, parse it mentally.
- Extract the specific values needed to answer the question.
- NEVER respond with generic success messages.
- ALWAYS include actual values from the tool output in the Final Answer.
 
If the user asks for an exchange rate:
- Call latest_exchange_rates with the correct base currency.
- Extract the target currency rate from the returned JSON.
- Clearly state the numeric exchange rate in the Final Answer.

Rules:
- If both base and target currencies are mentioned, never ask clarifying questions.
- Always call latest_exchange_rates with base and symbols.
- Assume real-time spot rates.
- Do not answer without using the tool.


Question: {input}
Thought: {agent_scratchpad}

"""

# Flip this to False to disable all middleware (A/B demo in class)
# WITH_MIDDLEWARE = True
# WITH_MIDDLEWARE = False

# middleware = [enhance_final_output] if WITH_MIDDLEWARE else []

# agent = create_agent(
#     llm,
#     tools=tools,                    # your latest_exchange_rates tool list
#     system_prompt=SYSTEM_CONTENT,   # your existing system rules
#     middleware=middleware,          # <— the new bit
# )

agent = create_agent(
    llm, tools=tools, system_prompt=SYSTEM_CONTENT)