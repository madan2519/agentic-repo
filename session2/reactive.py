# reactive_agent.py
import os
from typing import Optional

# (Optional) Load .env for local dev
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
    print("✅ Loaded .env file.")
except Exception:
    pass

from langchain_openai import ChatOpenAI                       # OpenAI provider pkg
from langchain_core.prompts import ChatPromptTemplate         # Prompt composer
from langchain_core.output_parsers import StrOutputParser     # Get plain string

# --- Model config ---
# Pick any Chat Completions model you have access to (e.g., "gpt-4o-mini").
# The ChatOpenAI integration uses the OpenAI Chat Completions API under the hood.
MODEL_ID = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL_ID, temperature=0.2)

# --- Prompt: single-turn guidance, no memory/tools ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise, helpful assistant. "
            "Answer the user's request directly. If uncertain, say so briefly."
        ),
        ("human", "{question}"),
    ]
)

# LCEL pipeline: prompt -> model -> parse to str
chain = prompt | llm | StrOutputParser()


def reactive_response(question: str, *, system_hint: Optional[str] = None) -> str:
    """
    One-shot response. If system_hint is provided, it overrides the default system message.
    """
    if system_hint:
        custom_prompt = ChatPromptTemplate.from_messages(
            [("system", system_hint), ("human", "{question}")]
        )
        return (custom_prompt | llm | StrOutputParser()).invoke({"question": question})

    chain_output = chain.invoke({"question": question})
    print("Chain output:", chain_output)  # Debug: see raw model output
    return chain_output


if __name__ == "__main__":
    print("✅ Reactive Agent (LangChain) ready.")
    print("Model:", MODEL_ID)
    print("Type your question (or 'exit'):\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExisting.")
            break

        if not query or query.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            ans = reactive_response(query)
            print(f"Assistant: {ans}\n")
        except Exception as e:
            print("Error:", e)
            print("Tip: Ensure OPENAI_API_KEY is set and the model name is valid.\n")