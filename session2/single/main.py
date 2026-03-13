from single_agent import agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

if __name__ == "__main__":
    # --- Example 1: Using the currency tool (USD -> INR) ---
    print("\n--- Running Example 1 (Currency Tool: USD -> INR) ---")
    # using strOutputparser
    chain = agent | RunnableLambda(lambda x: x['messages'][-1]) | StrOutputParser()
    result_1 = chain.invoke({"input": "What is the current exchange rate from USD to INR?"})
    print(result_1)