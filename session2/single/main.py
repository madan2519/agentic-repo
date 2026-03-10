from single_agent import agent

if __name__ == "__main__":
    # --- Example 1: Using the currency tool (USD -> INR) ---
    print("\n--- Running Example 1 (Currency Tool: USD -> INR) ---")
    result_1 = agent.invoke({"input": "What is the current exchange rate from USD to INR?"})
    print(result_1)
    final_answer = None
    for msg in reversed(result_1["messages"]):
        if msg.type == "ai" and msg.content:
            final_answer = msg.content
            break

    print("\nFinal Answer:")
    print(final_answer)