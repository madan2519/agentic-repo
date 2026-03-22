# =========================
# 6) Run a couple of examples
# =========================
from graph_flow import build_graph

async def main():
    graph = build_graph()

    print("=== Example A: Math + Weather (normal) ===")
    # state = await graph.ainvoke(
    #     {
    #         "messages": [
    #             ("user", "First compute 12*(5+2). Then tell me the weather in Chennai.")
    #         ]
    #     }
    # )
    # print("\nFinal Assistant Message:\n", state["messages"][-1].content)

    print("\n=== Example B: Ask for a brief answer to trigger conditional edge ===")
    # state2 = await graph.ainvoke(
    #     {"messages": [("user", "Briefly explain LangGraph and why it's useful. One sentence please.")]})
    # print("\nFinal Assistant Message:\n", state2["messages"][-1].content)

    print("\n=== Example C: Long answer auto‑routes to concise ===")
    state3 = await graph.ainvoke(
        {"messages": [("user", "Explain LangGraph in detail and include how nodes, edges, and state work together.")]}
    )
    print("\nFinal Assistant Message:\n", state3["messages"][-1].content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
