from multi_agent import ( 
    researcher_agent,
    coder_agent,
    reviewer_agent,
    MODEL_ID,
)
# -------------------------
# Shared helpers
# -------------------------
def extract_content(agent_result) -> str:
    """
    Robustly extract final assistant content from agent.invoke(...) return value
    across versions: AIMessage-like or dict with 'messages'.
    """
    content = getattr(agent_result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(agent_result, dict) and "messages" in agent_result and agent_result["messages"]:
        last = agent_result["messages"][-1]
        return getattr(last, "content", str(last))
    return str(agent_result)

# -------------------------
# Simple Supervisor Orchestrator
# -------------------------
def run_mas(user_task: str, max_fix_loops: int = 1) -> dict:
    """
    Orchestrates Researcher -> Coder -> Reviewer.
    If Reviewer says FAIL and fix budget remains, send feedback back to Coder once.
    Returns a dict with all stage outputs.
    """
    trace = {}

    # 1) Researcher
    researcher_state = researcher_agent.invoke({"messages": [{"role": "user", "content": user_task}]})
    r_spec = extract_content(researcher_state)
    trace["researcher_spec"] = r_spec

    # 2) Coder (first pass)
    coder_prompt = [
        {"role": "system", "content": f"Researcher SPEC:\n{r_spec}"},
        {"role": "user", "content": "Produce the solution. Use tools if needed."},
    ]
    coder_state = coder_agent.invoke({"messages": coder_prompt})
    c_solution = extract_content(coder_state)
    trace["coder_solution_v1"] = c_solution

    # 3) Reviewer
    review_prompt = [
        {"role": "system", "content": f"SPEC:\n{r_spec}\n\nSOLUTION:\n{c_solution}"},
        {"role": "user", "content": "Review and reply with PASS or FAIL and short feedback."},
    ]
    reviewer_state = reviewer_agent.invoke({"messages": review_prompt})
    verdict = extract_content(reviewer_state)
    trace["reviewer_verdict_v1"] = verdict

    # 4) Optional fix loop if FAIL detected
    if max_fix_loops > 0 and "FAIL" in verdict.upper():
        fix_prompt = [
            {"role": "system", "content": f"Researcher SPEC:\n{r_spec}"},
            {"role": "user", "content": f"The Reviewer reported FAIL with feedback:\n{verdict}\n\nRevise the solution and correct the issues. Use tools if needed."},
        ]
        c2_state = coder_agent.invoke({"messages": fix_prompt})
        c_solution2 = extract_content(c2_state)
        trace["coder_solution_v2"] = c_solution2

        review2_prompt = [
            {"role": "system", "content": f"SPEC:\n{r_spec}\n\nSOLUTION (revised):\n{c_solution2}"},
            {"role": "user", "content": "Review again and reply with PASS or FAIL and short feedback."},
        ]
        v2_state = reviewer_agent.invoke({"messages": review2_prompt})
        verdict2 = extract_content(v2_state)
        trace["reviewer_verdict_v2"] = verdict2

    return trace

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    print("✅ Multi-Agent System (LangChain `create_agent`) ready.")
    print("Model:", MODEL_ID)
    print("Examples:\n"
          "  • Plan a short 1-day itinerary for Chennai including weather and compute a budget: 2 meals at 300 each + cab 750.\n"
          "  • Given the task 'Compute 12*(5+2) and explain the choice of method briefly'.\n")
    try:
        while True:
            task = input("Task (or 'exit'): ").strip()
            if not task or task.lower() in {"exit", "quit"}:
                print("Bye!")
                break
            out = run_mas(task, max_fix_loops=1)
            print("\n— Researcher SPEC —\n", out.get("researcher_spec", ""))
            print("\n— Coder Solution v1 —\n", out.get("coder_solution_v1", ""))
            print("\n— Reviewer Verdict v1 —\n", out.get("reviewer_verdict_v1", ""))
            if "coder_solution_v2" in out:
                print("\n— Coder Solution v2 —\n", out.get("coder_solution_v2", ""))
                print("\n— Reviewer Verdict v2 —\n", out.get("reviewer_verdict_v2", ""))
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")