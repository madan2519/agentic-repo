# mw_modify_output.py
from langchain.agents.middleware import after_model
from langchain_core.messages import AIMessage

@after_model
def enhance_final_output(state, runtime):
    """
    Very simple middleware:
    - Detect the final AI message
    - Wrap or modify the answer so it is visibly different
    - No complex logic — just a learning demo
    """
    try:
        msgs = state.get("messages", [])
        if not msgs:
            return None

        last = msgs[-1]

        # Only modify AI messages (the model's final output)
        if isinstance(last, AIMessage) and isinstance(last.content, str):
            enhanced = f"[Enhanced Mode] {last.content}"
            new_msg = AIMessage(content=enhanced)

            # Replace last message with enhanced version
            return {"messages": msgs[:-1] + [new_msg]}

        return None
    except Exception:
        return None